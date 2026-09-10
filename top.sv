module attn_core_top #(
    parameter N      = 8,
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                   ld_weight_valid,
    input  logic [$clog2(N)-1:0]   ld_weight_row,
    input  logic [N*DATA_W-1:0]    ld_weight_data_flat,

    input  logic                   act_valid,
    input  logic [N*DATA_W-1:0]    act_row_flat,
    output logic                   act_ready,

    input  logic [N-1:0]             row_en_override,
    input  logic signed [DATA_W-1:0] sparsity_thresh,

    input  logic                     gate_sel,
    input  logic [15:0]              lif_thresh,

    output logic                     res_valid,
    output logic [N*ACC_W-1:0]       res_row_flat
);

    logic rst_n_arr, rst_n_fifo, rst_n_lif, rst_n_misc;
    BUF_X16 u_rst_buf_arr  (.A(rst_n), .Z(rst_n_arr));
    BUF_X16 u_rst_buf_fifo (.A(rst_n), .Z(rst_n_fifo));
    BUF_X16 u_rst_buf_lif  (.A(rst_n), .Z(rst_n_lif));
    BUF_X16 u_rst_buf_misc (.A(rst_n), .Z(rst_n_misc));

    wire                    weight_load    = ld_weight_valid;
    wire [$clog2(N)-1:0]    weight_row_sel = ld_weight_row;
    wire [N*DATA_W-1:0]     weight_row_in_flat = ld_weight_data_flat;

    logic fifo_full, fifo_empty;
    logic [N*DATA_W-1:0] fifo_rd_row_flat;

    row_fifo #(.N(N), .DATA_W(DATA_W), .DEPTH(16)) u_fifo (
        .clk         (clk),
        .rst_n       (rst_n_fifo),
        .wr_en       (act_valid),
        .wr_row_flat (act_row_flat),
        .full        (fifo_full),
        .rd_en       (!fifo_empty),
        .rd_row_flat (fifo_rd_row_flat),
        .empty       (fifo_empty)
    );

    assign act_ready = !fifo_full;

    logic [N*DATA_W-1:0] a_read_row_flat;
    assign a_read_row_flat = fifo_empty ? '0 : fifo_rd_row_flat;

    logic [N-1:0] row_active_static, row_active_lif, row_active_pre;

    sparsity_detect #(.N(N), .DATA_W(DATA_W)) u_sparsity (
        .row_in_flat (a_read_row_flat),
        .thresh      (sparsity_thresh),
        .row_active  (row_active_static)
    );

    lif_gate #(.N(N), .DATA_W(DATA_W), .POT_W(16), .LEAK_SHIFT(3)) u_lif (
        .clk         (clk),
        .rst_n       (rst_n_lif),
        .update      (!fifo_empty),
        .row_in_flat (a_read_row_flat),
        .thresh      (lif_thresh),
        .fire        (row_active_lif)
    );

    assign row_active_pre = gate_sel ? row_active_lif : row_active_static;

    logic [N*DATA_W-1:0] a_col_in_flat;
    logic [N-1:0]        row_en_skewed;
    genvar gr;
    generate
        for (gr = 0; gr < N; gr++) begin : SKEW
            wire signed [DATA_W-1:0] row_in = a_read_row_flat[gr*DATA_W +: DATA_W];
            wire                     en_in  = row_active_pre[gr];
            if (gr == 0) begin
                assign a_col_in_flat[0 +: DATA_W] = row_in;
                assign row_en_skewed[0]           = en_in;
            end else begin
                logic signed [DATA_W-1:0] sk    [gr];
                logic                     sk_en [gr];
                integer si;
                always_ff @(posedge clk or negedge rst_n_misc) begin
                    if (!rst_n_misc) begin
                        for (si = 0; si < gr; si++) begin
                            sk[si]    <= '0;
                            sk_en[si] <= 1'b0;
                        end
                    end else begin
                        sk[0]    <= row_in;
                        sk_en[0] <= en_in;
                        for (si = 1; si < gr; si++) begin
                            sk[si]    <= sk[si-1];
                            sk_en[si] <= sk_en[si-1];
                        end
                    end
                end
                assign a_col_in_flat[gr*DATA_W +: DATA_W] = sk[gr-1];
                assign row_en_skewed[gr]                  = sk_en[gr-1];
            end
        end
    endgenerate

    wire [N-1:0] row_clk_en = row_en_skewed & row_en_override;

    mac_array_exact_only_v2 #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_array (
        .clk                 (clk),
        .rst_n               (rst_n_arr),
        .weight_load         (weight_load),
        .weight_row_sel      (weight_row_sel),
        .weight_row_in_flat  (weight_row_in_flat),
        .row_clk_en          (row_clk_en),
        .a_col_in_flat       (a_col_in_flat),
        .psum_col_out_flat   (res_row_flat)
    );

    localparam LAT = N;
    logic [LAT-1:0] valid_sr;
    always_ff @(posedge clk or negedge rst_n_misc) begin
        if (!rst_n_misc)
            valid_sr <= '0;
        else
            valid_sr <= {valid_sr[LAT-2:0], !fifo_empty};
    end
    assign res_valid = valid_sr[LAT-1];

endmodule
