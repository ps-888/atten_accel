module mac_array_exact #(
    parameter N       = 8,
    parameter DATA_W  = 8,
    parameter ACC_W   = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                       weight_load,
    input  logic [$clog2(N)-1:0]       weight_row_sel,
    input  logic [N*DATA_W-1:0]        weight_row_in_flat,

    input  logic [N-1:0]               row_clk_en,

    input  logic [N*DATA_W-1:0]        a_col_in_flat,

    output logic [N*ACC_W-1:0]         psum_col_out_flat
);

    logic signed [ACC_W-1:0] psum_net [N+1][N];

    genvar r, c;
    generate
        for (c = 0; c < N; c++) begin : ZERO_TOP
            assign psum_net[0][c] = '0;
        end
    endgenerate

    logic [N-1:0] rst_n_row_buf;
    generate
        for (r = 0; r < N; r++) begin : RSTBUF
            BUF_X16 u_rst_buf (.A(rst_n), .Z(rst_n_row_buf[r]));
        end
    endgenerate

    generate
        for (r = 0; r < N; r++) begin : ROW
            for (c = 0; c < N; c++) begin : COL
                wire this_weight_load = weight_load && (weight_row_sel == r);
                logic signed [DATA_W-1:0] a_out_unused;

                pe_exact_only #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
                    .clk         (clk),
                    .rst_n       (rst_n_row_buf[r]),
                    .weight_load (this_weight_load),
                    .weight_in   (weight_row_in_flat[c*DATA_W +: DATA_W]),
                    .clk_en      (row_clk_en[r]),
                    .a_in        (a_col_in_flat[r*DATA_W +: DATA_W]),
                    .psum_in     (psum_net[r][c]),
                    .a_out       (a_out_unused),
                    .psum_out    (psum_net[r+1][c])
                );
            end
        end
    endgenerate

    generate
        for (c = 0; c < N; c++) begin : OUT
            assign psum_col_out_flat[c*ACC_W +: ACC_W] = psum_net[N][c];
        end
    endgenerate

endmodule
