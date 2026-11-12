module attn_core_db #(parameter int N = 8, parameter int DATA_W = 8, parameter int ACC_W = 32) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     ld_valid,
    input  logic [$clog2(N)-1:0]     ld_row,
    input  logic [N*DATA_W-1:0]      ld_data,
    input  logic                     iss_swap,
    input  logic [N*DATA_W-1:0]      iss_data,
    output logic [N*ACC_W-1:0]       res_row_flat,
    output logic [N-1:0]             swap_rows
);
    logic signed [ACC_W-1:0] psum [N+1][N];
    logic signed [DATA_W-1:0] a_sk [N];
    logic                     sw_sk [N];

    genvar r, c;
    generate for (r = 0; r < N; r++) begin : SKEW
        if (r == 0) begin : R0
            assign a_sk[0]  = iss_data[0 +: DATA_W];
            assign sw_sk[0] = iss_swap;
        end else begin : RN
            logic signed [DATA_W-1:0] d [r];
            logic                     s [r];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < r; i++) begin d[i] <= '0; s[i] <= 1'b0; end
                end else begin
                    d[0] <= iss_data[r*DATA_W +: DATA_W]; s[0] <= (r == 0) ? 1'b0 : iss_swap;
                    for (int i = 1; i < r; i++) begin d[i] <= d[i-1]; s[i] <= s[i-1]; end
                end
            end
            assign a_sk[r]  = d[r-1];
            assign sw_sk[r] = s[r-1];
        end
    end endgenerate

    generate
        for (c = 0; c < N; c++) begin : TOP0
            assign psum[0][c] = '0;
        end
        for (r = 0; r < N; r++) begin : ROW
            for (c = 0; c < N; c++) begin : COL
                pe_db #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
                    .clk, .rst_n,
                    .ld(ld_valid && (ld_row == r)), .w_in(ld_data[c*DATA_W +: DATA_W]),
                    .swap(sw_sk[r]), .a_in(a_sk[r]),
                    .psum_in(psum[r][c]), .psum_out(psum[r+1][c]));
            end
        end
        for (r = 0; r < N; r++) begin : SWO
            assign swap_rows[r] = sw_sk[r];
        end
        for (c = 0; c < N; c++) begin : OUT
            assign res_row_flat[c*ACC_W +: ACC_W] = psum[N][c];
        end
    endgenerate
endmodule
