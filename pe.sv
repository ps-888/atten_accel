module pe_exact_only #(
    parameter DATA_W = 8,
    parameter ACC_W   = 32
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  weight_load,
    input  logic signed [DATA_W-1:0] weight_in,

    input  logic                  clk_en,

    input  logic signed [DATA_W-1:0] a_in,
    input  logic signed [ACC_W-1:0]  psum_in,

    output logic signed [DATA_W-1:0] a_out,
    output logic signed [ACC_W-1:0]  psum_out
);

    logic signed [DATA_W-1:0] weight_r;
    logic signed [DATA_W-1:0] a_r;
    logic signed [ACC_W-1:0]  psum_r;

    wire a_is_zero = (a_in == '0);
    wire do_mac    = clk_en && !a_is_zero;

    logic signed [2*DATA_W-1:0] mult_result;
    assign mult_result = a_in * weight_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_r <= '0;
            a_r      <= '0;
            psum_r   <= '0;
        end else begin
            if (weight_load)
                weight_r <= weight_in;

            a_r <= a_in;

            if (do_mac)
                psum_r <= psum_in + mult_result;
            else
                psum_r <= psum_in;
        end
    end

    assign a_out    = a_r;
    assign psum_out = psum_r;

endmodule
