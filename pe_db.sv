module pe_db #(parameter int DATA_W = 8, parameter int ACC_W = 32) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     ld,
    input  logic signed [DATA_W-1:0] w_in,
    input  logic                     swap,
    input  logic signed [DATA_W-1:0] a_in,
    input  logic signed [ACC_W-1:0]  psum_in,
    output logic signed [ACC_W-1:0]  psum_out
);
    logic signed [DATA_W-1:0] w_act, w_sh;
    wire  signed [DATA_W-1:0] w_eff = swap ? w_sh : w_act;
    wire                      do_mac = (a_in != '0);
    wire  signed [2*DATA_W-1:0] prod = a_in * w_eff;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_act <= '0; w_sh <= '0; psum_out <= '0;
        end else begin
            if (ld)   w_sh  <= w_in;
            if (swap) w_act <= w_sh;
            psum_out <= do_mac ? psum_in + ACC_W'(prod) : psum_in;
        end
    end
endmodule
