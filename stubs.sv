module osm #(parameter int N = 4, IN_W = 32, T_W = 32, MAXT = 2, XF = 6, EXP_FRAC = 12,
             SEED_BITS = 3, R = 16, PF = 7, LWID = 14) (
    input  logic clk, rst_n, clear, valid_in,
    input  logic [N*IN_W-1:0] row_in_flat, input logic [$clog2(N)-1:0] row_in,
    input  logic [$clog2(MAXT):0] tile_in, input logic [15:0] sm_mult, input logic [5:0] sm_shift,
    input  logic causal, input logic [$clog2(MAXT):0] qblk,
    output logic valid_out, output logic [$clog2(N)-1:0] row_out, output logic [$clog2(MAXT):0] tile_out,
    output logic [N*8-1:0] p_flat, output logic [EXP_FRAC:0] alpha_out,
    input  logic [$clog2(N)-1:0] l_rd_sel, output logic [LWID-1:0] l_rd);
    (* anyseq *) logic [N*8-1:0] ap; (* anyseq *) logic [EXP_FRAC:0] aa; (* anyseq *) logic [LWID-1:0] al;
    logic [6:0] v; logic [$clog2(N)-1:0] r [7]; logic [$clog2(MAXT):0] t [7];
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) v <= '0;
        else begin
            v <= {v[5:0], valid_in}; r[0] <= row_in; t[0] <= tile_in;
            for (int k = 1; k < 7; k++) begin r[k] <= r[k-1]; t[k] <= t[k-1]; end
        end
    assign valid_out = v[6]; assign row_out = r[6]; assign tile_out = t[6];
    assign p_flat = ap; assign alpha_out = aa; assign l_rd = al;
endmodule

module normalizer #(parameter int N = 4, ACC_W = 32, LWID = 14, XF = 6, EXP_FRAC = 12,
                    SEED_BITS = 3, R = 16, OF = 8) (
    input  logic clk, rst_n, valid_in, input logic [$clog2(N)-1:0] row_in,
    input  logic [LWID-1:0] l_in, input logic [N*ACC_W-1:0] o_in_flat,
    output logic valid_out, output logic [$clog2(N)-1:0] row_out, output logic [N*ACC_W-1:0] o_out_flat);
    (* anyseq *) logic [N*ACC_W-1:0] ao;
    logic [8:0] v; logic [$clog2(N)-1:0] r [9];
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) v <= '0;
        else begin
            v <= {v[7:0], valid_in}; r[0] <= row_in;
            for (int k = 1; k < 9; k++) r[k] <= r[k-1];
        end
    assign valid_out = v[8]; assign row_out = r[8]; assign o_out_flat = ao;
endmodule
