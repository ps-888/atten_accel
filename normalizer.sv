module normalizer #(
    parameter int N = 8, parameter int ACC_W = 32, parameter int LWID = 14,
    parameter int XF = 6, parameter int EXP_FRAC = 12, parameter int SEED_BITS = 3,
    parameter int R = 16, parameter int OF = 8
) (
    input  logic                    clk, rst_n,
    input  logic                    valid_in,
    input  logic [$clog2(N)-1:0]    row_in,
    input  logic [LWID-1:0]         l_in,
    input  logic [N*ACC_W-1:0]      o_in_flat,
    output logic                    valid_out,
    output logic [$clog2(N)-1:0]    row_out,
    output logic [N*ACC_W-1:0]      o_out_flat
);
    `include "softmax_hp_lut.svh"
    localparam int CW = $clog2(N), PSW = $clog2(LWID) + 1;
    localparam int QW = (ACC_W + 3) / 4;
    localparam int ST = 9;
    localparam int SHD = 6;

    logic [ST-1:0]  v;
    logic [CW-1:0]  rw [ST];
    logic           zl [ST];
    logic           sg [ST][N];
    logic [PSW-1:0]   pos [7];
    logic [LWID-1:0]  mn [4];
    logic [R:0]       y [5];
    logic [R+1:0]     tm2, tm3;
    logic [ACC_W-1:0] mag [6][N];
    logic [QW+R:0]    pp [N][4];
    logic [ACC_W+R:0] pr [N], q5 [N];
    logic [6:0]       shm1;

    function automatic logic [R+1:0] two_minus_my(input logic [LWID-1:0] m, input logic [R:0] yy);
        logic [LWID+R:0] my;
        my = (LWID+R+1)'(m) * (LWID+R+1)'(yy);
        return (R+2)'(2 << R) - (R+2)'(my >> (LWID-1));
    endfunction
    function automatic logic [R:0] y_times(input logic [R:0] yy, input logic [R+1:0] t);
        logic [2*R+2:0] p;
        p = (2*R+3)'(yy) * (2*R+3)'(t);
        return (R+1)'(p >> R);
    endfunction

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) v <= '0;
        else begin
            v <= {v[ST-2:0], valid_in};
            for (int k = 1; k < ST; k++) begin
                rw[k] <= rw[k-1]; zl[k] <= zl[k-1];
                for (int i = 0; i < N; i++) sg[k][i] <= sg[k-1][i];
            end
            for (int k = 1; k < 7; k++) pos[k] <= pos[k-1];
            for (int k = 1; k < 6; k++) for (int i = 0; i < N; i++) mag[k][i] <= mag[k-1][i];
            begin
                logic [PSW-1:0] p; logic [LWID-1:0] m;
                p = '0; for (int b = 0; b < LWID; b++) if (l_in[b]) p = PSW'(b);
                m = l_in << (PSW'(LWID-1) - p);
                rw[0] <= row_in; zl[0] <= (l_in == '0);
                pos[0] <= p; mn[0] <= m; y[0] <= recip_seed_lut(m[LWID-2 -: SEED_BITS]);
                for (int i = 0; i < N; i++) begin
                    logic signed [ACC_W-1:0] o; o = $signed(o_in_flat[i*ACC_W +: ACC_W]);
                    sg[0][i] <= o[ACC_W-1]; mag[0][i] <= o[ACC_W-1] ? ACC_W'(-o) : ACC_W'(o);
                end
            end
            tm2 <= two_minus_my(mn[0], y[0]); mn[1] <= mn[0]; y[1] <= y[0];
            y[2] <= y_times(y[1], tm2);        mn[2] <= mn[1];
            tm3 <= two_minus_my(mn[2], y[2]); mn[3] <= mn[2]; y[3] <= y[2];
            y[4] <= y_times(y[3], tm3);
            for (int i = 0; i < N; i++)
                for (int q = 0; q < 4; q++) begin
                    logic [ACC_W+3:0] mx; mx = (ACC_W+4)'(mag[4][i]);
                    pp[i][q] <= (QW+R+1)'(mx[q*QW +: QW]) * (QW+R+1)'(y[4]);
                end
            shm1 <= 7'(pos[5]) + 7'(R - OF - 1);
            for (int i = 0; i < N; i++)
                pr[i] <= (ACC_W+R+1)'(pp[i][0]) + ((ACC_W+R+1)'(pp[i][1]) << QW)
                       + ((ACC_W+R+1)'(pp[i][2]) << (2*QW)) + ((ACC_W+R+1)'(pp[i][3]) << (3*QW));
            for (int i = 0; i < N; i++) q5[i] <= pr[i] >> shm1;
            row_out <= rw[ST-2];
            for (int i = 0; i < N; i++) begin
                logic [ACC_W+R:0] q;
                q = (q5[i] + 1'b1) >> 1;
                o_out_flat[i*ACC_W +: ACC_W] <= zl[ST-2] ? '0 : (sg[ST-2][i] ? -ACC_W'(q) : ACC_W'(q));
            end
        end
    assign valid_out = v[ST-1];
endmodule
