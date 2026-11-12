module osm #(
    parameter int N = 8, parameter int IN_W = 32,
    parameter int MAXT = 8, parameter int XF = 6, parameter int EXP_FRAC = 12,
    parameter int SEED_BITS = 3, parameter int R = 16, parameter int PF = 7,
    parameter int LWID = 14
) (
    input  logic                       clk, rst_n,
    input  logic                       clear,
    input  logic                       valid_in,
    input  logic [N*IN_W-1:0]          row_in_flat,
    input  logic [$clog2(N)-1:0]       row_in,
    input  logic [$clog2(MAXT):0]      tile_in,
    input  logic [15:0]                sm_mult,
    input  logic [5:0]                 sm_shift,
    input  logic                       causal,
    input  logic [$clog2(MAXT):0]      qblk,
    output logic                       valid_out,
    output logic [$clog2(N)-1:0]       row_out,
    output logic [$clog2(MAXT):0]      tile_out,
    output logic [N*8-1:0]             p_flat,
    output logic [EXP_FRAC:0]          alpha_out,
    input  logic [$clog2(N)-1:0]       l_rd_sel,
    output logic [LWID-1:0]            l_rd
);
    `include "softmax_hp_lut.svh"
    localparam int CW = $clog2(N), TBW = $clog2(MAXT) + 1, EW = EXP_FRAC + 1;
    localparam int DW = IN_W + 1;
    localparam int PW = DW + 16;
    localparam int DLIM = (EXP_FRAC + 1) << XF;
    localparam int LW = $clog2(DLIM + 1);
    initial if (N < 4) $fatal(1, "osm requires N >= 4 (row-state hazard distance)");

    logic signed [IN_W-1:0] m_st [N];
    logic                   mv_st [N];
    logic [LWID-1:0]        l_st [N];
    assign l_rd = l_st[l_rd_sel];

    function automatic logic [EW-1:0] exp2n(input logic [LW-1:0] d);
        logic [LW-1:0] ip;
        ip = d >> XF;
        return (ip > LW'(EXP_FRAC)) ? '0 : (exp2_frac_lut(d[XF-1:0]) >> ip);
    endfunction

    logic o1_v; logic [CW-1:0] o1_r; logic [TBW-1:0] o1_t;
    logic signed [IN_W-1:0] o1_s [N], o1_mx; logic [N-1:0] o1_m; logic o1_any;
    localparam logic signed [IN_W-1:0] MINV = {1'b1, {(IN_W-1){1'b0}}};
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) o1_v <= 1'b0;
        else begin
            logic signed [IN_W-1:0] lv [2*N]; logic [N-1:0] mk;
            o1_v <= valid_in; o1_r <= row_in; o1_t <= tile_in;
            lv[0] = '0;
            for (int i = 0; i < N; i++) begin
                mk[i] = causal && ((32'(tile_in) * N + i) > (32'(qblk) * N + 32'(row_in)));
                o1_s[i] <= $signed(row_in_flat[i*IN_W +: IN_W]);
                lv[N+i] = mk[i] ? MINV : $signed(row_in_flat[i*IN_W +: IN_W]);
            end
            for (int k = N-1; k >= 1; k--) lv[k] = (lv[2*k] > lv[2*k+1]) ? lv[2*k] : lv[2*k+1];
            o1_m <= mk; o1_mx <= lv[1]; o1_any <= ~&mk;
        end

    logic o2_v; logic [CW-1:0] o2_r; logic [TBW-1:0] o2_t;
    logic [DW-1:0] o2_d [N], o2_da; logic [N-1:0] o2_m; logic o2_mvold, o2_mvnew;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            o2_v <= 1'b0;
            for (int i = 0; i < N; i++) mv_st[i] <= 1'b0;
        end else begin
            logic signed [IN_W-1:0] mo, mn; logic mvo;
            o2_v <= o1_v; o2_r <= o1_r; o2_t <= o1_t; o2_m <= o1_m;
            mo = m_st[o1_r]; mvo = mv_st[o1_r];
            mn = !o1_any ? mo : (!mvo ? o1_mx : ((o1_mx > mo) ? o1_mx : mo));
            for (int i = 0; i < N; i++)
                o2_d[i] <= o1_m[i] ? '0 : DW'(DW'(mn) - DW'(o1_s[i]));
            o2_da <= mvo ? DW'(DW'(mn) - DW'(mo)) : '0;
            o2_mvold <= mvo; o2_mvnew <= mvo | o1_any;
            if (clear) for (int i = 0; i < N; i++) mv_st[i] <= 1'b0;
            else if (o1_v) begin m_st[o1_r] <= mn; mv_st[o1_r] <= mvo | o1_any; end
        end

    localparam int DQ = (DW + 3) / 4;
    logic o3a_v; logic [CW-1:0] o3a_r; logic [TBW-1:0] o3a_t; logic [N-1:0] o3a_m; logic o3a_mvold, o3a_mvnew;
    logic [DQ+15:0] o3a_pp [N+1][4];
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) o3a_v <= 1'b0;
        else begin
            o3a_v <= o2_v; o3a_r <= o2_r; o3a_t <= o2_t; o3a_m <= o2_m;
            o3a_mvold <= o2_mvold; o3a_mvnew <= o2_mvnew;
            for (int i = 0; i <= N; i++) begin
                logic [4*DQ-1:0] d; d = (4*DQ)'((i == N) ? o2_da : o2_d[i]);
                for (int q = 0; q < 4; q++)
                    o3a_pp[i][q] <= (DQ+16)'(d[q*DQ +: DQ]) * (DQ+16)'(sm_mult);
            end
        end

    logic o3_v; logic [CW-1:0] o3_r; logic [TBW-1:0] o3_t; logic [N-1:0] o3_m; logic o3_mvold, o3_mvnew;
    logic [PW-1:0] o3_p [N], o3_pa;
    function automatic logic [PW-1:0] sum4(input logic [DQ+15:0] a0, a1, a2, a3);
        return PW'(a0) + (PW'(a1) << DQ) + (PW'(a2) << (2*DQ)) + (PW'(a3) << (3*DQ));
    endfunction
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) o3_v <= 1'b0;
        else begin
            o3_v <= o3a_v; o3_r <= o3a_r; o3_t <= o3a_t; o3_m <= o3a_m;
            o3_mvold <= o3a_mvold; o3_mvnew <= o3a_mvnew;
            for (int i = 0; i < N; i++) o3_p[i] <= sum4(o3a_pp[i][0], o3a_pp[i][1], o3a_pp[i][2], o3a_pp[i][3]);
            o3_pa <= sum4(o3a_pp[N][0], o3a_pp[N][1], o3a_pp[N][2], o3a_pp[N][3]);
        end

    logic o4_v; logic [CW-1:0] o4_r; logic [TBW-1:0] o4_t; logic [N-1:0] o4_m; logic o4_mvold, o4_mvnew;
    logic [LW-1:0] o4_d [N], o4_da;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) o4_v <= 1'b0;
        else begin
            logic [PW-1:0] q;
            o4_v <= o3_v; o4_r <= o3_r; o4_t <= o3_t; o4_m <= o3_m;
            o4_mvold <= o3_mvold; o4_mvnew <= o3_mvnew;
            for (int i = 0; i < N; i++) begin
                q = o3_p[i] >> sm_shift;
                o4_d[i] <= (q > PW'(DLIM)) ? LW'(DLIM) : LW'(q);
            end
            q = o3_pa >> sm_shift;
            o4_da <= (q > PW'(DLIM)) ? LW'(DLIM) : LW'(q);
        end

    logic o5_v; logic [CW-1:0] o5_r; logic [TBW-1:0] o5_t;
    logic [7:0] o5_p [N]; logic [EW-1:0] o5_a; logic [LWID-1:0] o5_sum;
    localparam int PSH = EXP_FRAC - PF;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) o5_v <= 1'b0;
        else begin
            logic [EW-1:0] e; logic [EW:0] pr; logic [LWID-1:0] sm;
            o5_v <= o4_v; o5_r <= o4_r; o5_t <= o4_t;
            sm = '0;
            for (int i = 0; i < N; i++) begin
                e  = (o4_m[i] || !o4_mvnew) ? '0 : exp2n(o4_d[i]);
                pr = ((EW+1)'(e) + (EW+1)'(1 << (PSH-1))) >> PSH;
                if (pr > (EW+1)'((1 << PF) - 1)) pr = (EW+1)'((1 << PF) - 1);
                o5_p[i] <= 8'(pr);
                sm = sm + LWID'(pr);
            end
            o5_sum <= sm;
            o5_a   <= o4_mvold ? exp2n(o4_da) : '0;
        end

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) valid_out <= 1'b0;
        else begin
            valid_out <= o5_v; row_out <= o5_r; tile_out <= o5_t; alpha_out <= o5_a;
            for (int i = 0; i < N; i++) p_flat[i*8 +: 8] <= o5_p[i];
            if (clear) for (int i = 0; i < N; i++) l_st[i] <= '0;
            else if (o5_v)
                l_st[o5_r] <= LWID'(((LWID+EW)'(l_st[o5_r]) * (LWID+EW)'(o5_a)) >> EXP_FRAC) + o5_sum;
        end
endmodule
