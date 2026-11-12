module attn_tiled #(
    parameter int N      = 8,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32,
    parameter int MAXT   = 8,
    parameter int OF     = 8
) (
    input  logic clk, rst_n,

    input  logic                         ld_q_valid,
    input  logic [$clog2(N)-1:0]         ld_q_row,
    input  logic [N*DATA_W-1:0]          ld_q_data,
    input  logic                         ld_k_valid,
    input  logic [$clog2(MAXT*N)-1:0]    ld_k_row,
    input  logic [N*DATA_W-1:0]          ld_k_data,
    input  logic                         ld_v_valid,
    input  logic [$clog2(MAXT*N)-1:0]    ld_v_row,
    input  logic [N*DATA_W-1:0]          ld_v_data,

    input  logic                         cfg_valid,
    input  logic [$clog2(MAXT):0]        cfg_ntiles,
    input  logic                         cfg_causal,
    input  logic [$clog2(MAXT):0]        cfg_qblk,
    input  logic [$clog2(MAXT):0]        cfg_tile_lo,
    input  logic [15:0]                  cfg_sm_mult,
    input  logic [5:0]                   cfg_sm_shift,
    input  logic [DATA_W-1:0]            cfg_th_qk,
    input  logic [DATA_W-1:0]            cfg_th_pv,

    input  logic                         start,
    output logic                         busy,
    output logic                         done,

    input  logic [$clog2(N)-1:0]         out_row_sel,
    output logic [N*ACC_W-1:0]           out_row_flat,
    output logic [31:0]                  dbg_l,

    output logic [31:0]                  perf_cycles,
    output logic [31:0]                  perf_mac_qk,
    output logic [31:0]                  perf_mac_pv,
    output logic [31:0]                  perf_rows_issued,
    output logic [31:0]                  perf_tiles
);
    localparam int CW = $clog2(N), TBW = $clog2(MAXT) + 1, MAXL = MAXT * N;
    localparam int SW = $clog2(2 * MAXT) + 1;
    localparam int EF = 12, PF = 7;
    localparam int LWID = PF + $clog2(MAXL) + 1;
    localparam int SCW = 2*DATA_W + CW;
    localparam int NAW = LWID + DATA_W + 1;

    logic signed [DATA_W-1:0] q_mem [N][N];
    logic signed [DATA_W-1:0] k_mem [MAXL][N];
    logic signed [DATA_W-1:0] v_mem [MAXL][N];
    logic        [7:0]        pbuf  [2][N][N];
    logic                     prdy  [2][N];
    logic        [EF:0]       alpha [2][N];
    logic signed [NAW-1:0]    oacc  [N][N];
    logic signed [ACC_W-1:0]  ofin  [N][N];

    logic [TBW-1:0] ntiles_r, qblk_r, te, tlo_r, tlo;
    logic           causal_r;
    logic [15:0]    mult_r; logic [5:0] shift_r;
    logic [DATA_W-1:0] thqk_r, thpv_r;

    always_ff @(posedge clk) begin
        if (ld_q_valid) for (int c = 0; c < N; c++) q_mem[ld_q_row][c] <= $signed(ld_q_data[c*DATA_W +: DATA_W]);
        if (ld_k_valid) for (int c = 0; c < N; c++) k_mem[ld_k_row][c] <= $signed(ld_k_data[c*DATA_W +: DATA_W]);
        if (ld_v_valid) for (int c = 0; c < N; c++) v_mem[ld_v_row][c] <= $signed(ld_v_data[c*DATA_W +: DATA_W]);
    end

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            ntiles_r <= TBW'(1); qblk_r <= '0; causal_r <= 1'b0; tlo_r <= '0;
            mult_r <= 16'h8000; shift_r <= 6'd20; thqk_r <= '0; thpv_r <= '0;
        end else if (cfg_valid && !busy) begin
            ntiles_r <= cfg_ntiles; qblk_r <= cfg_qblk; causal_r <= cfg_causal; tlo_r <= cfg_tile_lo;
            mult_r <= cfg_sm_mult; shift_r <= cfg_sm_shift; thqk_r <= cfg_th_qk; thpv_r <= cfg_th_pv;
        end

    function automatic logic [TBW:0] slot_dec(input logic [SW-1:0] s, input logic [TBW-1:0] t);
        if (s == 0)                         return {1'b0, TBW'(tlo)};
        else if (s == SW'(2*t - 1))         return {1'b1, TBW'(tlo + t - 1)};
        else if (s[0])                      return {1'b0, TBW'(tlo + ((s + 1) >> 1))};
        else                                return {1'b1, TBW'(tlo + (s >> 1) - 1)};
    endfunction

    function automatic logic signed [DATA_W-1:0] gate(input logic signed [DATA_W-1:0] a,
                                                       input logic [DATA_W-1:0] th);
        logic signed [DATA_W-1:0] t; t = $signed(th);
        return ((a > t) || (a < -t)) ? a : '0;
    endfunction

    logic                core_ld_valid; logic [CW-1:0] core_ld_row; logic [N*DATA_W-1:0] core_ld_data;
    logic                iss_swap;      logic [N*DATA_W-1:0] iss_data;
    logic [N*ACC_W-1:0]  res_row;
    logic [N-1:0]        swap_rows;
    attn_core_db #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_core (
        .clk, .rst_n, .ld_valid(core_ld_valid), .ld_row(core_ld_row), .ld_data(core_ld_data),
        .iss_swap, .iss_data, .res_row_flat(res_row), .swap_rows(swap_rows));

    logic          tg_v [N+1]; logic tg_p [N+1];
    logic [TBW-1:0] tg_t [N+1]; logic [CW-1:0] tg_r [N+1];
    logic          iss_v, iss_p; logic [TBW-1:0] iss_t; logic [CW-1:0] iss_r;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) for (int k = 0; k <= N; k++) tg_v[k] <= 1'b0;
        else begin
            tg_v[0] <= iss_v; tg_p[0] <= iss_p; tg_t[0] <= iss_t; tg_r[0] <= iss_r;
            for (int k = 1; k <= N; k++) begin
                tg_v[k] <= tg_v[k-1]; tg_p[k] <= tg_p[k-1]; tg_t[k] <= tg_t[k-1]; tg_r[k] <= tg_r[k-1];
            end
        end
    wire          rv = tg_v[N-1], rp = tg_p[N-1];
    wire [TBW-1:0] rt = tg_t[N-1];
    wire [CW-1:0]  rr_o = tg_r[N-1];

    logic                osm_v_raw; logic [CW-1:0] osm_r_raw; logic [TBW-1:0] osm_t_raw;
    logic [N*8-1:0]      osm_p_raw; logic [EF:0] osm_a_raw;
    logic                osm_v; logic [CW-1:0] osm_r; logic [TBW-1:0] osm_t;
    logic [N*8-1:0]      osm_p; logic [EF:0] osm_a;
    logic [N-1:0]        osm_we [2];
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) begin osm_v <= 1'b0; osm_we[0] <= '0; osm_we[1] <= '0; end
        else begin
            osm_v <= osm_v_raw; osm_r <= osm_r_raw; osm_t <= osm_t_raw; osm_p <= osm_p_raw; osm_a <= osm_a_raw;
            for (int b = 0; b < 2; b++)
                for (int r = 0; r < N; r++)
                    osm_we[b][r] <= osm_v_raw && (osm_t_raw[0] == b[0]) && (osm_r_raw == CW'(r));
        end
    logic [CW-1:0]       l_sel; logic [LWID-1:0] l_val;
    logic                clear;
    logic [N*SCW-1:0]    s_narrow;
    always_comb for (int c = 0; c < N; c++) s_narrow[c*SCW +: SCW] = res_row[c*ACC_W +: SCW];
    osm #(.N(N), .IN_W(SCW), .MAXT(MAXT), .LWID(LWID), .PF(PF), .EXP_FRAC(EF)) u_osm (
        .clk, .rst_n, .clear,
        .valid_in(rv && !rp), .row_in_flat(s_narrow), .row_in(rr_o), .tile_in(rt),
        .sm_mult(mult_r), .sm_shift(shift_r), .causal(causal_r), .qblk(qblk_r),
        .valid_out(osm_v_raw), .row_out(osm_r_raw), .tile_out(osm_t_raw), .p_flat(osm_p_raw), .alpha_out(osm_a_raw),
        .l_rd_sel(l_sel), .l_rd(l_val));

    logic nm_vi, nm_vo; logic [CW-1:0] nm_ri, nm_ro; logic [N*NAW-1:0] nm_oi, nm_oo;
    normalizer #(.N(N), .ACC_W(NAW), .LWID(LWID), .OF(OF)) u_norm (
        .clk, .rst_n, .valid_in(nm_vi), .row_in(nm_ri), .l_in(l_val), .o_in_flat(nm_oi),
        .valid_out(nm_vo), .row_out(nm_ro), .o_out_flat(nm_oo));

    typedef enum logic [2:0] { ST_IDLE, ST_RUN, ST_DRAIN, ST_NORM, ST_NWAIT, ST_DONE } st_t;
    st_t state;
    logic [SW-1:0]  s, ld_slot;
    logic [CW:0]    rr, ld_r, nctr, ncnt;
    logic           ld_act, sh_full;
    logic [SW-1:0]  sh_slot;
    logic [15:0]    pdrained;
    logic           oa_v; logic [CW-1:0] oa_r;
    logic signed [NAW+EF+1:0] oa_prod [N];
    logic signed [NAW-1:0]    oa_res  [N];
    logic           pc_v, pc_p; logic [CW:0] pc_nz;

    wire [TBW:0]   cur  = slot_dec(s, te);
    wire           cur_p = cur[TBW];
    wire [TBW-1:0] cur_t = cur[TBW-1:0];
    wire [SW-1:0]  nslots = SW'(2 * te);

    function automatic logic [N*DATA_W-1:0] ld_row_data(input logic [SW-1:0] sl, input logic [CW:0] r);
        logic [TBW:0] d; logic [N*DATA_W-1:0] o;
        d = slot_dec(sl, te);
        for (int c = 0; c < N; c++)
            o[c*DATA_W +: DATA_W] = d[TBW] ? v_mem[32'(d[TBW-1:0]) * N + 32'(r[CW-1:0])][c]
                                           : k_mem[32'(d[TBW-1:0]) * N + c][r[CW-1:0]];
        return o;
    endfunction

    logic [N*DATA_W-1:0] cand; logic [CW:0] cand_nz;
    always_comb begin
        cand_nz = '0;
        for (int c = 0; c < N; c++) begin
            logic signed [DATA_W-1:0] a;
            a = cur_p ? gate($signed(pbuf[cur_t[0]][rr[CW-1:0]][c]), thpv_r) : gate(q_mem[rr[CW-1:0]][c], thqk_r);
            cand[c*DATA_W +: DATA_W] = a;
            if (a != '0) cand_nz = cand_nz + 1'b1;
        end
    end
    wire can_issue = (state == ST_RUN) && (s < nslots) &&
                     ((rr == 0) ? (sh_full && sh_slot == s) : 1'b1) &&
                     (!cur_p || prdy[cur_t[0]][rr[CW-1:0]]);

    assign clear = (state == ST_IDLE) && start;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE; busy <= 1'b0; done <= 1'b0;
            s <= '0; rr <= '0; ld_act <= 1'b0; ld_r <= '0; ld_slot <= '0; sh_full <= 1'b0; sh_slot <= '0;
            core_ld_valid <= 1'b0; core_ld_row <= '0; core_ld_data <= '0;
            iss_v <= 1'b0; iss_swap <= 1'b0; iss_data <= '0; iss_p <= 1'b0; iss_t <= '0; iss_r <= '0;
            nm_vi <= 1'b0; nctr <= '0; ncnt <= '0; pdrained <= '0; te <= TBW'(1); tlo <= '0; oa_v <= 1'b0; pc_v <= 1'b0;
            perf_cycles <= '0; perf_mac_qk <= '0; perf_mac_pv <= '0; perf_rows_issued <= '0; perf_tiles <= '0;
            for (int b = 0; b < 2; b++) for (int r = 0; r < N; r++) prdy[b][r] <= 1'b0;
        end else begin
            done <= 1'b0; core_ld_valid <= 1'b0; iss_v <= 1'b0; iss_swap <= 1'b0; iss_data <= '0; nm_vi <= 1'b0;
            pc_v <= can_issue; pc_p <= cur_p; pc_nz <= cand_nz;
            if (pc_v) begin
                if (pc_p) perf_mac_pv <= perf_mac_pv + 32'(pc_nz) * N;
                else      perf_mac_qk <= perf_mac_qk + 32'(pc_nz) * N;
            end
            if (busy) perf_cycles <= perf_cycles + 1;

            case (state)
            ST_IDLE: if (start) begin
                logic [TBW-1:0] t, hi, lo;
                hi = (causal_r && (qblk_r + 1 < ntiles_r)) ? TBW'(qblk_r + 1) : ntiles_r;
                lo = (tlo_r < hi) ? tlo_r : TBW'(hi - 1);
                t  = TBW'(hi - lo);
                te <= t; tlo <= lo; busy <= 1'b1; s <= '0; rr <= '0; pdrained <= '0;
                ld_act <= 1'b1; ld_slot <= '0; ld_r <= '0; sh_full <= 1'b0;
                perf_cycles <= '0; perf_mac_qk <= '0; perf_mac_pv <= '0; perf_rows_issued <= '0;
                perf_tiles <= 32'(t);
                state <= ST_RUN;
            end

            ST_RUN: begin
                if (ld_act) begin
                    core_ld_valid <= 1'b1; core_ld_row <= ld_r[CW-1:0]; core_ld_data <= ld_row_data(ld_slot, ld_r);
                    ld_r <= ld_r + 1'b1;
                    if (ld_r == N-1) begin ld_act <= 1'b0; sh_full <= 1'b1; sh_slot <= ld_slot; end
                end
                if (can_issue) begin
                    iss_v <= 1'b1; iss_swap <= (rr == 0); iss_data <= cand;
                    iss_p <= cur_p; iss_t <= cur_t; iss_r <= rr[CW-1:0];
                    perf_rows_issued <= perf_rows_issued + 1;
                    if (cur_p) prdy[cur_t[0]][rr[CW-1:0]] <= 1'b0;
                    if (rr == 0 && s + 1 < nslots) begin
                        core_ld_valid <= 1'b1; core_ld_row <= '0; core_ld_data <= ld_row_data(s + 1'b1, '0);
                        ld_act <= 1'b1; ld_r <= 1; ld_slot <= s + 1'b1; sh_full <= 1'b0;
                    end
                    if (rr == N-1) begin rr <= '0; s <= s + 1'b1; end
                    else rr <= rr + 1'b1;
                end
                if (s == nslots) state <= ST_DRAIN;
            end

            ST_DRAIN: if (pdrained == 16'(te) * N) begin nctr <= '0; ncnt <= '0; state <= ST_NORM; end

            ST_NORM: begin
                nm_vi <= 1'b1; nm_ri <= nctr[CW-1:0];
                for (int c = 0; c < N; c++) nm_oi[c*NAW +: NAW] <= oacc[nctr[CW-1:0]][c];
                nctr <= nctr + 1'b1;
                if (nctr == N-1) state <= ST_NWAIT;
            end
            ST_NWAIT: if (ncnt == N) state <= ST_DONE;
            ST_DONE:  begin busy <= 1'b0; done <= 1'b1; state <= ST_IDLE; end
            default:  state <= ST_IDLE;
            endcase

            oa_v <= rv && rp;
            if (rv && rp) begin
                oa_r <= rr_o;
                for (int c = 0; c < N; c++) begin
                    oa_prod[c] <= $signed(oacc[rr_o][c]) * $signed({1'b0, alpha[rt[0]][rr_o]});
                    oa_res[c]  <= NAW'($signed(res_row[c*ACC_W +: ACC_W]));
                end
            end
            if (oa_v) begin
                for (int c = 0; c < N; c++) oacc[oa_r][c] <= NAW'(oa_prod[c] >>> EF) + oa_res[c];
                pdrained <= pdrained + 1'b1;
            end
            for (int b = 0; b < 2; b++)
                for (int r = 0; r < N; r++)
                    if (osm_we[b][r]) begin
                        for (int c = 0; c < N; c++) pbuf[b][r][c] <= osm_p[c*8 +: 8];
                        alpha[b][r] <= osm_a;
                        prdy[b][r]  <= 1'b1;
                    end
            if (nm_vo) begin
                for (int c = 0; c < N; c++) ofin[nm_ro][c] <= ACC_W'($signed(nm_oo[c*NAW +: NAW]));
                ncnt <= ncnt + 1'b1;
            end
        end
    end

    assign l_sel = (state == ST_NORM || state == ST_NWAIT) ? nm_ri : out_row_sel;
    assign dbg_l = 32'(l_val);

    genvar g;
    generate for (g = 0; g < N; g++) begin : OP
        assign out_row_flat[g*ACC_W +: ACC_W] = ofin[out_row_sel][g];
    end endgenerate

`ifdef FORMAL
    reg f_past = 1'b0;
    always @(posedge clk) f_past <= 1'b1;
    always @(*) if (!f_past) assume (!rst_n);
    always @(*) assume (cfg_ntiles >= 1 && cfg_ntiles <= MAXT && cfg_qblk < MAXT && cfg_tile_lo < MAXT);

    reg [N-1:0] f_pend;
    always @(posedge clk)
        if (!rst_n) f_pend <= '0;
        else for (int r = 0; r < N; r++) begin
            if (core_ld_valid && core_ld_row == r) f_pend[r] <= 1'b1;
            else if (swap_rows[r])                 f_pend[r] <= 1'b0;
        end
    reg [11:0] f_busy;
    always @(posedge clk) f_busy <= (!rst_n || !busy) ? 12'd0 : f_busy + 12'd1;

    always @(posedge clk) if (f_past && rst_n) begin
        for (int r = 0; r < N; r++) begin
            assert (!(core_ld_valid && core_ld_row == r && f_pend[r] && !swap_rows[r]));
            assert (!(swap_rows[r] && !f_pend[r]));
        end
        if (can_issue && cur_p) assert (prdy[cur_t[0]][rr[CW-1:0]]);
        if (osm_v && prdy[osm_t[0]][osm_r])
            assert (can_issue && cur_p && cur_t[0] == osm_t[0] && rr[CW-1:0] == osm_r);
        if (done) begin
            for (int b = 0; b < 2; b++) for (int r = 0; r < N; r++) assert (!prdy[b][r]);
        end
        if (state == ST_NORM) assert (pdrained == 16'(te) * N);
`ifndef FORMAL_NO_LIVENESS
        assert (f_busy < 12'((2 * MAXT + 2) * N + 48));
`endif
        assert (s <= nslots);
    end
    always @(posedge clk) if (f_past && rst_n) cover (done && te == MAXT && causal_r == 1'b0);
`endif

`ifndef SYNTHESIS
`ifndef FORMAL
    logic [N-1:0] sh_pending;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) sh_pending <= '0;
        else for (int r = 0; r < N; r++) begin
            logic sw; sw = swap_rows[r];
            if (core_ld_valid && core_ld_row == r && sh_pending[r] && !sw)
                $error("ASSERT: shadow row %0d overwritten before swap", r);
            if (sw && !sh_pending[r]) $error("ASSERT: row %0d swapped with no fresh weights", r);
            if (core_ld_valid && core_ld_row == r) sh_pending[r] <= 1'b1;
            else if (sw) sh_pending[r] <= 1'b0;
        end
    always_ff @(posedge clk) if (rst_n) begin
        if (osm_v && prdy[osm_t[0]][osm_r] && !(can_issue && cur_p && cur_t[0] == osm_t[0] && rr[CW-1:0] == osm_r))
            $error("ASSERT: probability row overwritten before it was consumed");
        if (rv && rp && state == ST_IDLE) $error("ASSERT: P*V result while idle");
        if (done) for (int b = 0; b < 2; b++) for (int r = 0; r < N; r++)
            if (prdy[b][r]) $error("ASSERT: unconsumed probability row at done");
    end
`endif
`endif
endmodule
