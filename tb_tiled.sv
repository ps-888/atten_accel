`timescale 1ns/1ps
module tb_tiled;
    parameter int N = 8, MAXT = 8;
    localparam int DW = 8, AW = 32, CW = $clog2(N), LA = $clog2(MAXT*N), TB = $clog2(MAXT)+1;
    logic clk = 0, rst_n = 0; always #5 clk = ~clk;
    logic lq=0, lk=0, lv=0; logic [CW-1:0] qr; logic [LA-1:0] kr, vr;
    logic [N*DW-1:0] qd, kd, vd;
    logic cfg_valid=0, causal; logic [TB-1:0] ntiles, qblk, tlo; logic [15:0] mult; logic [5:0] shift;
    logic [7:0] thqk, thpv; logic start=0, busy, done; logic [CW-1:0] sel=0;
    logic [N*AW-1:0] orow; logic [31:0] dbg_l, pc, pq, pp, pr, pt;
    attn_tiled #(.N(N), .MAXT(MAXT)) dut (.clk, .rst_n,
        .ld_q_valid(lq), .ld_q_row(qr), .ld_q_data(qd), .ld_k_valid(lk), .ld_k_row(kr), .ld_k_data(kd),
        .ld_v_valid(lv), .ld_v_row(vr), .ld_v_data(vd),
        .cfg_valid, .cfg_ntiles(ntiles), .cfg_causal(causal), .cfg_qblk(qblk), .cfg_tile_lo(tlo), .cfg_sm_mult(mult),
        .cfg_sm_shift(shift), .cfg_th_qk(thqk), .cfg_th_pv(thpv), .start, .busy, .done,
        .out_row_sel(sel), .out_row_flat(orow), .dbg_l,
        .perf_cycles(pc), .perf_mac_qk(pq), .perf_mac_pv(pp), .perf_rows_issued(pr), .perf_tiles(pt));
    int fi, fo, T, rc, lat, nt, cz, qb, tl, mu, sh, tq, tp, do_rst, poke;
    int qm[N][N], km[MAXT*N][N], vm[MAXT*N][N];
    initial begin
        string vf, rf;
        if (!$value$plusargs("VEC=%s", vf)) vf = "vectors.txt";
        if (!$value$plusargs("RES=%s", rf)) rf = "results.txt";
        fi = $fopen(vf, "r"); fo = $fopen(rf, "w"); if (fi == 0) $fatal(1, "no %s", vf);
        rc = $fscanf(fi, "%d", T);
        repeat (3) @(posedge clk); rst_n = 1;
        for (int t = 0; t < T; t++) begin
            rc = $fscanf(fi, "%d %d %d %d %d %d %d %d %d %d", nt, cz, qb, tl, mu, sh, tq, tp, do_rst, poke);
            for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) rc = $fscanf(fi, "%d", qm[i][j]);
            for (int i = 0; i < nt*N; i++) for (int j = 0; j < N; j++) rc = $fscanf(fi, "%d", km[i][j]);
            for (int i = 0; i < nt*N; i++) for (int j = 0; j < N; j++) rc = $fscanf(fi, "%d", vm[i][j]);
            if (do_rst) begin @(negedge clk) rst_n = 0; @(negedge clk) rst_n = 1; end
            for (int ii = 0; ii < nt*N; ii++) begin
                int i; i = (t % 2) ? nt*N-1-ii : ii;
                @(negedge clk); lk = 1; lv = 1; kr = LA'(i); vr = LA'(i);
                lq = (i < N); qr = CW'(i);
                for (int j = 0; j < N; j++) begin
                    kd[j*DW +: DW] = DW'(km[i][j]); vd[j*DW +: DW] = DW'(vm[i][j]);
                    if (i < N) qd[j*DW +: DW] = DW'(qm[i][j]);
                end
            end
            @(negedge clk); lk = 0; lv = 0; lq = 0;
            cfg_valid = 1; ntiles = TB'(nt); causal = cz[0]; qblk = TB'(qb); tlo = TB'(tl); mult = 16'(mu); shift = 6'(sh);
            thqk = 8'(tq); thpv = 8'(tp);
            @(negedge clk); cfg_valid = 0; start = 1;
            @(negedge clk); start = 0; lat = 1;
            while (!done) begin
                @(negedge clk); lat++;
                if (poke && lat == 7) begin start = 1; cfg_valid = 1; ntiles = 1; end
                else begin start = 0; cfg_valid = 0; end
                if (lat > 200000) $fatal(1, "timeout test %0d", t);
            end
            $fwrite(fo, "T %0d %0d %0d %0d %0d %0d %0d\n", t, lat, pc, pq, pp, pr, pt);
            for (int r = 0; r < N; r++) begin
                sel = CW'(r); #1;
                $fwrite(fo, "%0d", dbg_l);
                for (int c = 0; c < N; c++) $fwrite(fo, " %0d", $signed(orow[c*AW +: AW]));
                $fwrite(fo, "\n");
            end
        end
        $fclose(fo); $display("TB: %0d tests", T); $finish;
    end
endmodule
