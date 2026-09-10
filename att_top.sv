module attention_top #(
    parameter N        = 8,
    parameter DATA_W   = 8,
    parameter ACC_W    = 32,
    parameter SM_OUT_W = 8
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                       ld_k_valid,
    input  logic [$clog2(N)-1:0]       ld_k_row,
    input  logic [N*DATA_W-1:0]        ld_k_data_flat,
    input  logic                       ld_v_valid,
    input  logic [$clog2(N)-1:0]       ld_v_row,
    input  logic [N*DATA_W-1:0]        ld_v_data_flat,

    input  logic                       ld_q_valid,
    input  logic [$clog2(N)-1:0]       ld_q_row,
    input  logic [N*DATA_W-1:0]        ld_q_data_flat,

    input  logic                       scale_shift_valid,
    input  logic [4:0]                 scale_shift_in,

    input  logic                       start,
    output logic                       busy,
    output logic                       done,

    input  logic [$clog2(N)-1:0]       out_row_sel,
    output logic [N*ACC_W-1:0]         out_row_flat
);

    logic signed [DATA_W-1:0] k_mem   [N][N];
    logic signed [DATA_W-1:0] v_mem   [N][N];
    logic signed [DATA_W-1:0] q_mem   [N][N];
    logic signed [ACC_W-1:0]  score_mem [N][N];
    logic        [SM_OUT_W-1:0] norm_mem  [N][N];
    logic signed [DATA_W-1:0] norm_act_mem [N][N];
    logic signed [ACC_W-1:0]  out_mem  [N][N];
    logic [4:0]                scale_shift_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scale_shift_r <= 5'd3;
        end else begin
            if (ld_k_valid) for (int c = 0; c < N; c++) k_mem[ld_k_row][c] <= $signed(ld_k_data_flat[c*DATA_W +: DATA_W]);
            if (ld_v_valid) for (int c = 0; c < N; c++) v_mem[ld_v_row][c] <= $signed(ld_v_data_flat[c*DATA_W +: DATA_W]);
            if (ld_q_valid) for (int c = 0; c < N; c++) q_mem[ld_q_row][c] <= $signed(ld_q_data_flat[c*DATA_W +: DATA_W]);
            if (scale_shift_valid) scale_shift_r <= scale_shift_in;
        end
    end

    logic core_rst_n;
    logic core_ld_weight_valid;
    logic [$clog2(N)-1:0] core_ld_weight_row;
    logic [N*DATA_W-1:0]  core_ld_weight_data_flat;
    logic core_act_valid;
    logic [N*DATA_W-1:0]  core_act_row_flat;
    logic core_act_ready;
    logic [N-1:0] core_row_en_override;
    logic core_res_valid;
    logic [N*ACC_W-1:0] core_res_row_flat;

    assign core_row_en_override = '1;

    attn_core_top_exact_only_v2 #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_core (
        .clk(clk), .rst_n(core_rst_n),
        .ld_weight_valid(core_ld_weight_valid),
        .ld_weight_row(core_ld_weight_row),
        .ld_weight_data_flat(core_ld_weight_data_flat),
        .act_valid(core_act_valid),
        .act_row_flat(core_act_row_flat),
        .act_ready(core_act_ready),
        .row_en_override(core_row_en_override),
        .sparsity_thresh(8'sd0),
        .gate_sel(1'b0),
        .lif_thresh(16'd0),
        .res_valid(core_res_valid),
        .res_row_flat(core_res_row_flat)
    );

    logic signed [ACC_W-1:0] core_res_row_arr [N];
    genvar gru;
    generate
        for (gru = 0; gru < N; gru++) begin : RES_UNPACK
            assign core_res_row_arr[gru] = $signed(core_res_row_flat[gru*ACC_W +: ACC_W]);
        end
    endgenerate

    logic sm_valid_in, sm_valid_out;
    logic signed [N*ACC_W-1:0] sm_row_in_flat;
    logic [N*SM_OUT_W-1:0]     sm_row_out_flat;

    softmax_approx_pipelined #(.N(N), .IN_W(ACC_W), .OUT_W(SM_OUT_W)) u_softmax (
        .clk(clk), .rst_n(rst_n),
        .valid_in(sm_valid_in),
        .row_in_flat(sm_row_in_flat),
        .scale_shift(scale_shift_r),
        .valid_out(sm_valid_out),
        .row_out_flat(sm_row_out_flat)
    );

    logic [SM_OUT_W-1:0] sm_row_out_arr [N];
    genvar gsu;
    generate
        for (gsu = 0; gsu < N; gsu++) begin : SM_UNPACK
            assign sm_row_out_arr[gsu] = sm_row_out_flat[gsu*SM_OUT_W +: SM_OUT_W];
        end
    endgenerate

    typedef enum logic [3:0] {
        ST_IDLE, ST_RST1, ST_LOAD_K, ST_STREAM_Q, ST_DRAIN_Q,
        ST_SOFTMAX, ST_DRAIN_SM, ST_RST2, ST_LOAD_V, ST_STREAM_NORM,
        ST_DRAIN_OUT, ST_DONE
    } state_t;
    state_t state;

    logic [$clog2(N)+1:0] ctr;
    logic [$clog2(N)+1:0] drain_ctr;

    logic [N*DATA_W-1:0] k_row_flat, v_row_flat, q_row_flat, norm_row_flat;
    logic [N*ACC_W-1:0]  score_row_flat;
    genvar gp;
    generate
        for (gp = 0; gp < N; gp++) begin : PACKROWS
            assign k_row_flat[gp*DATA_W +: DATA_W]     = k_mem[ctr[$clog2(N)-1:0]][gp];
            assign v_row_flat[gp*DATA_W +: DATA_W]     = v_mem[ctr[$clog2(N)-1:0]][gp];
            assign q_row_flat[gp*DATA_W +: DATA_W]     = q_mem[ctr[$clog2(N)-1:0]][gp];
            assign norm_row_flat[gp*DATA_W +: DATA_W]  = norm_act_mem[ctr[$clog2(N)-1:0]][gp];
            assign score_row_flat[gp*ACC_W +: ACC_W]   = score_mem[ctr[$clog2(N)-1:0]][gp];
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                    <= ST_IDLE;
            ctr                      <= '0;
            drain_ctr                <= '0;
            core_rst_n               <= 1'b0;
            core_ld_weight_valid     <= 1'b0;
            core_act_valid           <= 1'b0;
            sm_valid_in              <= 1'b0;
            busy                     <= 1'b0;
            done                     <= 1'b0;
        end else begin
            core_ld_weight_valid <= 1'b0;
            core_act_valid       <= 1'b0;
            sm_valid_in          <= 1'b0;
            done                 <= 1'b0;

            case (state)
                ST_IDLE: begin
                    core_rst_n <= 1'b1;
                    if (start) begin
                        busy  <= 1'b1;
                        ctr   <= '0;
                        state <= ST_RST1;
                    end
                end

                ST_RST1: begin
                    core_rst_n <= 1'b0;
                    ctr <= ctr + 1'b1;
                    if (ctr == 3) begin
                        core_rst_n <= 1'b1;
                        ctr <= '0;
                        state <= ST_LOAD_K;
                    end
                end

                ST_LOAD_K: begin
                    core_ld_weight_valid <= 1'b1;
                    core_ld_weight_row   <= ctr[$clog2(N)-1:0];
                    core_ld_weight_data_flat <= k_row_flat;
                    ctr <= ctr + 1'b1;
                    if (ctr == N-1) begin
                        ctr   <= '0;
                        state <= ST_STREAM_Q;
                    end
                end

                ST_STREAM_Q: begin
                    core_act_valid    <= 1'b1;
                    core_act_row_flat <= q_row_flat;
                    ctr <= ctr + 1'b1;
                    if (ctr == N-1) begin
                        ctr        <= '0;
                        drain_ctr  <= '0;
                        state      <= ST_DRAIN_Q;
                    end
                end

                ST_DRAIN_Q: begin
                    if (core_res_valid) begin
                        for (int c = 0; c < N; c++)
                            score_mem[drain_ctr[$clog2(N)-1:0]][c] <= core_res_row_arr[c];
                        drain_ctr <= drain_ctr + 1'b1;
                        if (drain_ctr == N-1) begin
                            ctr        <= '0;
                            drain_ctr  <= '0;
                            state      <= ST_SOFTMAX;
                        end
                    end
                end

                ST_SOFTMAX: begin
                    if (ctr < N) begin
                        sm_valid_in    <= 1'b1;
                        sm_row_in_flat <= score_row_flat;
                        ctr <= ctr + 1'b1;
                    end
                    if (sm_valid_out) begin
                        for (int c = 0; c < N; c++) begin
                            norm_mem[drain_ctr[$clog2(N)-1:0]][c] <= sm_row_out_arr[c];
                            norm_act_mem[drain_ctr[$clog2(N)-1:0]][c] <=
                                $signed({1'b0, sm_row_out_arr[c][SM_OUT_W-1:1]});
                        end
                        drain_ctr <= drain_ctr + 1'b1;
                        if (drain_ctr == N-1) begin
                            ctr   <= '0;
                            state <= ST_RST2;
                        end
                    end
                end

                ST_RST2: begin
                    core_rst_n <= 1'b0;
                    ctr <= ctr + 1'b1;
                    if (ctr == 3) begin
                        core_rst_n <= 1'b1;
                        ctr <= '0;
                        state <= ST_LOAD_V;
                    end
                end

                ST_LOAD_V: begin
                    core_ld_weight_valid <= 1'b1;
                    core_ld_weight_row   <= ctr[$clog2(N)-1:0];
                    core_ld_weight_data_flat <= v_row_flat;
                    ctr <= ctr + 1'b1;
                    if (ctr == N-1) begin
                        ctr   <= '0;
                        state <= ST_STREAM_NORM;
                    end
                end

                ST_STREAM_NORM: begin
                    core_act_valid    <= 1'b1;
                    core_act_row_flat <= norm_row_flat;
                    ctr <= ctr + 1'b1;
                    if (ctr == N-1) begin
                        ctr        <= '0;
                        drain_ctr  <= '0;
                        state      <= ST_DRAIN_OUT;
                    end
                end

                ST_DRAIN_OUT: begin
                    if (core_res_valid) begin
                        for (int c = 0; c < N; c++)
                            out_mem[drain_ctr[$clog2(N)-1:0]][c] <= core_res_row_arr[c];
                        drain_ctr <= drain_ctr + 1'b1;
                        if (drain_ctr == N-1)
                            state <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    genvar go;
    generate
        for (go = 0; go < N; go++) begin : OUTPACK
            assign out_row_flat[go*ACC_W +: ACC_W] = out_mem[out_row_sel][go];
        end
    endgenerate

endmodule
