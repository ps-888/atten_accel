module softmax #(
    parameter N          = 8,
    parameter IN_W        = 32,
    parameter EXP_FRAC     = 8,
    parameter MAX_SHIFT    = 16,
    parameter RECIP_FRAC   = 16,
    parameter OUT_W        = 8
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                       valid_in,
    input  logic signed [N*IN_W-1:0]   row_in_flat,
    input  logic [4:0]                 scale_shift,

    output logic                       valid_out,
    output logic [N*OUT_W-1:0]         row_out_flat
);

    localparam int SUM_W = EXP_FRAC + 1 + $clog2(N) + 1;
    localparam logic [EXP_FRAC:0] EXP_ONE = (1 << EXP_FRAC);

    logic signed [IN_W-1:0] score_in [N];
    genvar gu;
    generate
        for (gu = 0; gu < N; gu++) begin : UNPACK
            assign score_in[gu] = row_in_flat[gu*IN_W +: IN_W];
        end
    endgenerate

    logic                    s1_valid;
    logic signed [IN_W-1:0]  s1_x   [N];
    logic signed [IN_W-1:0]  s1_max;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_max   <= '0;
        end else begin
            logic signed [IN_W-1:0] xi, m;
            s1_valid <= valid_in;
            m = score_in[0] >>> scale_shift;
            for (int i = 0; i < N; i++) begin
                xi = score_in[i] >>> scale_shift;
                s1_x[i] <= xi;
                if (xi > m) m = xi;
            end
            s1_max <= m;
        end
    end

    logic               s2_valid;
    logic [EXP_FRAC:0]  s2_exp [N];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
        end else begin
            logic signed [IN_W-1:0] d;
            logic [IN_W-1:0]        neg_d;
            s2_valid <= s1_valid;
            for (int i = 0; i < N; i++) begin
                d         = s1_x[i] - s1_max;
                neg_d     = -d;
                s2_exp[i] <= (neg_d > MAX_SHIFT) ? '0 : (EXP_ONE >> neg_d);
            end
        end
    end

    logic                      s3_valid;
    logic [SUM_W-1:0]          s3_sum;
    logic [EXP_FRAC:0]         s3_exp [N];
    logic [RECIP_FRAC:0]       s3_r0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid <= 1'b0;
        end else begin
            logic [SUM_W-1:0] acc;
            int                pos;
            s3_valid <= s2_valid;

            acc = '0;
            for (int i = 0; i < N; i++) begin
                acc        = acc + s2_exp[i];
                s3_exp[i] <= s2_exp[i];
            end
            s3_sum <= acc;

            pos = 0;
            for (int b = 0; b < SUM_W; b++)
                if (acc[b]) pos = b;

            if (RECIP_FRAC >= pos + 2)
                s3_r0 <= (RECIP_FRAC+1)'(3) << (RECIP_FRAC - pos - 2);
            else
                s3_r0 <= (RECIP_FRAC+1)'(3) >> (pos + 2 - RECIP_FRAC);
        end
    end

    logic                 s4a_valid;
    logic [SUM_W-1:0]     s4a_sum;
    logic [EXP_FRAC:0]    s4a_exp [N];
    logic [63:0]          s4a_r1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4a_valid <= 1'b0;
        end else begin
            logic [63:0] sr, two_minus_sr;
            s4a_valid <= s3_valid;
            s4a_sum   <= s3_sum;
            for (int i = 0; i < N; i++) s4a_exp[i] <= s3_exp[i];

            sr           = 64'(s3_sum) * 64'(s3_r0);
            two_minus_sr = (64'd2 << RECIP_FRAC) - sr;
            s4a_r1       <= (64'(s3_r0) * two_minus_sr) >> RECIP_FRAC;
        end
    end

    logic                 s4b_valid;
    logic [EXP_FRAC:0]    s4b_exp [N];
    logic [63:0]          s4b_r2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4b_valid <= 1'b0;
        end else begin
            logic [63:0] sr, two_minus_sr;
            s4b_valid <= s4a_valid;
            for (int i = 0; i < N; i++) s4b_exp[i] <= s4a_exp[i];

            sr           = 64'(s4a_sum) * s4a_r1;
            two_minus_sr = (64'd2 << RECIP_FRAC) - sr;
            s4b_r2       <= (s4a_r1 * two_minus_sr) >> RECIP_FRAC;
        end
    end

    logic [OUT_W-1:0] s4c_out [N];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
        end else begin
            logic [63:0] p;
            valid_out <= s4b_valid;
            for (int i = 0; i < N; i++) begin
                p = ((64'(s4b_exp[i]) * s4b_r2) + (64'd1 << (RECIP_FRAC-OUT_W-1)))
                    >> (RECIP_FRAC - OUT_W);
                s4c_out[i] <= (p > ((1 << OUT_W) - 1)) ? ((1 << OUT_W) - 1) : p[OUT_W-1:0];
            end
        end
    end

    genvar gp;
    generate
        for (gp = 0; gp < N; gp++) begin : PACK
            assign row_out_flat[gp*OUT_W +: OUT_W] = s4c_out[gp];
        end
    endgenerate

endmodule
