module lif_gate #(
    parameter N          = 8,
    parameter DATA_W     = 8,
    parameter POT_W       = 16,
    parameter LEAK_SHIFT  = 3
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  update,
    input  logic [N*DATA_W-1:0]   row_in_flat,
    input  logic [POT_W-1:0]      thresh,
    output logic [N-1:0]          fire
);

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : ROW
            logic [POT_W-1:0] mem;

            wire signed [DATA_W-1:0] v     = row_in_flat[i*DATA_W +: DATA_W];
            wire signed [POT_W-1:0]  v_ext = {{(POT_W-DATA_W){v[DATA_W-1]}}, v};
            wire        [POT_W-1:0]  mag   = v_ext[POT_W-1] ? (~v_ext + 1'b1) : v_ext;

            wire [POT_W-1:0]   leaked        = mem - (mem >> LEAK_SHIFT);
            wire [POT_W:0]     candidate_ext = {1'b0, leaked} + {1'b0, mag};
            wire [POT_W-1:0]   candidate     = candidate_ext[POT_W] ? '1 : candidate_ext[POT_W-1:0];
            wire               will_fire     = update && (candidate >= thresh);

            assign fire[i] = will_fire;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    mem <= '0;
                else if (update)

                    mem <= will_fire ? (candidate - thresh) : candidate;
            end
        end
    endgenerate

endmodule
