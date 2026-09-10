module sparsity_detect #(
    parameter N      = 8,
    parameter DATA_W = 8
) (
    input  logic [N*DATA_W-1:0]      row_in_flat,
    input  logic signed [DATA_W-1:0] thresh,
    output logic [N-1:0]             row_active
);

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : DET
            wire signed [DATA_W-1:0] v = row_in_flat[i*DATA_W +: DATA_W];

            assign row_active[i] = (v > thresh) || (v < -thresh);
        end
    endgenerate

endmodule
