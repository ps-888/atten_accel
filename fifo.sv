module row_fifo #(
    parameter N      = 8,
    parameter DATA_W = 8,
    parameter DEPTH  = 16
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                  wr_en,
    input  logic [N*DATA_W-1:0]   wr_row_flat,
    output logic                  full,

    input  logic                  rd_en,
    output logic [N*DATA_W-1:0]   rd_row_flat,
    output logic                  empty
);

    localparam AW = $clog2(DEPTH);

    logic [N*DATA_W-1:0] mem [DEPTH];
    logic [AW-1:0] wr_ptr, rd_ptr;
    logic [AW:0]   count;

    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            if (wr_en && !full) begin
                mem[wr_ptr] <= wr_row_flat;
                wr_ptr      <= wr_ptr + 1'b1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1'b1;
            end

            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

    assign rd_row_flat = mem[rd_ptr];

endmodule
