// Caracterización de costos (D2.2), NO es el diseño: sumador registrado de W bits.
module add_reg #(parameter W = 18) (
    input  wire                clk,
    input  wire signed [W-1:0] a_i, b_i,
    output reg  signed [W:0]   s_o
);
    reg signed [W-1:0] a, b;
    always @(posedge clk) begin
        a <= a_i; b <= b_i;
        s_o <= a + b;
    end
endmodule
