// Caracterización (D2.3), NO es el diseño: rotador por constante (a + jb)(C + jS) en DSP48E2,
// 3 productos con el coeficiente fijo en el puerto B:
//   p1 = C·(a + b), p2 = (S − C)·a, p3 = (C + S)·b;  re = p1 − p3, im = p1 + p2.
// (S − C) y (C + S) se calculan al elaborar (WC + 1 bits <= 18, puerto B); a + b va al pre-sumador.
module rot_const_dsp #(parameter WD = 18, parameter WC = 16,
                       parameter signed [WC-1:0] C = 0, parameter signed [WC-1:0] S = 0) (
    input  wire                    clk,
    input  wire signed [WD-1:0]    a_i, b_i,
    output reg  signed [WD+WC+1:0] re_o, im_o
);
    localparam signed [WC:0] K1 = C;
    localparam signed [WC:0] K2 = S - C;
    localparam signed [WC:0] K3 = C + S;
    reg signed [WD-1:0] a, b, a2, b2;
    reg signed [WD:0]   s_ab;
    (* use_dsp = "yes" *) reg signed [WD+WC+1:0] p1, p2, p3;
    always @(posedge clk) begin
        a <= a_i; b <= b_i;
        s_ab <= a + b; a2 <= a; b2 <= b;
        p1 <= K1 * s_ab; p2 <= K2 * a2; p3 <= K3 * b2;
        re_o <= p1 - p3; im_o <= p1 + p2;
    end
endmodule
