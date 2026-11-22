// Caracterización (D2.2): multiplicador complejo variable con 3 productos
// (re = c(a+b) − b(c+d), im = c(a+b) + a(d−c)), pre-sumador y registros para el DSP48E2.
module cmul3 #(parameter WD = 18, parameter WC = 18) (
    input  wire                    clk,
    input  wire signed [WD-1:0]    a_i, b_i,
    input  wire signed [WC-1:0]    c_i, d_i,
    output reg  signed [WD+WC+1:0] re_o, im_o
);
    reg signed [WD-1:0] a, b, a2, b2;
    reg signed [WC-1:0] c, d, c2;
    reg signed [WD:0]   s_ab;
    reg signed [WC:0]   s_cd, d_dc;
    reg signed [WD+WC+1:0] p1, p2, p3;
    always @(posedge clk) begin
        a <= a_i; b <= b_i; c <= c_i; d <= d_i;
        s_ab <= a + b; s_cd <= c + d; d_dc <= d - c; a2 <= a; b2 <= b; c2 <= c;
        p1 <= c2 * s_ab; p2 <= a2 * d_dc; p3 <= b2 * s_cd;
        re_o <= p1 - p3; im_o <= p1 + p2;
    end
endmodule
