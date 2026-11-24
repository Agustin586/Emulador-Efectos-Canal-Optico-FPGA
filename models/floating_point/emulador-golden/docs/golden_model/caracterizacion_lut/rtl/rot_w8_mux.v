// Caracterización (D2.10): rotador W8 con multiplexor contra 1 (carril que alterna entre 1 y W8).
// sel = 1: ((a + b) + j(b − a))·K, K = cos(π/4) en WC bits; sel = 0: (a + jb)·1, alineado a la misma escala (·2^(WC−1)).
module rot_w8_mux #(parameter WD = 17, parameter WC = 12, parameter signed [WC-1:0] K = 0) (
    input  wire                    clk,
    input  wire                    sel_i,
    input  wire signed [WD-1:0]    a_i, b_i,
    output reg  signed [WD+WC:0]   re_o, im_o
);
    reg signed [WD-1:0] a, b;
    reg sel;
    wire signed [WD:0] s = a + b;
    wire signed [WD:0] d = b - a;
    (* use_dsp = "no" *) wire signed [WD+WC:0] ps = s * K;
    (* use_dsp = "no" *) wire signed [WD+WC:0] pd = d * K;
    wire signed [WD+WC:0] a1 = a <<< (WC - 1);
    wire signed [WD+WC:0] b1 = b <<< (WC - 1);
    always @(posedge clk) begin
        a <= a_i; b <= b_i; sel <= sel_i;
        re_o <= sel ? ps : a1;
        im_o <= sel ? pd : b1;
    end
endmodule
