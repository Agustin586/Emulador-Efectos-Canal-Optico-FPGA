// Caracterización (D2.2): rotador por constante (a + jb)(C + jS) en LUT (sin DSP),
// producto exacto de WD + WC bits, registrado.
module rot_const #(parameter WD = 18, parameter WC = 16,
                   parameter signed [WC-1:0] C = 0, parameter signed [WC-1:0] S = 0) (
    input  wire                    clk,
    input  wire signed [WD-1:0]    a_i, b_i,
    output reg  signed [WD+WC:0]   re_o, im_o
);
    reg signed [WD-1:0] a, b;
    (* use_dsp = "no" *) wire signed [WD+WC-1:0] ac = a * C;
    (* use_dsp = "no" *) wire signed [WD+WC-1:0] bs = b * S;
    (* use_dsp = "no" *) wire signed [WD+WC-1:0] as_ = a * S;
    (* use_dsp = "no" *) wire signed [WD+WC-1:0] bc = b * C;
    always @(posedge clk) begin
        a <= a_i; b <= b_i;
        re_o <= ac - bs;
        im_o <= as_ + bc;
    end
endmodule
