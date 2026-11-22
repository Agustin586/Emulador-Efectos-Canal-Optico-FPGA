// Caracterización (D2.2): rotador W8 = (1 − j)/√2: ((a + b) + j(b − a))·K, K = cos(π/4) en WC bits.
module rot_w8 #(parameter WD = 18, parameter WC = 16, parameter signed [WC-1:0] K = 0) (
    input  wire                    clk,
    input  wire signed [WD-1:0]    a_i, b_i,
    output reg  signed [WD+WC:0]   re_o, im_o
);
    reg signed [WD-1:0] a, b;
    wire signed [WD:0] s = a + b;
    wire signed [WD:0] d = b - a;
    (* use_dsp = "no" *) wire signed [WD+WC:0] ps = s * K;
    (* use_dsp = "no" *) wire signed [WD+WC:0] pd = d * K;
    always @(posedge clk) begin
        a <= a_i; b <= b_i;
        re_o <= ps; im_o <= pd;
    end
endmodule
