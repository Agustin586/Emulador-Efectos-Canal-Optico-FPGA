// Caracterización (D2.2): mariposa radix-2 compleja registrada, a ± b exacto con W + 1 bits.
module bfly_r2 #(parameter W = 18) (
    input  wire                clk,
    input  wire signed [W-1:0] ar_i, ai_i, br_i, bi_i,
    output reg  signed [W:0]   sr_o, si_o, dr_o, di_o
);
    reg signed [W-1:0] ar, ai, br, bi;
    always @(posedge clk) begin
        ar <= ar_i; ai <= ai_i; br <= br_i; bi <= bi_i;
        sr_o <= ar + br; si_o <= ai + bi;
        dr_o <= ar - br; di_o <= ai - bi;
    end
endmodule
