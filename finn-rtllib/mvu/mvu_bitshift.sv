/******************************************************************************
 * @brief   MVU compute kernel for power-of-two (right-shift) weights.
 *
 * @details
 *  Weight encoding (WEIGHT_WIDTH bits, sign-magnitude — NOT two's complement):
 *    w[WEIGHT_WIDTH-1]       : sign  (0 = add partial product, 1 = subtract)
 *    w[WEIGHT_WIDTH-2 : 0]  : exponent for right-shift (EXP_BITS bits)
 *
 *  Zero-weight encoding:
 *    exp == '1  (all exponent bits set) is reserved as the zero-weight code.
 *    Valid shift amounts are therefore  0 .. 2^EXP_BITS - 2.
 *    The module-level `zero` input also forces all partial products to zero.
 *
 *  Operation per (PE, SIMD) pair:
 *    1. Decode weight into (sign, exp).
 *    2. Right-shift activation by exp  →  |partial_product| <= activation.
 *    3. Negate if sign == 1.
 *    4. Accumulate across folds, reset at each output boundary (L[3] flag).
 *    5. Sum across SIMD dimension via pipelined add_multi.
 *
 *  Pipeline depth  =  3  +  $clog2(SIMD)  +  (SIMD == 1)  +  1
 *  (matches original mvu.sv so the AXI wrapper's output buffer is unchanged)
 *****************************************************************************/
 
module mvu_bitshift #(
    int unsigned  PE,
    int unsigned  SIMD,
    int unsigned  WEIGHT_WIDTH,       // must be >= 2:  sign[1] + exp[WEIGHT_WIDTH-1]
    int unsigned  ACTIVATION_WIDTH,   // must be >= 2
    int unsigned  ACCU_WIDTH,
 
    bit  SIGNED_ACTIVATIONS = 0
)(
    input  logic  clk,
    input  logic  rst,
    input  logic  en,
 
    input  logic  last,
    input  logic  zero,   // force all partial products to zero this cycle
 
    // Sign-magnitude weights (NOT two's complement)
    input  logic [PE  -1:0][SIMD-1:0][WEIGHT_WIDTH    -1:0]  w,
    // Activations (unsigned by default, signed when SIGNED_ACTIVATIONS=1)
    input  logic            [SIMD-1:0][ACTIVATION_WIDTH-1:0]  a,
 
    output logic  vld,
    output logic signed [PE-1:0][ACCU_WIDTH-1:0]  p
);
 
    import mvu_pkg::*;
 
    // -----------------------------------------------------------------------
    // Parameter validation
    // -----------------------------------------------------------------------
    initial begin
        if (WEIGHT_WIDTH < 2) begin
            $error("%m: WEIGHT_WIDTH=%0d must be >= 2 (1 sign bit + 1 exponent bit).",
                   WEIGHT_WIDTH);
            $finish;
        end
        if (ACTIVATION_WIDTH < 2) begin
            $error("%m: ACTIVATION_WIDTH=%0d must be >= 2.", ACTIVATION_WIDTH);
            $finish;
        end
    end
 
    // -----------------------------------------------------------------------
    // Derived constants
    // -----------------------------------------------------------------------
    localparam int unsigned  EXP_BITS   = WEIGHT_WIDTH - 1;
 
    // Signed partial product per (PE,SIMD) pair after shift+sign.
    // Right-shift keeps magnitude <= activation, so ACTIVATION_WIDTH+1 bits
    // (one extra for the sign) is sufficient for a single fold contribution.
    localparam int unsigned  PP_WIDTH   = ACTIVATION_WIDTH + 1;
 
    // Total pipeline depth (identical formula to mvu.sv so the AXI wrapper
    // output queue sizing and vld alignment remain compatible).
    localparam int unsigned  PIPE_DEPTH = 3 + $clog2(SIMD) + (SIMD == 1) + 1;
 
    // -----------------------------------------------------------------------
    // Last-flag pipeline
    //   L[k] = `last` captured k edges ago.
    //   vld fires PIPE_DEPTH cycles after the corresponding `last`.
    // -----------------------------------------------------------------------
/* verilator lint_off LITENDIAN */
    logic [1:PIPE_DEPTH]  L = '0;
/* verilator lint_on LITENDIAN */
    always_ff @(posedge clk) begin
        if (rst)     L <= '0;
        else if (en) L <= {last, L[1:PIPE_DEPTH-1]};
    end
    assign vld = L[PIPE_DEPTH];
 
    // -----------------------------------------------------------------------
    // Per-PE compute pipe
    // -----------------------------------------------------------------------
    for (genvar c = 0; c < PE; c++) begin : genPipes
 
        // Per-SIMD accumulated partial sums fed into the SIMD reduction tree.
        uwire signed [ACCU_WIDTH-1:0]  p3[SIMD];
 
        // -------------------------------------------------------------------
        // Per-SIMD stages 1–3
        // -------------------------------------------------------------------
        for (genvar s = 0; s < SIMD; s++) begin : genSIMD
 
            // ---------------------------------------------------------------
            // Stage 1 — Register inputs, decode weight fields
            // ---------------------------------------------------------------
            logic [ACTIVATION_WIDTH-1:0]  A1 = '0;
            logic [EXP_BITS-1:0]          E1 = '0;   // right-shift exponent
            logic                         S1 = '0;   // sign:  0→add, 1→subtract
            logic                         Z1 = '0;   // zero flag (module or weight)
 
            always_ff @(posedge clk) begin
                if (rst) begin
                    A1 <= '0;  E1 <= '0;  S1 <= '0;  Z1 <= '0;
                end else if (en) begin
                    A1 <= a[s];
                    S1 <= w[c][s][WEIGHT_WIDTH-1];
                    E1 <= w[c][s][EXP_BITS-1:0];
                    // Zero: module-level flag  OR  reserved all-ones exponent code
                    Z1 <= zero || (&w[c][s][EXP_BITS-1:0]);
                end
            end
 
            // ---------------------------------------------------------------
            // Stage 2 — Barrel right-shift + sign-magnitude → signed value
            //
            //   shifted  =  A1 >> E1           (logical or arithmetic)
            //   M2       =  Z1 ? 0 : (S1 ? –shifted : +shifted)
            //
            // The right-shift ensures |shifted| <= |A1|, so PP_WIDTH bits
            // (ACTIVATION_WIDTH + 1 sign bit) always suffice.
            // ---------------------------------------------------------------
            logic signed [PP_WIDTH-1:0]  M2 = '0;
 
            always_ff @(posedge clk) begin
                if (rst) begin
                    M2 <= '0;
                end else if (en) begin
                    automatic logic [ACTIVATION_WIDTH-1:0]  shifted;
 
                    if (Z1) begin
                        shifted = '0;
                    end else if (SIGNED_ACTIVATIONS) begin
                        // Arithmetic right-shift preserves sign of activation.
                        shifted = $unsigned($signed(A1) >>> E1);
                    end else begin
                        // Logical right-shift for unsigned activations.
                        shifted = A1 >> E1;
                    end
 
                    // Apply sign:  sign-magnitude → two's complement for adder tree.
                    M2 <= S1 ? -$signed({1'b0, shifted})
                             :  $signed({1'b0, shifted});
                end
            end
 
            // ---------------------------------------------------------------
            // Stage 3 — Fold accumulation
            //
            //   P3 <= M2 + (L[3] ? 0 : P3)
            //
            //   L[3] is the `last` flag from 2 cycles ago, which arrives
            //   aligned with the first partial product of a NEW output window.
            //   This resets P3 to just M2, starting a fresh accumulation —
            //   exactly the same pattern as the original mvu.sv.
            //
            //   P3 is ACCU_WIDTH bits to accommodate accumulation across all
            //   SF = MW/SIMD folds without overflow.
            // ---------------------------------------------------------------
            logic signed [ACCU_WIDTH-1:0]  P3 = '0;
 
            always_ff @(posedge clk) begin
                if (rst)     P3 <= '0;
                else if (en) P3 <= $signed(M2) + (L[3] ? '0 : P3);
            end
 
            assign p3[s] = P3;
 
        end : genSIMD
 
        // -------------------------------------------------------------------
        // Stage 4 — SIMD reduction (pipelined adder tree across SIMD elements)
        //
        //   Each p3[s] is a signed ACCU_WIDTH-bit value.
        //   ARG_LO / ARG_HI enable signed reduction inside add_multi.
        //   DEPTH absorbs (PIPE_DEPTH - 4) pipeline registers into the tree.
        // -------------------------------------------------------------------
        localparam longint signed  ARG_LO = -(longint'(1) << (ACCU_WIDTH-1));
        localparam longint signed  ARG_HI =  (longint'(1) << (ACCU_WIDTH-1)) - 1;
        localparam int unsigned    SUM_W  =  sumwidth(SIMD, ACCU_WIDTH, ARG_LO, ARG_HI);
 
        uwire [ACCU_WIDTH-1:0]  arg[SIMD];
        for (genvar s = 0; s < SIMD; s++) assign arg[s] = p3[s];
 
        uwire [SUM_W-1:0]  simd_sum;
        add_multi #(
            .N         (SIMD),
            .DEPTH     (PIPE_DEPTH - 4),
            .ARG_WIDTH (ACCU_WIDTH),
            .ARG_LO    (ARG_LO),
            .ARG_HI    (ARG_HI)
        ) reduce (
            .clk, .rst, .en,
            .arg (arg),
            .sum (simd_sum)
        );
 
        // -------------------------------------------------------------------
        // Stage 5 — Final register
        //
        //   Truncates the SIMD sum to ACCU_WIDTH.  The caller is responsible
        //   for ensuring ACCU_WIDTH is wide enough to prevent overflow of the
        //   full SIMD × fold accumulation.
        // -------------------------------------------------------------------
        logic signed [ACCU_WIDTH-1:0]  Res5 = 'x;
 
        always_ff @(posedge clk) begin
            if (rst)     Res5 <= 'x;
            else if (en) Res5 <= simd_sum[ACCU_WIDTH-1:0];
        end
 
        assign p[c] = Res5;
 
    end : genPipes
 
endmodule : mvu_bitshift