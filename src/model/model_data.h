/* 由 scripts/export_model.py 自動產生，請勿手動修改 */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#include <stdint.h>

typedef enum {
    OP_QUANTIZE = 0,
    OP_CONV_2D = 1,
    OP_DEPTHWISE_CONV_2D = 2,
    OP_ADD = 3,
    OP_AVERAGE_POOL_2D = 4,
    OP_FULLY_CONNECTED = 5,
    OP_SOFTMAX = 6,
    OP_RESHAPE = 7,
} op_type_t;

/* activation tensor：資料放在 g_arena[off]，layout NHWC，int8 (is_uint8=1 時為 uint8) */
typedef struct {
    uint32_t off;
    uint16_t n, h, w, c;
    uint8_t  is_uint8;
} tensor_t;

/*
 * requant (TFLite double-rounding)：
 *   out = clamp( MBQM(acc, mult[c], shift[c]) + out_zp, act_min, act_max )
 *   MBQM(x,m,s) = RoundingDivideByPOT( SRDHM(x << max(s,0), m), max(-s,0) )
 * CONV_2D  : w = [O][KH][KW][I]，acc = bias[o] + sum w*(x - in0_zp)   (padding 視為 x = in0_zp)
 * DWCONV   : w = [KH][KW][C*M]，out ch = ic*M + m
 * FC       : w = [O][I]，acc = bias[o] + sum (w - w_zp)*(x - in0_zp)
 * bias_folded = bias - in0_zp*sum(w)，用了就可以直接算 sum w*x (padding 要填 in0_zp)
 * ADD      : a' = MBQM((a-in0_zp)<<left_shift, in0_mult, in0_shift)，b' 同理，
 *            out = MBQM(a'+b', mult[0], shift[0]) + out_zp
 * QUANTIZE : out = MBQM(x - in0_zp, mult[0], shift[0]) + out_zp
 * AVGPOOL  : 輸入輸出同 scale，sum 後 round-half-away 除以有效個數
 */
typedef struct {
    uint8_t  type;
    int16_t  in0, in1, out;          /* tensor id，-1 = 無 */
    uint8_t  kh, kw, sh, sw, dh, dw, pad_t, pad_l;
    uint16_t depth_mult;
    int32_t  in0_zp, in1_zp, out_zp, w_zp;
    int32_t  act_min, act_max;
    const int8_t  *w;
    const int32_t *bias;
    const int32_t *bias_folded;
    const int32_t *mult;             /* n_mult 個 (per output channel 或 1) */
    const int8_t  *shift;
    uint16_t n_mult;
    int8_t   left_shift;             /* ADD 用 */
    int32_t  in0_mult, in1_mult;
    int8_t   in0_shift, in1_shift;
    uint32_t golden_fnv1a;           /* golden 輸出的 FNV-1a 32-bit hash，Spike 上逐層檢查用 */
    uint32_t macs;
} layer_t;

#define N_LAYERS            64
#define N_TENSORS           65
#define ARENA_SIZE          1505280u
#define MODEL_INPUT_TENSOR  0
#define MODEL_OUTPUT_TENSOR 64

extern const layer_t  g_layers[N_LAYERS];
extern const tensor_t g_tensors[N_TENSORS];
extern int8_t g_arena[ARENA_SIZE];

#endif
