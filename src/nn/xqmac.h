/*
 * xqmac.h — Xqmac 自訂指令的 C 介面
 *
 *   qmac8 rd, rs1, rs2     custom-0 (opcode 0x0B), funct3 = 0, funct7 = 0
 *     rd = sext32( rd + Σ(i=0..7) int8(rs1.byte[i]) * int8(rs2.byte[i]) )
 *
 * RISC-V：用 .insn 直接產生機器碼 (GCC 不認識 qmac8 這個名字)
 * x86   ：用 C 算一樣的結果，讓 make host 也能編譯、驗證 kernel 邏輯
 */
#ifndef XQMAC_H
#define XQMAC_H
#include <stdint.h>

/* 參考答案：永遠用 C 算 (測試時拿來跟真正的指令比對) */
static inline int32_t qmac8_ref(int32_t acc, uint64_t a, uint64_t b)
{
    for (int i = 0; i < 8; i++)
        acc += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return acc;
}

/* acc = acc + dot8(a, b) */
static inline int32_t qmac8(int32_t acc, uint64_t a, uint64_t b)
{
#ifdef __riscv
    int64_t r = acc;
    /* .insn r opcode, funct3, funct7, rd, rs1, rs2
     * "+r"：rd 既是輸入 (舊的 acc) 也是輸出 (新的 acc) */
    __asm__(".insn r 0x0B, 0, 0, %0, %1, %2" : "+r"(r) : "r"(a), "r"(b));
    return (int32_t)r;
#else
    return qmac8_ref(acc, a, b);
#endif
}




/*
 *   qrequant rd, rs1, rs2  custom-0 (opcode 0x0B), funct3 = 1, funct7 = 0
 *     rd = sext32( clamp( MultiplyByQuantizedMultiplier(rs1, mult, shift) + out_zp ) )
 *     rs2 = act_max[63:56] | act_min[55:48] | out_zp[47:40] | shift[39:32] | mult[31:0]
 */

/* 把 requant 參數打包成 rs2 */
static inline uint64_t qrequant_pack(int32_t mult, int shift, int out_zp,
                                     int act_min, int act_max)
{
    return  (uint64_t)(uint32_t)mult
          | ((uint64_t)(uint8_t)shift   << 32)
          | ((uint64_t)(uint8_t)out_zp  << 40)
          | ((uint64_t)(uint8_t)act_min << 48)
          | ((uint64_t)(uint8_t)act_max << 56);
}

/* 參考答案：TFLite MultiplyByQuantizedMultiplier + zp + clamp */
static inline int32_t qrequant_ref(int32_t acc, uint64_t p)
{
    int32_t mult = (int32_t)p;
    int shift    = (int8_t)(p >> 32);
    int out_zp   = (int8_t)(p >> 40);
    int act_min  = (int8_t)(p >> 48);
    int act_max  = (int8_t)(p >> 56);

    int left  = shift > 0 ? shift : 0;
    int right = shift > 0 ? 0 : -shift;
    int32_t x = (int32_t)((uint32_t)acc << left);

    int32_t hi;                                   /* SaturatingRoundingDoublingHighMul */
    if (x == INT32_MIN && mult == INT32_MIN) {
        hi = INT32_MAX;
    } else {
        int64_t ab = (int64_t)x * mult;
        int64_t nudge = ab >= 0 ? (1LL << 30) : (1 - (1LL << 30));
        hi = (int32_t)((ab + nudge) / (1LL << 31));
    }

    int32_t mask = (int32_t)((1LL << right) - 1); /* RoundingDivideByPOT */
    int32_t rem  = hi & mask;
    int32_t thr  = (mask >> 1) + (hi < 0 ? 1 : 0);
    int32_t v    = (hi >> right) + (rem > thr ? 1 : 0);

    v += out_zp;
    if (v < act_min) v = act_min;
    if (v > act_max) v = act_max;
    return v;
}

/* requant 一個 int32 累加值，回傳 int8 範圍內的結果 */
static inline int32_t qrequant(int32_t acc, uint64_t p)
{
#ifdef __riscv
    int64_t r;
    __asm__(".insn r 0x0B, 1, 0, %0, %1, %2" : "=r"(r) : "r"((int64_t)acc), "r"(p));
    return (int32_t)r;
#else
    return qrequant_ref(acc, p);
#endif
}
#endif