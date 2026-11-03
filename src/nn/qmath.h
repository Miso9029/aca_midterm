/* qmath.h — TFLite 整數 requantization 運算 (double-rounding 版本，與 export_model.py 的 numpy reference 相同) */
#ifndef QMATH_H
#define QMATH_H
#include <stdint.h>

/* SaturatingRoundingDoublingHighMul：回傳 round(a*b / 2^31) */
static inline int32_t q_srdhm(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN)
        return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t nudge = ab >= 0 ? (1LL << 30) : (1 - (1LL << 30));
    return (int32_t)((ab + nudge) / (1LL << 31));   /* C 除法往 0 截斷，跟 TFLite 一樣 */
}

/* RoundingDivideByPOT：x / 2^e，四捨五入 (0.5 遠離 0) */
static inline int32_t q_rdbpot(int32_t x, int e)
{
    int32_t mask = (int32_t)((1LL << e) - 1);
    int32_t rem = x & mask;
    int32_t thr = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> e) + (rem > thr ? 1 : 0);
}

/* MultiplyByQuantizedMultiplier：x * mult * 2^shift (mult 是 Q31 定點數) */
static inline int32_t q_mbqm(int32_t x, int32_t mult, int shift)
{
    int left = shift > 0 ? shift : 0;
    int right = shift > 0 ? 0 : -shift;
    return q_rdbpot(q_srdhm((int32_t)((uint32_t)x << left), mult), right);
}

static inline int32_t q_clamp(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* 完整 requant：acc -> int8/uint8 */
static inline int32_t q_requant(int32_t acc, int32_t mult, int shift, int32_t out_zp, int32_t lo, int32_t hi)
{
    return q_clamp(q_mbqm(acc, mult, shift) + out_zp, lo, hi);
}

#endif
