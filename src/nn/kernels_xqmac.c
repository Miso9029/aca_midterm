/*
 * kernels_xqmac.c — X1：用 Xqmac 的 qmac8 指令加速 1x1 conv
 *   條件：kh=kw=1、stride=1、沒有 padding、IC 是 8 的倍數 (本模型 32 層 1x1 全部符合)
 *   不符合條件的層仍然走 B0 的 k_conv2d
 */
#include "kernels.h"
#include "qmath.h"
#include "xqmac.h"

typedef uint64_t u64a __attribute__((may_alias));   /* 用 64-bit 讀 int8 陣列 */

int k_conv1x1_can_use(const layer_t *L, const tensor_t *ti)
{
    return L->kh == 1 && L->kw == 1 && L->sh == 1 && L->sw == 1 &&
           L->pad_t == 0 && L->pad_l == 0 && (ti->c % 8) == 0;
}

/*
 * out[p][oc] = requant( bias_folded[oc] + Σ_ic w[oc][ic] * x[p][ic] )
 *   bias_folded = bias - in_zp * Σ w，所以內迴圈不用減 zero point
 */
void k_conv1x1_qmac8(const layer_t *L, const tensor_t *ti, const tensor_t *to,
                     const int8_t *in, int8_t *out)
{
    const int IC8 = ti->c / 8;
    const int OC = to->c;
    const int NPIX = to->h * to->w;

#ifdef USE_QREQUANT
    /* X2：每個輸出通道的 requant 參數先打包一次 (OC 最大 1280，約 10 KB stack) */
    uint64_t rq[OC];
    for (int oc = 0; oc < OC; oc++) {
        int c = L->n_mult > 1 ? oc : 0;
        rq[oc] = qrequant_pack(L->mult[c], L->shift[c],
                               L->out_zp, L->act_min, L->act_max);
    }
#endif

    for (int p = 0; p < NPIX; p++) {
        const u64a *x8 = (const u64a *)&in[p * ti->c];
        for (int oc = 0; oc < OC; oc++) {
            const u64a *w8 = (const u64a *)&L->w[oc * ti->c];
            int32_t acc = L->bias_folded[oc];
            for (int k = 0; k < IC8; k++)
                acc = qmac8(acc, x8[k], w8[k]);          /* ld, ld, qmac8 */
#ifdef USE_QREQUANT
            out[p * OC + oc] = (int8_t)qrequant(acc, rq[oc]);   /* X2：一條指令 */
#else
            int c = L->n_mult > 1 ? oc : 0;
            out[p * OC + oc] = (int8_t)q_requant(acc, L->mult[c], L->shift[c],
                                                 L->out_zp, L->act_min, L->act_max);
#endif
        }
    }
}
