/*
 * kernels.c — B0 baseline：直接照 TFLite reference kernel 的定義寫，沒有任何最佳化。
 * 之後 B1 (軟體最佳化) / X1 (Xqmac8) 都拿這份當正確答案對照。
 * 資料排列一律 NHWC (batch=1)：x[y][x][c]
 */
#include "kernels.h"
#include "qmath.h"

#define CH(L, c) ((L)->n_mult > 1 ? (c) : 0)   /* per-channel 或 per-tensor 參數 */

/* QUANTIZE：uint8 <-> int8 (或 scale 改變) */
void k_quantize(const layer_t *L, const tensor_t *ti, const tensor_t *to, const void *in, void *out)
{
    uint32_t n = tensor_size(to);
    for (uint32_t i = 0; i < n; i++) {
        int32_t x = ti->is_uint8 ? ((const uint8_t *)in)[i] : ((const int8_t *)in)[i];
        int32_t v = q_requant(x - L->in0_zp, L->mult[0], L->shift[0], L->out_zp, L->act_min, L->act_max);
        if (to->is_uint8)
            ((uint8_t *)out)[i] = (uint8_t)v;
        else
            ((int8_t *)out)[i] = (int8_t)v;
    }
}

/*
 * CONV_2D (一般版，1x1 和 stem 3x3 都用這個)
 *   w 排列 [OC][KH][KW][IC]
 *   out[oy][ox][oc] = requant( bias[oc] + Σ_ky,kx,ic w[oc][ky][kx][ic] * (x[iy][ix][ic] - in_zp) )
 *   超出邊界的 (iy,ix) 直接跳過 (= padding 值等於 in_zp)
 */
void k_conv2d(const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out)
{
    const int IH = ti->h, IW = ti->w, IC = ti->c;
    const int OH = to->h, OW = to->w, OC = to->c;
    const int KH = L->kh, KW = L->kw;
    const int32_t zp = L->in0_zp;

    for (int oy = 0; oy < OH; oy++) {
        for (int ox = 0; ox < OW; ox++) {
            for (int oc = 0; oc < OC; oc++) {
                int32_t acc = L->bias[oc];
                for (int ky = 0; ky < KH; ky++) {
                    int iy = oy * L->sh - L->pad_t + ky * L->dh;
                    if (iy < 0 || iy >= IH)
                        continue;
                    for (int kx = 0; kx < KW; kx++) {
                        int ix = ox * L->sw - L->pad_l + kx * L->dw;
                        if (ix < 0 || ix >= IW)
                            continue;
                        const int8_t *xp = &in[(iy * IW + ix) * IC];
                        const int8_t *wp = &L->w[((oc * KH + ky) * KW + kx) * IC];
                        for (int ic = 0; ic < IC; ic++)
                            acc += (int32_t)wp[ic] * ((int32_t)xp[ic] - zp);
                    }
                }
                out[(oy * OW + ox) * OC + oc] = (int8_t)q_requant(
                    acc, L->mult[CH(L, oc)], L->shift[CH(L, oc)], L->out_zp, L->act_min, L->act_max);
            }
        }
    }
}

/*
 * DEPTHWISE_CONV_2D
 *   w 排列 [KH][KW][OC]，OC = IC * depth_mult，輸出通道 oc 對應輸入通道 oc / depth_mult
 */
void k_dwconv2d(const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out)
{
    const int IH = ti->h, IW = ti->w, IC = ti->c;
    const int OH = to->h, OW = to->w, OC = to->c;
    const int KH = L->kh, KW = L->kw, M = L->depth_mult;
    const int32_t zp = L->in0_zp;

    for (int oy = 0; oy < OH; oy++) {
        for (int ox = 0; ox < OW; ox++) {
            for (int oc = 0; oc < OC; oc++) {
                int ic = oc / M;
                int32_t acc = L->bias[oc];
                for (int ky = 0; ky < KH; ky++) {
                    int iy = oy * L->sh - L->pad_t + ky * L->dh;
                    if (iy < 0 || iy >= IH)
                        continue;
                    for (int kx = 0; kx < KW; kx++) {
                        int ix = ox * L->sw - L->pad_l + kx * L->dw;
                        if (ix < 0 || ix >= IW)
                            continue;
                        acc += (int32_t)L->w[(ky * KW + kx) * OC + oc] *
                               ((int32_t)in[(iy * IW + ix) * IC + ic] - zp);
                    }
                }
                out[(oy * OW + ox) * OC + oc] = (int8_t)q_requant(
                    acc, L->mult[CH(L, oc)], L->shift[CH(L, oc)], L->out_zp, L->act_min, L->act_max);
            }
        }
    }
}

/*
 * ADD (residual)：兩個輸入 scale 不同，先各自 rescale 到共同尺度再相加
 */
void k_add(const layer_t *L, const tensor_t *to, const int8_t *a, const int8_t *b, int8_t *out)
{
    uint32_t n = tensor_size(to);
    const int ls = L->left_shift;
    for (uint32_t i = 0; i < n; i++) {
        int32_t sa = q_mbqm((int32_t)((uint32_t)(a[i] - L->in0_zp) << ls), L->in0_mult, L->in0_shift);
        int32_t sb = q_mbqm((int32_t)((uint32_t)(b[i] - L->in1_zp) << ls), L->in1_mult, L->in1_shift);
        out[i] = (int8_t)q_requant(sa + sb, L->mult[0], L->shift[0], L->out_zp, L->act_min, L->act_max);
    }
}

/*
 * AVERAGE_POOL_2D：輸入輸出同 scale/zp，不需要 requant；除法四捨五入 (0.5 遠離 0)
 */
void k_avgpool(const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out)
{
    const int IH = ti->h, IW = ti->w, C = ti->c;
    const int OH = to->h, OW = to->w;
    for (int oy = 0; oy < OH; oy++) {
        for (int ox = 0; ox < OW; ox++) {
            for (int c = 0; c < C; c++) {
                int32_t acc = 0, cnt = 0;
                for (int ky = 0; ky < L->kh; ky++) {
                    int iy = oy * L->sh - L->pad_t + ky;
                    if (iy < 0 || iy >= IH)
                        continue;
                    for (int kx = 0; kx < L->kw; kx++) {
                        int ix = ox * L->sw - L->pad_l + kx;
                        if (ix < 0 || ix >= IW)
                            continue;
                        acc += in[(iy * IW + ix) * C + c];
                        cnt++;
                    }
                }
                acc = acc > 0 ? (acc + cnt / 2) / cnt : (acc - cnt / 2) / cnt;
                out[(oy * OW + ox) * C + c] = (int8_t)q_clamp(acc, L->act_min, L->act_max);
            }
        }
    }
}

/*
 * FULLY_CONNECTED：w 排列 [OC][IC]
 */
void k_fc(const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out)
{
    const int IC = (int)tensor_size(ti);
    const int OC = (int)tensor_size(to);
    for (int oc = 0; oc < OC; oc++) {
        int32_t acc = L->bias[oc];
        const int8_t *wp = &L->w[oc * IC];
        for (int ic = 0; ic < IC; ic++)
            acc += ((int32_t)wp[ic] - L->w_zp) * ((int32_t)in[ic] - L->in0_zp);
        out[oc] = (int8_t)q_requant(acc, L->mult[CH(L, oc)], L->shift[CH(L, oc)], L->out_zp,
                                    L->act_min, L->act_max);
    }
}

/* RESHAPE：資料不變，只是換形狀 → 複製過去 */
void k_copy(const tensor_t *to, const int8_t *in, int8_t *out)
{
    uint32_t n = tensor_size(to);
    if (in == out)
        return;
    for (uint32_t i = 0; i < n; i++)
        out[i] = in[i];
}
