/* kernels.h — B0 baseline kernel (純 C、無 FP、只用 rv64imc 指令) */
#ifndef KERNELS_H
#define KERNELS_H
#include <stdint.h>
#include "model_data.h"

/* 每個 kernel：L = 這層的描述，ti/ti1/to = 輸入/第二輸入/輸出 tensor 形狀，in/in1/out = 資料位置 */
void k_quantize(const layer_t *L, const tensor_t *ti, const tensor_t *to, const void *in, void *out);
void k_conv2d  (const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out);
void k_dwconv2d(const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out);
void k_add     (const layer_t *L, const tensor_t *to, const int8_t *a, const int8_t *b, int8_t *out);
void k_avgpool (const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out);
void k_fc      (const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out);
void k_copy    (const tensor_t *to, const int8_t *in, int8_t *out);

/* X1：Xqmac (kernels_xqmac.c) */
int  k_conv1x1_can_use(const layer_t *L, const tensor_t *ti);
void k_conv1x1_qmac8(const layer_t *L, const tensor_t *ti, const tensor_t *to, const int8_t *in, int8_t *out);

static inline uint32_t tensor_size(const tensor_t *t)
{
    return (uint32_t)t->n * t->h * t->w * t->c;
}

#endif
