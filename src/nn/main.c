/*
 * main.c — 跑完整個 EfficientNet-Lite0 INT8
 *   1. 把測試影像放進 arena
 *   2. 照 g_layers[] 一層一層呼叫 kernel
 *   3. 每層算完：記錄指令數，並用 FNV-1a hash 跟 golden 比對
 *   4. 印出 FC 輸出的 top-1，跟 GOLDEN_TOP1 比對
 */
#include <stdint.h>
#include "model_data.h"
#include "test_input.h"
#include "kernels.h"
#include "platform.h"

static const char *op_name(int t)
{
    switch (t) {
    case OP_QUANTIZE:          return "QUANTIZE";
    case OP_CONV_2D:           return "CONV_2D";
    case OP_DEPTHWISE_CONV_2D: return "DWCONV";
    case OP_ADD:               return "ADD";
    case OP_AVERAGE_POOL_2D:   return "AVGPOOL";
    case OP_FULLY_CONNECTED:   return "FC";
    case OP_SOFTMAX:           return "SOFTMAX";
    case OP_RESHAPE:           return "RESHAPE";
    default:                   return "?";
    }
}

static uint32_t fnv1a(const void *p, uint32_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint32_t h = 0x811C9DC5u;
    for (uint32_t i = 0; i < n; i++)
        h = (h ^ b[i]) * 0x01000193u;
    return h;
}

#define PTR(t) ((int8_t *)&g_arena[g_tensors[t].off])

/* 依層數統計 (給 profiling 用) */
static uint64_t op_instret[8], op_cycles[8], op_macs[8];

int main(void)
{
    int n_fail = 0, last_fc_out = -1;
    uint64_t total_inst = 0, total_cyc = 0;

    /* 1. 輸入影像 */
    {
        const tensor_t *t = &g_tensors[MODEL_INPUT_TENSOR];
        uint8_t *dst = (uint8_t *)PTR(MODEL_INPUT_TENSOR);
        for (uint32_t i = 0; i < tensor_size(t); i++)
            dst[i] = (uint8_t)g_test_input[i];
    }

#if defined(USE_XQMAC) && defined(USE_QREQUANT)
    print_str("config: X2 (qmac8 + qrequant)\n");
#elif defined(USE_XQMAC)
    print_str("config: X1 (Xqmac qmac8 for 1x1 conv)\n");
#else
    print_str("config: B0 (baseline)\n");
#endif
    print_str("op  type       instret         cycles       MAC/inst*100  check\n");

    /* 2. 逐層執行 */
    for (int i = 0; i < N_LAYERS; i++) {
        const layer_t *L = &g_layers[i];
        const tensor_t *ti = &g_tensors[L->in0];
        const tensor_t *to = &g_tensors[L->out];
        int8_t *in = PTR(L->in0), *out = PTR(L->out);
        int skipped = 0;

        uint64_t i0 = plat_instret(), c0 = plat_cycle();
        switch (L->type) {
        case OP_QUANTIZE:          k_quantize(L, ti, to, in, out); break;
        case OP_CONV_2D:
#ifdef USE_XQMAC
            if (k_conv1x1_can_use(L, ti)) {          /* X1：1x1 conv 用 qmac8 */
                k_conv1x1_qmac8(L, ti, to, in, out);
                break;
            }
#endif
            k_conv2d(L, ti, to, in, out);            /* B0 */
            break;
        case OP_DEPTHWISE_CONV_2D: k_dwconv2d(L, ti, to, in, out); break;
        case OP_ADD:               k_add(L, to, in, PTR(L->in1), out); break;
        case OP_AVERAGE_POOL_2D:   k_avgpool(L, ti, to, in, out); break;
        case OP_FULLY_CONNECTED:   k_fc(L, ti, to, in, out); last_fc_out = L->out; break;
        case OP_RESHAPE:           k_copy(to, in, out); break;
        default:                   skipped = 1; break;   /* SOFTMAX：B0 先不做 */
        }
        uint64_t di = plat_instret() - i0, dc = plat_cycle() - c0;
        total_inst += di;
        total_cyc += dc;
        op_instret[L->type] += di;
        op_cycles[L->type] += dc;
        op_macs[L->type] += L->macs;

        /* 3. 比對 golden */
        const char *status;
        if (skipped) {
            status = "SKIP";
        } else if (fnv1a(out, tensor_size(to)) == L->golden_fnv1a) {
            status = "OK";
        } else {
            status = "FAIL";
            n_fail++;
        }

        print_u64_pad((uint64_t)i, 2);
        print_str("  ");
        print_pad(op_name(L->type), 9);
        print_u64_pad(di, 12);
        print_u64_pad(dc, 15);
        print_u64_pad(di ? (uint64_t)L->macs * 100 / di : 0, 14);
        print_str("  ");
        print_str(status);
        print_str("\n");
        if (skipped)
            break;   /* softmax 之後的層 (最後的 QUANTIZE) 依賴 softmax，B0 不跑 */
    }

    /* 依 op 種類彙總 */
    print_str("\n== per-op-type summary ==\ntype            instret   share         MACs\n");
    for (int t = 0; t < 8; t++) {
        if (!op_instret[t] && !op_macs[t])
            continue;
        uint64_t pm = total_inst ? op_instret[t] * 1000 / total_inst : 0;   /* 千分比 */
        print_pad(op_name(t), 9);
        print_u64_pad(op_instret[t], 14);
        print_u64_pad(pm / 10, 6);
        print_str(".");
        print_u64(pm % 10);
        print_str("%");
        print_u64_pad(op_macs[t], 13);
        print_str("\n");
    }

    /* 4. top-1 (取 FC 輸出，softmax 不影響排序) */
    int top1 = -1;
    if (last_fc_out >= 0) {
        const int8_t *y = PTR(last_fc_out);
        uint32_t n = tensor_size(&g_tensors[last_fc_out]);
        int best = -129;
        for (uint32_t k = 0; k < n; k++)
            if (y[k] > best) {
                best = y[k];
                top1 = (int)k;
            }
    }

    print_str("\ntotal instret = ");
    print_u64(total_inst);
    print_str("\ntotal cycles  = ");
    print_u64(total_cyc);
    print_str("\ntop-1 = ");
    print_i64(top1);
    print_str(" (golden ");
    print_i64(GOLDEN_TOP1);
    print_str(")\nlayer failures = ");
    print_i64(n_fail);
    print_str(n_fail == 0 && top1 == GOLDEN_TOP1 ? "\nRESULT: PASS\n" : "\nRESULT: FAIL\n");
    print_flush();
    plat_exit(n_fail == 0 && top1 == GOLDEN_TOP1 ? 0 : 1);
    return 0;
}
