#!/usr/bin/env python3
"""
export_model.py — EfficientNet-Lite0 INT8 .tflite  ->  static C data + golden reference

輸出:
  <out-dir>/model_data.h     layer/tensor 描述結構、巨集、extern 宣告
  <out-dir>/model_data.c     所有 weight / bias / requant 參數 (static const) + activation arena
  <out-dir>/test_input.h     測試輸入影像 (uint8)，給 Spike 上跑
  <golden-dir>/              每個 op 輸出的 golden .bin + manifest.json (x86 逐層比對用)

功能:
  1. 用 LiteRT 的 BUILTIN_REF kernel (= TFLite reference integer kernel，不走 XNNPACK) 跑一次，
     保留所有中間 tensor 當 golden。
  2. 把 scale 離線換成 TFLite 的 (int32 multiplier, shift)，C 端只需要整數運算。
  3. 用 numpy 寫的整數 reference (與 TFLite reference kernel 同一套 rounding) 逐層重算，
     確認匯出的參數是 bit-exact，同時當作 C kernel 的規格。
  4. 依 tensor 生命週期做靜態記憶體規劃 (單一 arena)。

用法 (在 ~/project 底下，venv 已啟用):
  uv pip install tflite            # 只多這一個套件 (flatbuffer schema，用來讀 stride/padding/activation)
  python scripts/export_model.py --model models/efficientnet-lite0/int8/model.tflite \
         --image some_cat.jpg      # 可省略，省略時用固定 seed 的亂數影像
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import tflite
except ImportError:
    sys.exit("缺少 tflite 套件：請先執行  uv pip install tflite")

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    sys.exit("缺少 ai-edge-litert：請先執行  uv pip install ai-edge-litert")

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1

BUILTIN = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}
ACT = {v: k for k, v in vars(tflite.ActivationFunctionType).items() if not k.startswith("_")}

# C 端 op 編號 (要跟 model_data.h 的 enum 一致)
OP_IDS = ["QUANTIZE", "CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "AVERAGE_POOL_2D",
          "FULLY_CONNECTED", "SOFTMAX", "RESHAPE"]


# ---------------------------------------------------------------------------
# TFLite 定點運算 (與 tensorflow/lite/kernels/internal/common.h 相同語意)
# ---------------------------------------------------------------------------
def round_away(x):
    """TfLiteRound / std::round: 0.5 遠離 0"""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def quantize_multiplier(m):
    """QuantizeMultiplier(double) -> (int32 multiplier in [2^30, 2^31), shift)"""
    if m == 0.0:
        return 0, 0
    q, shift = math.frexp(m)
    q_fixed = round_away(q * (1 << 31))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    if shift > 30:
        raise ValueError(f"multiplier {m} 太大")
    return int(q_fixed), int(shift)


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul(int32 a, int32 b)"""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    ab = a * b
    t = ab + np.where(ab >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    res = np.where(t >= 0, t >> 31, -((-t) >> 31))      # C 的除法是往 0 截斷
    return np.where((a == INT32_MIN) & (b == INT32_MIN), np.int64(INT32_MAX), res)


def rdbpot(x, e):
    """RoundingDivideByPOT(x, exponent)"""
    x = np.asarray(x, np.int64)
    e = np.asarray(e, np.int64)
    mask = (np.int64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> e) + (rem > thr)


def mbqm(x, mult, shift):
    """MultiplyByQuantizedMultiplier(int32 x, int32 mult, int shift)  (double-rounding 版本)"""
    shift = np.asarray(shift, np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    return rdbpot(srdhm(np.asarray(x, np.int64) << left, mult), right)


def qrange(dtype):
    return (-128, 127) if dtype == np.int8 else (0, 255)


def act_range(fused, scale, zp, dtype):
    """CalculateActivationRangeQuantized"""
    qmin, qmax = qrange(dtype)

    def q(f):
        return int(zp) + round_away(float(np.float32(f) / np.float32(scale)))

    name = ACT.get(fused, "NONE")
    if name == "NONE":
        return qmin, qmax
    if name == "RELU":
        return max(qmin, q(0.0)), qmax
    if name == "RELU6":
        return max(qmin, q(0.0)), min(qmax, q(6.0))
    if name == "RELU_N1_TO_1":
        return max(qmin, q(-1.0)), min(qmax, q(1.0))
    raise NotImplementedError(f"fused activation {name}")


def compute_padding(in_size, k, stride, dil, padding_same):
    eff = (k - 1) * dil + 1
    if padding_same:
        out = (in_size + stride - 1) // stride
    else:
        out = (in_size - eff + stride) // stride
    total = max((out - 1) * stride + eff - in_size, 0)
    return out, total // 2          # pad_before (多出來的 1 放在下/右)


# ---------------------------------------------------------------------------
# numpy integer reference kernels (= C kernel 的規格)
# ---------------------------------------------------------------------------
def pad_and_window(x, pt, pl, oh, ow, kh, kw, sh, sw, dh, dw):
    """x: (N,H,W,C) 已減掉 zero point 的 int64。回傳 generator: (ky,kx, (N,OH,OW,C) 視窗)"""
    n, h, w, c = x.shape
    hp = (oh - 1) * sh + (kh - 1) * dh + 1
    wp = (ow - 1) * sw + (kw - 1) * dw + 1
    xp = np.zeros((n, max(hp, h + pt), max(wp, w + pl), c), x.dtype)   # pad 值 = 0 (= 原本的 zero point)
    xp[:, pt:pt + h, pl:pl + w, :] = x
    for ky in range(kh):
        for kx in range(kw):
            y0, x0 = ky * dh, kx * dw
            yield ky, kx, xp[:, y0:y0 + (oh - 1) * sh + 1:sh, x0:x0 + (ow - 1) * sw + 1:sw, :]


def ref_conv(L, x):
    xo = x.astype(np.int64) - L["in0_zp"]
    w = L["w"].astype(np.int64)                       # (O,KH,KW,I)
    n = x.shape[0]
    oh, ow = L["out_shape"][1], L["out_shape"][2]
    acc = np.zeros((n, oh, ow, w.shape[0]), np.float64)
    for ky, kx, win in pad_and_window(xo, L["pad_t"], L["pad_l"], oh, ow, L["kh"], L["kw"],
                                      L["sh"], L["sw"], L["dh"], L["dw"]):
        # float64 matmul 是精確的：|w|<=128, |x|<=255, I<=1280 -> 總和 < 2^53
        acc += win.astype(np.float64) @ w[:, ky, kx, :].T.astype(np.float64)
    acc = acc.astype(np.int64) + L["bias"]
    out = mbqm(acc, L["mult"], L["shift"]) + L["out_zp"]
    return np.clip(out, L["act_min"], L["act_max"])


def ref_dwconv(L, x):
    xo = x.astype(np.int64) - L["in0_zp"]
    m = L["depth_mult"]
    if m > 1:
        xo = np.repeat(xo, m, axis=-1)                # out channel = ic*M + m
    w = L["w"].astype(np.int64)                       # (1,KH,KW,C*M)
    oh, ow = L["out_shape"][1], L["out_shape"][2]
    acc = np.zeros((x.shape[0], oh, ow, w.shape[3]), np.int64)
    for ky, kx, win in pad_and_window(xo, L["pad_t"], L["pad_l"], oh, ow, L["kh"], L["kw"],
                                      L["sh"], L["sw"], L["dh"], L["dw"]):
        acc += win * w[0, ky, kx, :]
    acc += L["bias"]
    out = mbqm(acc, L["mult"], L["shift"]) + L["out_zp"]
    return np.clip(out, L["act_min"], L["act_max"])


def ref_add(L, a, b):
    ls = L["left_shift"]
    sa = mbqm((a.astype(np.int64) - L["in0_zp"]) << ls, L["in0_mult"], L["in0_shift"])
    sb = mbqm((b.astype(np.int64) - L["in1_zp"]) << ls, L["in1_mult"], L["in1_shift"])
    out = mbqm(sa + sb, L["mult"][0], L["shift"][0]) + L["out_zp"]
    return np.clip(out, L["act_min"], L["act_max"])


def ref_avgpool(L, x):
    xi = x.astype(np.int64)
    n, h, w, c = xi.shape
    oh, ow = L["out_shape"][1], L["out_shape"][2]
    acc = np.zeros((n, oh, ow, c), np.int64)
    cnt = np.zeros((1, oh, ow, 1), np.int64)
    ones = np.ones((1, h, w, 1), np.int64)
    args = (L["pad_t"], L["pad_l"], oh, ow, L["kh"], L["kw"], L["sh"], L["sw"], 1, 1)
    for (_, _, win), (_, _, mwin) in zip(pad_and_window(xi, *args), pad_and_window(ones, *args)):
        acc += win * mwin
        cnt += mwin
    num = np.where(acc > 0, acc + cnt // 2, acc - cnt // 2)
    out = np.sign(num) * (np.abs(num) // cnt)          # C 的整數除法 (往 0 截斷)
    return np.clip(out, L["act_min"], L["act_max"])


def ref_fc(L, x):
    w = L["w"].astype(np.float64)                      # (O,I)
    xi = x.reshape(-1, w.shape[1]).astype(np.float64) - L["in0_zp"]
    acc = (xi @ (w - L["w_zp"]).T).astype(np.int64) + L["bias"]
    out = mbqm(acc, L["mult"], L["shift"]) + L["out_zp"]
    return np.clip(out, L["act_min"], L["act_max"]).reshape(L["out_shape"])


def ref_quantize(L, x):
    out = mbqm(x.astype(np.int64) - L["in0_zp"], L["mult"][0], L["shift"][0]) + L["out_zp"]
    return np.clip(out, L["act_min"], L["act_max"])


# ---------------------------------------------------------------------------
# 讀模型
# ---------------------------------------------------------------------------
def load_image(path, shape, dtype, seed):
    _, h, w, c = shape
    if path:
        try:
            from PIL import Image
        except ImportError:
            sys.exit("要讀圖片需要 Pillow：uv pip install pillow")
        img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)[None]
        if dtype == np.int8:
            arr = (arr.astype(np.int16) - 128).astype(np.int8)
        return arr
    rng = np.random.default_rng(seed)
    lo, hi = qrange(dtype)
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int64).astype(dtype)


def get_opts(op, cls):
    t = op.BuiltinOptions()
    if t is None:
        return None
    o = cls()
    o.Init(t.Bytes, t.Pos)
    return o


def fnv1a(arr):
    h = 0x811C9DC5
    for b in np.ascontiguousarray(arr).view(np.uint8).ravel().tolist():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def build(args):
    with open(args.model, "rb") as f:
        buf = f.read()
    model = tflite.Model.GetRootAsModel(buf, 0)
    if model.SubgraphsLength() != 1:
        print("警告：模型有多個 subgraph，只處理第 0 個")
    sg = model.Subgraphs(0)

    interp = Interpreter(model_path=args.model,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                         experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    det = {d["index"]: d for d in interp.get_tensor_details()}

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    x_in = load_image(args.image, tuple(in_det["shape"]), in_det["dtype"], args.seed)
    interp.set_tensor(in_det["index"], x_in)
    interp.invoke()

    def T(i):
        return interp.get_tensor(int(i))

    def qp(i):
        q = det[int(i)]["quantization_parameters"]
        return q["scales"].astype(np.float32), q["zero_points"].astype(np.int64)

    def dtype(i):
        return det[int(i)]["dtype"]

    def shape(i):
        return tuple(int(s) for s in det[int(i)]["shape"])

    layers = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = model.OperatorCodes(op.OpcodeIndex())
        code = max(getattr(oc, "BuiltinCode", lambda: 0)(), oc.DeprecatedBuiltinCode())
        name = BUILTIN.get(code, f"OP{code}")
        ins = [int(t) for t in op.InputsAsNumpy()]
        outs = [int(t) for t in op.OutputsAsNumpy()]
        o = outs[0]
        so, zo = qp(o)
        L = dict(idx=i, op=name, inputs=ins, in0=ins[0], in1=-1, out=o,
                 in_shape=shape(ins[0]), out_shape=shape(o),
                 in0_zp=int(qp(ins[0])[1][0]) if len(qp(ins[0])[1]) else 0,
                 in1_zp=0, out_zp=int(zo[0]) if len(zo) else 0,
                 kh=1, kw=1, sh=1, sw=1, dh=1, dw=1, pad_t=0, pad_l=0, depth_mult=1,
                 left_shift=0, in0_mult=0, in0_shift=0, in1_mult=0, in1_shift=0,
                 w=None, w_zp=0, bias=None, mult=np.zeros(1, np.int64), shift=np.zeros(1, np.int64),
                 macs=0, note="")
        si = qp(ins[0])[0]

        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            dw_ = name == "DEPTHWISE_CONV_2D"
            opt = get_opts(op, tflite.DepthwiseConv2DOptions if dw_ else tflite.Conv2DOptions)
            w = T(ins[1]).astype(np.int8)
            sw_scales, wzp = qp(ins[1])
            if np.any(wzp != 0):
                print(f"警告：op {i} weight zero point 不為 0")
            n_out = w.shape[3] if dw_ else w.shape[0]
            kh, kw = (w.shape[1], w.shape[2])
            bias = T(ins[2]).astype(np.int64) if len(ins) > 2 and ins[2] >= 0 else np.zeros(n_out, np.int64)
            sws = np.broadcast_to(sw_scales, (n_out,))
            mults, shifts = zip(*[quantize_multiplier(float(si[0]) * float(s) / float(so[0])) for s in sws])
            _, h, wd, _ = L["in_shape"]
            same = opt.Padding() == tflite.Padding.SAME
            oh, pt = compute_padding(h, kh, opt.StrideH(), opt.DilationHFactor(), same)
            ow, pl = compute_padding(wd, kw, opt.StrideW(), opt.DilationWFactor(), same)
            assert (oh, ow) == L["out_shape"][1:3], f"op {i} 輸出尺寸對不上"
            L.update(w=w, bias=bias, mult=np.array(mults, np.int64), shift=np.array(shifts, np.int64),
                     kh=kh, kw=kw, sh=opt.StrideH(), sw=opt.StrideW(),
                     dh=opt.DilationHFactor(), dw=opt.DilationWFactor(), pad_t=pt, pad_l=pl)
            if dw_:
                L["depth_mult"] = max(opt.DepthMultiplier(), 1)
                L["macs"] = oh * ow * n_out * kh * kw
                wsum = w.astype(np.int64).sum(axis=(0, 1, 2))
            else:
                L["macs"] = oh * ow * n_out * kh * kw * w.shape[3]
                wsum = w.astype(np.int64).sum(axis=(1, 2, 3))
            # bias_folded：把 -zp_in * sum(w) 先併進 bias，kernel 內迴圈就只剩 sum(w*x)
            # 注意：有 padding 的地方，padding 值必須填 zp_in (不是 0) 才會正確
            L["bias_folded"] = bias - L["in0_zp"] * wsum
            L["fused"] = opt.FusedActivationFunction()

        elif name == "FULLY_CONNECTED":
            opt = get_opts(op, tflite.FullyConnectedOptions)
            w = T(ins[1]).astype(np.int8)
            sw_scales, wzp = qp(ins[1])
            n_out = w.shape[0]
            bias = T(ins[2]).astype(np.int64) if len(ins) > 2 and ins[2] >= 0 else np.zeros(n_out, np.int64)
            if len(sw_scales) == 1:
                # GetQuantizedConvolutionMultipler: input*filter 先用 float32 乘
                m = float(np.float32(si[0]) * np.float32(sw_scales[0])) / float(so[0])
                mults, shifts = [quantize_multiplier(m)[0]], [quantize_multiplier(m)[1]]
            else:
                mults, shifts = zip(*[quantize_multiplier(float(si[0]) * float(s) / float(so[0])) for s in sw_scales])
            L.update(w=w, w_zp=int(wzp[0]) if len(wzp) else 0, bias=bias,
                     mult=np.array(mults, np.int64), shift=np.array(shifts, np.int64))
            L["bias_folded"] = bias - L["in0_zp"] * w.astype(np.int64).sum(axis=1)
            L["macs"] = n_out * w.shape[1]
            L["fused"] = opt.FusedActivationFunction() if opt else 0

        elif name == "ADD":
            opt = get_opts(op, tflite.AddOptions)
            s1, z1 = qp(ins[1])
            ls = 20
            twice_max = 2.0 * float(max(si[0], s1[0]))
            m0, sh0 = quantize_multiplier(float(si[0]) / twice_max)
            m1, sh1 = quantize_multiplier(float(s1[0]) / twice_max)
            mo, sho = quantize_multiplier(twice_max / ((1 << ls) * float(so[0])))
            assert shape(ins[0]) == shape(ins[1]), f"op {i} ADD 有 broadcast，未支援"
            L.update(in1=ins[1], in1_zp=int(z1[0]), left_shift=ls, in0_mult=m0, in0_shift=sh0,
                     in1_mult=m1, in1_shift=sh1, mult=np.array([mo]), shift=np.array([sho]))
            L["fused"] = opt.FusedActivationFunction() if opt else 0

        elif name == "AVERAGE_POOL_2D":
            opt = get_opts(op, tflite.Pool2DOptions)
            _, h, wd, _ = L["in_shape"]
            same = opt.Padding() == tflite.Padding.SAME
            oh, pt = compute_padding(h, opt.FilterHeight(), opt.StrideH(), 1, same)
            ow, pl = compute_padding(wd, opt.FilterWidth(), opt.StrideW(), 1, same)
            if not (np.isclose(si[0], so[0]) and L["in0_zp"] == L["out_zp"]):
                print(f"警告：op {i} AVERAGE_POOL 輸入輸出 quantization 不同，TFLite 不會 requant")
            L.update(kh=opt.FilterHeight(), kw=opt.FilterWidth(), sh=opt.StrideH(), sw=opt.StrideW(),
                     pad_t=pt, pad_l=pl)
            L["fused"] = opt.FusedActivationFunction()

        elif name == "QUANTIZE":
            m, s = quantize_multiplier(float(si[0]) / float(so[0]))
            L.update(mult=np.array([m]), shift=np.array([s]))
            L["fused"] = 0

        elif name in ("SOFTMAX", "RESHAPE", "SQUEEZE"):
            L["fused"] = 0
            if name == "SQUEEZE":
                L["op"] = "RESHAPE"
            if name == "SOFTMAX":
                L["note"] = "int8 softmax 不在 numpy reference 裡 (只比 golden)；top-1 用 FC 輸出 argmax 即可"
        else:
            L["fused"] = 0
            L["note"] = "未支援的 op"
            print(f"警告：op {i} {name} 未支援")

        L["act_min"], L["act_max"] = act_range(L["fused"], so[0] if len(so) else 1.0, L["out_zp"], dtype(o))
        L["in_dtype"], L["out_dtype"] = dtype(ins[0]), dtype(o)
        layers.append(L)

    model_in = [int(t) for t in sg.InputsAsNumpy()]
    model_out = [int(t) for t in sg.OutputsAsNumpy()]
    return interp, det, layers, model_in, model_out, x_in, T, out_det


# ---------------------------------------------------------------------------
# 驗證
# ---------------------------------------------------------------------------
def verify(layers, T):
    print(f"\n{'op':>3} {'type':<18} {'in -> out':<34} {'MACs':>11}  verify")
    total_macs, all_ok = 0, True
    for L in layers:
        x = T(L["in0"])
        g = T(L["out"])
        op = L["op"]
        ref = None
        if op == "CONV_2D":
            ref = ref_conv(L, x)
        elif op == "DEPTHWISE_CONV_2D":
            ref = ref_dwconv(L, x)
        elif op == "ADD":
            ref = ref_add(L, x, T(L["in1"]))
        elif op == "AVERAGE_POOL_2D":
            ref = ref_avgpool(L, x)
        elif op == "FULLY_CONNECTED":
            ref = ref_fc(L, x)
        elif op == "QUANTIZE":
            ref = ref_quantize(L, x)
        elif op == "RESHAPE":
            ref = x.reshape(g.shape)
        if ref is None:
            status = "skip (" + (L["note"] or "no ref") + ")"
        else:
            diff = np.abs(ref.reshape(g.shape).astype(np.int64) - g.astype(np.int64))
            n_bad = int((diff != 0).sum())
            L["verified"] = n_bad == 0
            status = "OK bit-exact" if n_bad == 0 else f"MISMATCH {n_bad}/{diff.size} max|d|={diff.max()}"
            all_ok &= n_bad == 0
        total_macs += L["macs"]
        io = f"{'x'.join(map(str, L['in_shape'][1:]))} -> {'x'.join(map(str, L['out_shape'][1:]))}"
        print(f"{L['idx']:>3} {op:<18} {io:<34} {L['macs']:>11,}  {status}")
    print(f"\n總 MACs: {total_macs:,}   全部 bit-exact: {all_ok}")
    return all_ok


# ---------------------------------------------------------------------------
# 靜態記憶體規劃
# ---------------------------------------------------------------------------
def plan_memory(layers, model_in, model_out, det, align=64):
    act = {}            # tensor -> [first, last]
    for t in model_in:
        act[t] = [-1, -1]
    for L in layers:
        act.setdefault(L["out"], [L["idx"], L["idx"]])
        for t in (L["in0"], L["in1"]):
            if t in act:
                act[t][1] = max(act[t][1], L["idx"])
    for t in model_out:
        act[t][1] = len(layers)

    def nbytes(t):
        return int(np.prod(det[t]["shape"])) * np.dtype(det[t]["dtype"]).itemsize

    placed, offs = [], {}
    for t in sorted(act, key=lambda t: -nbytes(t)):
        a0, a1 = act[t]
        size = nbytes(t)
        busy = sorted((offs[p], offs[p] + nbytes(p)) for p in placed
                      if not (act[p][1] < a0 or a1 < act[p][0]))
        off = 0
        for s, e in busy:
            if off + size <= s:
                break
            off = max(off, (e + align - 1) // align * align)
        offs[t] = off
        placed.append(t)
    arena = max(offs[t] + nbytes(t) for t in act)
    arena = (arena + align - 1) // align * align
    return act, offs, arena


# ---------------------------------------------------------------------------
# 產生 C 檔
# ---------------------------------------------------------------------------
def c_array(ctype, name, arr, per_line=32):
    vals = np.asarray(arr).ravel().tolist()
    lines = [", ".join(str(v) for v in vals[k:k + per_line]) for k in range(0, len(vals), per_line)]
    body = ",\n    ".join(lines)
    return f"const {ctype} {name}[{len(vals)}] __attribute__((aligned(8))) = {{\n    {body}\n}};\n"


def nhwc(shape):
    s = list(shape)
    while len(s) < 4:
        s.insert(1, 1)
    return s


def emit_c(layers, act, offs, arena, model_in, model_out, det, T, x_in, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tids = {t: k for k, t in enumerate(sorted(act, key=lambda t: (act[t][0], t)))}

    hdr = f"""/* 由 scripts/export_model.py 自動產生，請勿手動修改 */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#include <stdint.h>

typedef enum {{
{chr(10).join(f"    OP_{n} = {k}," for k, n in enumerate(OP_IDS))}
}} op_type_t;

/* activation tensor：資料放在 g_arena[off]，layout NHWC，int8 (is_uint8=1 時為 uint8) */
typedef struct {{
    uint32_t off;
    uint16_t n, h, w, c;
    uint8_t  is_uint8;
}} tensor_t;

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
typedef struct {{
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
}} layer_t;

#define N_LAYERS            {len(layers)}
#define N_TENSORS           {len(tids)}
#define ARENA_SIZE          {arena}u
#define MODEL_INPUT_TENSOR  {tids[model_in[0]]}
#define MODEL_OUTPUT_TENSOR {tids[model_out[0]]}

extern const layer_t  g_layers[N_LAYERS];
extern const tensor_t g_tensors[N_TENSORS];
extern int8_t g_arena[ARENA_SIZE];

#endif
"""
    with open(os.path.join(out_dir, "model_data.h"), "w") as f:
        f.write(hdr)

    with open(os.path.join(out_dir, "model_data.c"), "w") as f:
        f.write('/* 由 scripts/export_model.py 自動產生，請勿手動修改 */\n#include "model_data.h"\n\n')
        f.write(f"int8_t g_arena[ARENA_SIZE] __attribute__((aligned(64)));\n\n")
        for L in layers:
            p = f"L{L['idx']:02d}"
            if L["w"] is not None:
                w = L["w"]
                if L["op"] == "DEPTHWISE_CONV_2D":
                    w = w.reshape(w.shape[1:])
                f.write(f"/* op {L['idx']} {L['op']} w{list(w.shape)} */\n")
                f.write(c_array("int8_t", f"{p}_w", w))
                f.write(c_array("int32_t", f"{p}_bias", L["bias"]))
                f.write(c_array("int32_t", f"{p}_bias_folded", L["bias_folded"]))
            f.write(c_array("int32_t", f"{p}_mult", L["mult"]))
            f.write(c_array("int8_t", f"{p}_shift", L["shift"]))
            f.write("\n")

        f.write("const tensor_t g_tensors[N_TENSORS] = {\n")
        for t, k in sorted(tids.items(), key=lambda kv: kv[1]):
            n, h, w, c = nhwc(det[t]["shape"])
            u8 = int(det[t]["dtype"] == np.uint8)
            f.write(f"    /* {k:2d} '{det[t]['name'][:40]}' */ {{ {offs[t]}u, {n}, {h}, {w}, {c}, {u8} }},\n")
        f.write("};\n\nconst layer_t g_layers[N_LAYERS] = {\n")
        for L in layers:
            p = f"L{L['idx']:02d}"
            has_w = L["w"] is not None
            opname = L["op"] if L["op"] in OP_IDS else "RESHAPE"
            g = T(L["out"])
            f.write(f"    {{ /* {L['idx']} {L['op']} */ OP_{opname}, {tids.get(L['in0'], -1)}, "
                    f"{tids.get(L['in1'], -1)}, {tids[L['out']]},\n"
                    f"      {L['kh']}, {L['kw']}, {L['sh']}, {L['sw']}, {L['dh']}, {L['dw']}, "
                    f"{L['pad_t']}, {L['pad_l']}, {L['depth_mult']},\n"
                    f"      {L['in0_zp']}, {L['in1_zp']}, {L['out_zp']}, {L['w_zp']}, "
                    f"{L['act_min']}, {L['act_max']},\n"
                    f"      {p + '_w' if has_w else '0'}, {p + '_bias' if has_w else '0'}, "
                    f"{p + '_bias_folded' if has_w else '0'}, {p}_mult, {p}_shift, {len(L['mult'])},\n"
                    f"      {L['left_shift']}, {L['in0_mult']}, {L['in1_mult']}, {L['in0_shift']}, "
                    f"{L['in1_shift']}, 0x{fnv1a(g):08x}u, {L['macs']}u }},\n")
        f.write("};\n")

    # top-1 取最後一個 FC 的輸出 (C 端 B0 不做 softmax，比的也是 FC 輸出)
    fcs = [L for L in layers if L["op"] == "FULLY_CONNECTED"]
    out_g = T(fcs[-1]["out"] if fcs else model_out[0]).astype(np.int64)
    with open(os.path.join(out_dir, "test_input.h"), "w") as f:
        f.write("/* 由 scripts/export_model.py 自動產生 */\n#ifndef TEST_INPUT_H\n#define TEST_INPUT_H\n"
                "#include <stdint.h>\n")
        ctype = "uint8_t" if x_in.dtype == np.uint8 else "int8_t"
        f.write(c_array(ctype, "g_test_input", x_in).replace("const", "static const", 1))
        f.write(f"#define GOLDEN_TOP1 {int(np.argmax(out_g.ravel()))}\n#endif\n")


def dump_golden(layers, T, x_in, golden_dir, det):
    os.makedirs(golden_dir, exist_ok=True)
    x_in.tofile(os.path.join(golden_dir, "input.bin"))
    man = {"input": {"file": "input.bin", "shape": list(x_in.shape), "dtype": str(x_in.dtype)}, "ops": []}
    for L in layers:
        g = T(L["out"])
        fn = f"op{L['idx']:02d}_{L['op'].lower()}.bin"
        g.tofile(os.path.join(golden_dir, fn))
        man["ops"].append({"idx": L["idx"], "op": L["op"], "file": fn, "shape": list(g.shape),
                           "dtype": str(g.dtype), "fnv1a": f"0x{fnv1a(g):08x}",
                           "verified": L.get("verified"), "macs": L["macs"],
                           "tensor_name": det[L["out"]]["name"]})
    with open(os.path.join(golden_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/efficientnet-lite0/int8/model.tflite")
    ap.add_argument("--out-dir", default="src/model")
    ap.add_argument("--golden-dir", default="golden")
    ap.add_argument("--image", default=None, help="測試圖片 (需要 Pillow)；不給就用亂數")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    interp, det, layers, model_in, model_out, x_in, T, out_det = build(args)
    ok = True if args.no_verify else verify(layers, T)
    act, offs, arena = plan_memory(layers, model_in, model_out, det)
    w_bytes = sum(L["w"].size for L in layers if L["w"] is not None)
    print(f"weights: {w_bytes/1e6:.2f} MB   activation arena: {arena/1e6:.2f} MB")
    emit_c(layers, act, offs, arena, model_in, model_out, det, T, x_in, args.out_dir)
    dump_golden(layers, T, x_in, args.golden_dir, det)
    top = np.argsort(-T(model_out[0]).ravel().astype(np.int64))[:5]
    print(f"golden top-5 class: {top.tolist()}")
    print(f"輸出：{args.out_dir}/model_data.[ch], test_input.h；golden：{args.golden_dir}/")
    if not ok:
        print("\n有 op 沒有 bit-exact，請把上面的表貼回來。")
        sys.exit(1)


if __name__ == "__main__":
    main()
