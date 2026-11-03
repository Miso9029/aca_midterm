import sys
from collections import defaultdict
from ai_edge_litert.interpreter import Interpreter, OpResolverType

path = sys.argv[1]
it = Interpreter(
    model_path=path,
    experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
)
it.allocate_tensors()
T = {t["index"]: t for t in it.get_tensor_details()}
ops = it._get_ops_details()

def shape(i):
    return [int(x) for x in T[i]["shape"]]

def q(i):
    p = T[i]["quantization_parameters"]
    sc, zp = p["scales"], p["zero_points"]
    return len(sc), (float(sc[0]) if len(sc) else 0.0), (int(zp[0]) if len(zp) else 0)

print("model inputs :", [(d["dtype"].__name__, [int(x) for x in d["shape"]], d["quantization"]) for d in it.get_input_details()])
print("model outputs:", [(d["dtype"].__name__, [int(x) for x in d["shape"]], d["quantization"]) for d in it.get_output_details()])
print()

summary = defaultdict(lambda: [0, 0])
for op in ops:
    name = op["op_name"]
    if name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "ADD"):
        continue
    ins, outs = op["inputs"], op["outputs"]
    xi, oi = ins[0], outs[0]
    macs, kind, kern, wch = 0, name, "", "-"
    if name == "CONV_2D":
        O, kh, kw, I = shape(ins[1])
        _, H, W, _ = shape(oi)
        macs = H * W * O * kh * kw * I
        kern = f"{kh}x{kw} I{I} O{O}"
        kind = "stem conv" if I == 3 else ("conv 1x1" if kh == 1 else f"conv {kh}x{kw}")
        wch = q(ins[1])[0]
    elif name == "DEPTHWISE_CONV_2D":
        _, kh, kw, C = shape(ins[1])
        _, H, W, _ = shape(oi)
        macs = H * W * C * kh * kw
        kern = f"{kh}x{kw} C{C}"
        kind = f"depthwise {kh}x{kw}"
        wch = q(ins[1])[0]
    elif name == "FULLY_CONNECTED":
        O, I = shape(ins[1])
        macs = O * I
        kern = f"I{I} O{O}"
        kind = "fully connected"
        wch = q(ins[1])[0]
    else:
        kind = "add"
    summary[kind][0] += 1
    summary[kind][1] += macs
    n, s_in, z_in = q(xi)
    n, s_out, z_out = q(oi)
    print(f"{op['index']:3d} {kind:16s} {kern:16s} in={shape(xi)} out={shape(oi)} "
          f"MACs={macs/1e6:7.2f}M wch={wch} in(s={s_in:.4g},zp={z_in}) out(s={s_out:.4g},zp={z_out})")

total = sum(v[1] for v in summary.values())
print(f"\nTOTAL MACs: {total/1e6:.1f}M")
for kind, (cnt, m) in sorted(summary.items(), key=lambda kv: -kv[1][1]):
    pct = 100 * m / total if total else 0
    print(f"{kind:16s} count={cnt:3d}  MACs={m/1e6:8.1f}M  {pct:5.1f}%")
