---
library_name: onnx
license: apache-2.0
pipeline_tag: image-classification
tags:
  - onnx
  - onnxruntime
  - int8
  - uint8
---

# MediaPipe EfficientNet-Lite0 ImageNet 224 INT8/UINT8 ONNX

INT8/UINT8 ONNX version of MediaPipe EfficientNet-Lite0 ImageNet 224 for image classification.

- Quantization: Static 8-bit quantization with INT8 weights and UINT8 input/output boundaries, QDQ format


## Model Files

| File | Purpose | Format |
|---|---|---|
| `model.onnx` | Downloadable converted model | ONNX |
| `source/model.tflite` | Original model | TensorFlow Lite |
| `graphs/netron.png` | ONNX graph visualization | PNG |
| `mlir/onnx.mlir` | ONNX-MLIR import result | MLIR text |

## Parameter Summary

| Item | Value |
|---|---:|
| Precision | INT8/UINT8 |
| ONNX file size | 4.91 MiB |
| Initializer tensors | 290 |
| Stored initializer elements | 4,682,839 |
| External weight files | None |
| Original format | TensorFlow Lite |

`Stored initializer elements` includes weights, biases, quantization scales,
zero-points, and other constant tensors. It is not a trainable-parameter count.

## Original Model Inference

`pip install huggingface_hub numpy ai-edge-litert`

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from huggingface_hub import hf_hub_download

repo_id = "ketiswp/mediapipe-EfficientNet-Lite0-ImageNet-224-int8-uint8-onnx"
model_path = hf_hub_download(repo_id=repo_id, filename="source/model.tflite")
interpreter = Interpreter(model_path=model_path, num_threads=1)

for item in interpreter.get_input_details():
    signature = [int(value) for value in item.get("shape_signature", item["shape"])]
    shape = [value if value > 0 else 1 for value in signature]
    if shape != [int(value) for value in item["shape"]]:
        interpreter.resize_tensor_input(int(item["index"]), shape, strict=False)

interpreter.allocate_tensors()
for item in interpreter.get_input_details():
    value = np.zeros(tuple(int(dim) for dim in item["shape"]), dtype=item["dtype"])
    interpreter.set_tensor(int(item["index"]), value)

interpreter.invoke()
outputs = [interpreter.get_tensor(int(item["index"]))
           for item in interpreter.get_output_details()]
print([(value.shape, str(value.dtype)) for value in outputs])
```

## Converted ONNX Inference

`pip install huggingface_hub numpy onnxruntime`

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

repo_id = "ketiswp/mediapipe-EfficientNet-Lite0-ImageNet-224-int8-uint8-onnx"
model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    model_path,
    sess_options=options,
    providers=["CPUExecutionProvider"],
)

dtype_by_ort_type = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}
feeds = {}
for item in session.get_inputs():
    shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in item.shape]
    feeds[item.name] = np.zeros(shape, dtype=dtype_by_ort_type[item.type])

outputs = session.run(None, feeds)
print([(item.name, value.shape, str(value.dtype))
       for item, value in zip(session.get_outputs(), outputs)])
```

## Paired Model

[FP32 version](https://huggingface.co/ketiswp/mediapipe-EfficientNet-Lite0-ImageNet-224-fp32-onnx)

## Source

- [Original artifact](https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/int8/latest/efficientnet_lite0.tflite)
- [Source project](https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier)

## Project Validation

[FP32/quantized comparison, conversion results, and reproduction code](http://192.168.0.134:3000/GilhyeonLee/graph)

This repository also includes the Netron graph, ONNX Dialect MLIR, and static
MLIR dependency graph for this model variant.
