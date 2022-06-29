python export.py \
 --weights path\to\model.pth \
 --config configs/exam_ds/fc_net.yaml \
 --format onnx \
 --file path\to\model.onnx \
 --device cpu \
 --batch 1 \
 --opset 12
python -m onnxruntime.tools.convert_onnx_models_to_ort \
 path\to\model.onnx \
 --optimization_level disable