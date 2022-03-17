python export.py ^
 --weights weights/fc_net.pth ^
 --config configs/exam_ds/fc_net.yaml ^
 --format onnx ^
 --file weights/fc_net.onnx ^
 --device cpu ^
 --batch 1 ^
 --opset 12 ^