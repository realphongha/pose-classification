python export.py ^
 --weights E:\Learning\do_an_tot_nghiep\experiments\action\fc_net--exam--2022-03-01--07-42\best.pth ^
 --config configs/exam_ds/fc_net.yaml ^
 --format onnx ^
 --file E:\Learning\do_an_tot_nghiep\experiments\action\fc_net--exam--2022-03-01--07-42\fc_net.onnx ^
 --device cpu ^
 --batch 1 ^
 --opset 12
python -m onnxruntime.tools.convert_onnx_models_to_ort ^
 E:\Learning\do_an_tot_nghiep\experiments\action\fc_net--exam--2022-03-01--07-42\fc_net.onnx