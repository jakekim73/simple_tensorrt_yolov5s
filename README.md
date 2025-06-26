# Simple object detection app using TensorRT engine converted from the yolov5s model.  

The **'my_simple_yolov5_tensorrt.py'** is python object detection application using tensorrt engine.

It needs tensorrt engine converted from **'yolov5.onnx'** by using 'trtexec'. (like below, tested jetson nano)  

**/usr/src/tensorrt/bin/trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine --verbose --workspace=4096 --fp16**



# Regarding exporting onnx model from yolove5s.pt 


To exporting onnx model from 'yolov5s.pt' is very simple. 
Run **'yolov5s_to_onnx.py'** that loads 'yolov5s.pt' file and exporting its onnx model. 




