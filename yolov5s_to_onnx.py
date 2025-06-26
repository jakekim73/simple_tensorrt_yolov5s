import torch
import onnx

model_onnx_name='yolov5s_byCode.onnx'
dummy_input = torch.randn(1, 3, 640, 640)

# YOLOv5s 모델 로드
model = torch.load('yolov5s.pt', map_location='cpu')['model'].float()
print("Model Number of Classes : {}".format(model.nc))
print("Model Class names : {}". format(model.names))
model.eval()

# ONNX로 내보내기
torch.onnx.export(model, 
                    dummy_input, 
                    model_onnx_name, 
                    verbose=False, 
                    opset_version=13,   # 13: TensorRT >= 8, 12: TensorRT <= 8
                    training=torch.onnx.TrainingMode.EVAL,
                    do_constant_folding=True,
                    input_names=['images'],
                    output_names=['output'],
                    #dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                    #                'output': {0: 'batch', 1: 'anchors'}}  # shape(1,25200,85)
                    dynamic_axes=None
)

# Checks
model_onnx = onnx.load(model_onnx_name)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

print('YOLOv5s 모델이 yolov5s_byCode.onnx로 내보내졌습니다.')


