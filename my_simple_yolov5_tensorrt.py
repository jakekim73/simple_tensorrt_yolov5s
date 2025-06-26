import cv2
import time
import numpy as np
import torch
import tensorrt as trt
import random

from collections import OrderedDict, namedtuple
from utils.general import non_max_suppression, scale_coords, xywh2xyxy


# 설정값
CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_SIZE = 640
AGNOSTIC_NMS = False

# 1. TensorRT 모델 엔진
ENGINE_PATH = "yolov5s.engine"


# 2. COCO Data-Set Class name, 80개
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard','tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# 3. Class name별 Bound Box 색 설정
CLASS_COLORS = {
    "person": (255, 0, 0), "bicycle": (255, 128, 0), "car": (255, 255, 0), "motorbike": (128, 255, 0), "aeroplane": (0, 255, 0),
    "bus": (0, 255, 128), "train": (0, 255, 255), "truck": (0, 128, 255), "boat": (0, 0, 255), "traffic light": (128, 0, 255),
    "fire hydrant": (255, 0, 255), "stop sign": (255, 0, 128), "parking meter": (128, 0, 0), "bench": (128, 64, 0), "bird": (128, 128, 0),
    "cat": (64, 128, 0), "dog": (0, 128, 0), "horse": (0, 128, 64), "sheep": (0, 128, 128), "cow": (0, 64, 128),
    "elephant": (0, 0, 128), "bear": (64, 0, 128), "zebra": (128, 0, 128), "giraffe": (128, 0, 64), "backpack": (255, 128, 128),
    "umbrella": (255, 200, 128), "handbag": (255, 255, 128), "tie": (200, 255, 128), "suitcase": (128, 255, 128), "frisbee": (128, 255, 200),
    "skis": (128, 255, 255), "snowboard": (128, 200, 255), "sports ball": (128, 128, 255), "kite": (200, 128, 255), "baseball bat": (255, 128, 255),
    "baseball glove": (255, 128, 200), "skateboard": (255, 128, 128), "surfboard": (200, 128, 128), "tennis racket": (128, 128, 128), "bottle": (64, 64, 64),
    "wine glass": (192, 192, 192), "cup": (255, 192, 192), "fork": (255, 224, 192), "knife": (255, 255, 192), "spoon": (224, 255, 192),
    "bowl": (192, 255, 192), "banana": (192, 255, 224), "apple": (192, 255, 255), "sandwich": (192, 224, 255), "orange": (192, 192, 255),
    "broccoli": (224, 192, 255), "carrot": (255, 192, 255), "hot dog": (255, 192, 224), "pizza": (255, 192, 192), "donut": (224, 192, 192),
    "cake": (192, 192, 192), "chair": (128, 128, 64), "sofa": (64, 128, 64), "pottedplant": (64, 128, 128), "bed": (64, 64, 128),
    "diningtable": (128, 64, 128), "toilet": (192, 0, 0), "tvmonitor": (192, 64, 0), "laptop": (192, 128, 0), "mouse": (192, 192, 0),
    "remote": (128, 192, 0), "keyboard": (64, 192, 0), "cell phone": (0, 192, 0), "microwave": (0, 192, 64), "oven": (0, 192, 128),
    "toaster": (0, 192, 192), "sink": (0, 128, 192), "refrigerator": (0, 64, 192), "book": (0, 0, 192), "clock": (64, 0, 192),
    "vase": (128, 0, 192), "scissors": (192, 0, 192), "teddy bear": (192, 0, 128), "hair drier": (192, 0, 64), "toothbrush": (64, 0, 0),
    "default": (160, 160, 160)  
}

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# 4. TensorRT 엔진 binding을 위한 Binding tuple 타입 생성
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))


# 5. TensorRT 엔진 로드 함수
def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


# 6. 이미지를 모델 입력 크기(640x640)에 맞추고 종횡비(aspect ratio) 유지
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]       # 현재 이미지의 (height, width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


# 7. 이미지 전처리 함수
def preprocess_image(input_image, input_shape):
    image0 = input_image
    img_letterboxed, r, (dw, dh) = letterbox(image0, (input_shape[3], input_shape[2]))
    image = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float16) / 255.0
    #image = np.ascontiguousarray(image, dtype=np.float) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).cuda()
    return image0, image


# 8. Bound Box draw 함수
def draw_boxes(img, preds, class_names):
    for *xyxy, conf, cls in preds:
        class_id = class_names[int(cls)]
        label = f'{class_id} {conf:.2f}'
        color = CLASS_COLORS.get(class_id, CLASS_COLORS["default"])
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + 100, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# 9. Main 함수 
def main():
# 10. TensorRT 엔진 로드  
    engine = load_engine(ENGINE_PATH)
# 11. TensorRT context 생성 
    context = engine.create_execution_context()

# 12. TensorRT 엔진의 데이터 입력과 출력을 관리하기 위해 Binding 정보 수집 및 초기화
    bindings = OrderedDict()
    for i in range(engine.num_bindings):  # TensorRT 엔진이 가진 모든 바인딩 수만큼 반복
        name = engine.get_binding_name(i) # 바인딩의 이름을 가져옴 (예: "images", "output" 등).
        dtype = trt.nptype(engine.get_binding_dtype(i))  # TensorRT 내부의 데이터 타입을 NumPy 타입으로 변환
        shape = tuple(engine.get_binding_shape(i))       # 바인딩의 텐서 shape을 튜플로 변환. 예: (1, 3, 640, 640) 또는 (1, 25200, 85)
        torch_dtype = torch.float32 if dtype == np.float32 else torch.float16
        data = torch.empty(size=shape, dtype=torch_dtype, device='cuda') # 해당 shape의 **빈 GPU 메모리 공간(Tensor)**을 생성
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr())) # Binding namedtuple에 모든 정보를 담아, 이름을 key로 하여 bindings 딕셔너리에 저장

# 13. 각 바인딩의 GPU 메모리 주소(pointer)만 추출해서 'binding_addrs' OrderedDict으로 정리
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    input_shape = bindings['images'].shape
    output_shape = bindings['output'].shape

    # Binding address 정보 출력 및 모델 입력/출력 shape 정보 출력 
    print("binding_addrs :", binding_addrs)
    print("model input_shape :", input_shape)
    print("model output_shape :", output_shape)

# 14. 카메라 설정 (640x480 해상도로 설정)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
# 15. 카메라 영상 이미지 read
        ret, frame = cap.read()
# 16. 카메라 이미지 데이터 전처리 
        img0, image = preprocess_image(frame, input_shape)        
        image = image.half()
        print("image.shape :", image.shape)
        assert image.shape[1:] == input_shape[1:]   
        start = time.time()
# 17. 전처리된 이미지 데이터를 TensorRT 입력(bindings['images'])에 연결         
        bindings['images'].data.copy_(image[0])
# 18. 추론          
        context.execute_v2(list(binding_addrs.values()))
# 19. 추론 결과(binding['output'])데이터를 cpu 메모리로 옮김                   
        output = bindings['output'].data
        output = output[0].cpu()       
# 20. 추론 결과에 대해서 NMS 실행 
        pred = non_max_suppression(output.unsqueeze(0), CONF_THRES, IOU_THRES)[0]
        print('output shape :', output.shape)
        print('pred shape :', pred.shape)
        #print('pred :', pred)
        end = time.time()
# 21. 추론 시간 계산 (FPS)
        fps = 1.0 / (end - start)
        print("fps :",fps)

# 22. 탐지 객체들의 Bounding Box 좌표값을 실제 영상 이미지 크기에 맞춰 조정(Rescale)  
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(image.shape[2:], pred[:, :4], img0.shape).round()
            draw_boxes(img0, pred, CLASSES)

        cv2.putText(img0, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
# 23. 최종 객체 탐지 영상 이미지 출력  
        cv2.imshow("YOLOv5s TensorRT", img0)

# 24. ‘q’키가 눌려지면 while문 종료 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 25. Camera 캡쳐 종료 및 OpenCV window 해제     
    cap.release()
    cv2.destroyAllWindows()


# 25. 프로그램 시작 
if __name__ == '__main__':
    main()
