import torch
import time
import os
import tensorrt as trt
import cv2
from torch2trt import torch2trt, TRTModule
import easyocr
from easyocr.craft import CRAFT

IMAGE_PATH = '/home/jetson/tfg/data/cropped_images/0.jpg'
TRT_MODULE_PATH = 'module_trt_int8.pth'

# load image
image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (380, 260))
# create EasyOCR Reader class
reader = easyocr.Reader(['en'], gpu=True)


calib_data = []
for id in range(1000, 1050):
    img = cv2.imread('/'.join(IMAGE_PATH.split("/")[:-1]) + f"/{id}.jpg")
    img = cv2.resize(img, (380, 260))
    img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()
    calib_data.append(img_tensor)

# run inference with Reader class to get shape of input to internal PyTorch module
shape = None

def get_shape(module, inputs, outputs):
    global shape
    shape = inputs[0].shape

handle = reader.detector.module.register_forward_hook(get_shape)
reader.readtext(image)
handle.remove()

# create TensorRT optimized version of internal PyTorch module
module = reader.detector.module
module = module.cuda().eval()
data = torch.randn(shape).cuda()

print('Optimizing TensorRT...')
if os.path.exists(TRT_MODULE_PATH):
    module_trt = TRTModule()
    module_trt.load_state_dict(torch.load(TRT_MODULE_PATH))
else:
    module_trt = torch2trt(module, [data], int8_mode=True, int8_calib_dataset=calib_data)
    torch.save(module_trt.state_dict(), TRT_MODULE_PATH)

print('Done.')
