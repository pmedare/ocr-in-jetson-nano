import torch
import time
import os
import tensorrt as trt
import cv2
from torch2trt import torch2trt, TRTModule
import easyocr
from easyocr.craft import CRAFT

IMAGE_PATH = '/mnt/DADES/home/pmedina/tfg/data/preprocessed_images/0.jpg'
TRT_MODULE_PATH = 'module_trt_fp16.pth'

# load image
image = cv2.imread(IMAGE_PATH)

# create EasyOCR Reader class
reader = easyocr.Reader(['en'], gpu=True)

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
    module_trt = torch2trt(module, [data], fp16_mode=True)
    torch.save(module_trt.state_dict(), TRT_MODULE_PATH)
    
print('Done.')