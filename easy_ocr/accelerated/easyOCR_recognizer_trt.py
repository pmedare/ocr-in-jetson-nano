import torch
import time
import os
import tensorrt as trt
import cv2
from torch2trt import torch2trt, TRTModule
import easyocr
from easyocr.craft import CRAFT

IMAGE_PATH = '/home/jetson/tfg/data/cropped_images/0.jpg'
TRT_MODULE_PATH = 'module_recognizer_trt.pth'

# load image
# image = cv2.imread(IMAGE_PATH)
# image = cv2.resize(image, (380, 260))
# create EasyOCR Reader class

y = torch.ones((1, 3, 480, 640), dtype=torch.float).cuda()
reader = easyocr.Reader(['en'], gpu=True)

# calib_data = []
# for id in range(1000, 1050):
    # img = cv2.imread('/'.join(IMAGE_PATH.split("/")[:-1]) + f"/{id}.jpg")
    # img = cv2.resize(img, (380, 260))
    # img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()
    # calib_data.append(img_tensor)

# run inference with Reader class to get shape of input to internal PyTorch module
#shape = None

#def get_shape(module, inputs, outputs):
    #global shape
    #shape = inputs[0].shape

#handle = reader.detector.module.register_forward_hook(get_shape)
#reader.readtext(image)
#handle.remove()

# create TensorRT optimized version of internal PyTorch module
# module = reader.detector.module
# module = module.cuda().eval()
# data = torch.randn(shape).cuda()


x = torch.ones((1,1,64,320),dtype=torch.float).to('cuda')

print('Optimizing TensorRT...')
#if os.path.exists(TRT_MODULE_PATH):
 #    module_trt = TRTModule()
#     module_trt.load_state_dict(torch.load(TRT_MODULE_PATH))
# else:
   # module_trt = torch2trt(reader.recognizer, [x], use_onnx=True)
  #  torch.save(module_trt.state_dict(), TRT_MODULE_PATH)
    
#recognizer
print("\nRecognizer:")
x = torch.ones((1,1,64,320),dtype=torch.float).to('cuda')
reader.recognizer.eval()
print("Before Conversion:")
# profile(reader.recognizer, x) #throughput: 36.912    latency: 24.610

if os.path.isfile('torch2trt_models/easyocr_recognize.pth'):
    model_trt_rec = TRTModule()
    model_trt_rec.load_state_dict(torch.load('torch2trt_models/easyocr_recognize.pth'))
else:
    model_trt_rec = torch2trt(reader.detector, [y])
    torch.save(model_trt_rec.state_dict(),'torch2trt_models/easyocr_recognize.pth')
model_trt_rec = torch2trt(reader.recognizer, [x])#, use_onnx=True)

print("After Conversion")
# profile(model_trt_rec,x) #throughput: 2296.110       latency: 0.450
torch.save(model_trt_rec.state_dict(),'torch2trt_models/easyocr_recognize.pth')
print('Done.')
