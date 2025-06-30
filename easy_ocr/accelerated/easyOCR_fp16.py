import time
import json
import torch
import easyocr
import numpy as np
import cv2
import torch
import time
import os
import tensorrt as trt
import cv2
from torch2trt import TRTModule
import easyocr
from easyocr.craft import CRAFT

ids = [i for i in range(0, 13+1)] + \
      [50] + [i for i in range(56,59+1)] + \
      [i for i in range(66,75+1)] + \
      [i for i in range(78, 110+1)] + [i for i in range(113, 151)]

TRT_MODULE_PATH = 'module_trt_fp16.pth'

dataset_path = "/home/jetson/tfg/data/"
dataset = "/home/jetson/tfg/data/dataset.json"

with open(dataset, 'r') as file:
    data = json.load(file)

# Initialize model
reader = easyocr.Reader(['es'], gpu=True, quantize=True)

if os.path.exists(TRT_MODULE_PATH):
    module_trt = TRTModule()
    module_trt.load_state_dict(torch.load(TRT_MODULE_PATH))
    print("Using TensorRT acceleration.")

reader.detector = module_trt

cropped_image=cv2.imread("/home/jetson/tfg/data/cropped_images/0.jpg")
cropped_image = cv2.resize(cropped_image, (380, 260))
for i in range(35):
    results = reader.readtext(cropped_image)
    print(results)

global_start = time.perf_counter()
with open('/home/jetson/tfg/raw_runs/logs/easyOCR_jetson_GPU_tensorRT_detector_fp16.txt', 'w') as file:
    # file.write('id;time_duration;ground_truth;prediction\n')
    counter = 0
    for i, (id_, label) in enumerate(data.items()):
        # print(type(id_), type(ids[0]), id_, ids[0])
        if int(id_) in ids:
            counter += 1
            # Measure inference time
            overall_start = time.perf_counter()

            image_crop_start = time.perf_counter()
            image = cv2.imread("/home/jetson/tfg/" + "/".join(label['img_path'].split("/")[-4:]))
            # cropped_image = image[150:375, 125:700]

            path = f'/home/jetson/tfg/data/cropped_images/{id_}.jpg'
            cropped_image = cv2.resize(cv2.imread(path), (380, 260))
            # cv2.imwrite(path, cropped_image)
            image_crop_end = time.perf_counter()

            ocr_start = time.perf_counter()
            torch.cuda.current_stream().synchronize()
            results = reader.readtext(cropped_image) # f'/mnt/DADES/home/pmedina/tfg/data/preprocessed_images/{id_}.jpg', detail=0)
            # print(results)
            torch.cuda.current_stream().synchronize()
            extracted_text = ''.join([text for _, text, _ in results])
            ocr_end = time.perf_counter()

            overall_end = time.perf_counter()

            file.write(f"{id_};{overall_end-overall_start};{image_crop_end-image_crop_start};{ocr_end-ocr_start};{label['code']};{extracted_text}\n")

            if counter == 100:
                break

global_end = time.perf_counter()

global_time = global_end-global_start

print(f"Total execution lasted {round(global_time, 3)} seconds.")
print(f"Average seconds for image: {round(global_time/(i + 1), 3)*1000}ms.")
