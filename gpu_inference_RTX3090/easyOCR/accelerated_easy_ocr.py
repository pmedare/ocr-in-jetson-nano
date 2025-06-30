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

TRT_MODULE_PATH = 'module_trt_fp16.pth'

dataset_path = "/mnt/DADES2/STELA/data/"
dataset = "/mnt/DADES/home/pmedina/tfg/data/dataset.json"

with open(dataset, 'r') as file:
    data = json.load(file)

# Initialize model
reader = easyocr.Reader(['es'], gpu=True)

if os.path.exists(TRT_MODULE_PATH):
    module_trt = TRTModule()
    module_trt.load_state_dict(torch.load(TRT_MODULE_PATH))
    print("Using TensorRT acceleration.")

reader.detector = module_trt

global_start = time.perf_counter()
with open('/mnt/DADES/home/pmedina/tfg/src/raw_runs/logs/easyOCR_server_GPU_tensorRT_detector.txt', 'w') as file:
    # file.write('id;time_duration;ground_truth;prediction\n')
    for i, (id_, label) in enumerate(data.items()):
        # Measure inference time
        overall_start = time.perf_counter()

        image_crop_start = time.perf_counter()
        image = cv2.imread(label['img_path'])
        cropped_image = image[150:375, 125:700]

        path = f'/mnt/DADES/home/pmedina/tfg/data/cropped_images/{id_}.jpg'
        cv2.imwrite(path, cropped_image)
        image_crop_end = time.perf_counter()

        ocr_start = time.perf_counter()
        # torch.cuda.current_stream().synchronize()
        results = reader.readtext(f'/mnt/DADES/home/pmedina/tfg/data/preprocessed_images/{id_}.jpg', detail=0)
        # torch.cuda.current_stream().synchronize()
        extracted_text = "".join(results)
        ocr_end = time.perf_counter()

        overall_end = time.perf_counter()

        file.write(f"{id_};{overall_end-overall_start};{image_crop_end-image_crop_start};{ocr_end-ocr_start};{label['code']};{extracted_text}\n")
    
        if i == 99:
            break

global_end = time.perf_counter()

global_time = global_end-global_start

print(f"Total execution lasted {round(global_time, 3)} seconds.")
print(f"Average seconds for image: {round(global_time/(i + 1), 3)*1000}ms.")