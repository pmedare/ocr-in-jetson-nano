import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('ppocr').setLevel(logging.ERROR)

import os
os.environ["FLAGS_log_level"] = "3"
os.environ["GLOG_minloglevel"] = "3"

# import gc
import time
import json
import cv2
from utils import preprocess_image
from paddleocr import PaddleOCR
import paddle

print(paddle.device.get_device())

ids = [i for i in range(0, 13+1)] + \
      [50] + [i for i in range(56,59+1)] + \
      [i for i in range(66,75+1)] + \
      [i for i in range(78, 110+1)] + [i for i in range(113, 151)]

# Initialize model with GPU support
ocr = PaddleOCR(lang='en', ocr_version='PP-OCRv3' , use_gpu=True, use_tensorrt=True, use_angle_cls=False, 
                rec_batch_num=1, cpu_threads=4, layout=False, table=False, use_visual_backbone=False, 
                enable_mkldnn=False, gpu_mem=1000, precision="fp16", det_db_unclip_ratio=0.5, warmup=True,
                det_limit_side_len=380, use_calib_mode=True)

dataset_path = "/home/jetson/tfg/data/"
dataset = "/home/jetson/tfg/data/dataset.json"

with open(dataset, 'r') as file:
    data = json.load(file)

"""
print("Preloading data...")
cropped_images = {}
path = "/home/jetson/tfg/data/cropped_images/"
for id in ids:
    image = cv2.resize(cv2.imread(path + str(id) + ".jpg"), (380, 260))
    cropped_images[id] = image.copy()
"""

print("Starting warmup.")
image = cv2.resize(cv2.imread("/home/jetson/tfg/data/cropped_images/1103.jpg"), (380, 260))
for i in range(35):
    result = ocr.ocr(image, cls=False, det=True)
    print(result)
print("Warmup has been completed!")

global_start = time.perf_counter()
with open('/home/jetson/tfg/raw_runs/logs/paddleOCR_jetson_GPU_trt_fp16.txt', 'w') as file:
    counter=0
    for i, (id_, label) in enumerate(data.items()):
        if int(id_) in ids:
            counter += 1

            overall_start = time.perf_counter()
            image_crop_start = time.perf_counter()

            image = cv2.imread("/home/jetson/tfg/" + "/".join(label['img_path'].split("/")[-4:]))
            cropped_image = image[150:375, 125:700]

            image_crop_end = time.perf_counter()

            ocr_start = time.perf_counter()
            results = ocr.ocr(cropped_image, cls=False, det=True)
            if len(results) > 0:
                extracted_text = "".join([result[1][0] for result in results[0]])
            else:
                extracted_text = ""

            ocr_end = time.perf_counter()
            overall_end = time.perf_counter()

            file.write(f"{id_};{overall_end-overall_start};{image_crop_end-image_crop_start};{ocr_end-ocr_start};{label['code']};{extracted_text}\n")

            if counter == 100:
                break

global_end = time.perf_counter()

global_time = global_end-global_start

print(f"Total execution lasted {round(global_time, 3)} seconds.")
print(f"Average seconds for image: {round(global_time/(i+1)*1000, 3)}ms.")
