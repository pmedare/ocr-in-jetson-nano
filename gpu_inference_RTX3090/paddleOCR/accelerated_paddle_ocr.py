import time
import json
import cv2
from utils import preprocess_image
from paddleocr import PaddleOCR

# Initialize model with GPU support
ocr = PaddleOCR(lang='en', warmup=True, use_gpu=True, use_tensorrt=True, use_angle_cls=False)

dataset_path = "/mnt/DADES2/STELA/data/"
dataset = "/mnt/DADES/home/pmedina/tfg/data/dataset.json"

with open(dataset, 'r') as file:
    data = json.load(file)

for i in range(100):
    result = ocr.ocr("/mnt/DADES/home/pmedina/tfg/data/preprocessed_images/445.jpg", cls=False, det=False)

print("\n\nstarting now")

global_start = time.perf_counter()
with open('/mnt/DADES/home/pmedina/tfg/src/raw_runs/logs/paddleOCR_server_GPU_tensorRT.txt', 'w') as file:
    for i, (id_, label) in enumerate(data.items()):
        # Measure inference time
        overall_start = time.perf_counter()

        image_crop_start = time.perf_counter()
        image = cv2.imread(label['img_path'])
        cropped_image = image[150:375, 125:700]

        path = f'/mnt/DADES/home/pmedina/tfg/data/cropped_images/{id_}.jpg'
        cv2.imwrite(path, cropped_image)

        threshold_image = preprocess_image(path)
        cv2.imwrite(path, cropped_image)
        image_crop_end = time.perf_counter()

        ocr_start = time.perf_counter()
        results = ocr.ocr(f'/mnt/DADES/home/pmedina/tfg/data/preprocessed_images/{id_}.jpg', cls=False)

        if len(results) > 0:
            extracted_text = "".join([result[1][0] for result in results[0]])
        ocr_end = time.perf_counter()

        overall_end = time.perf_counter()

        file.write(f"{id_};{overall_end-overall_start};{image_crop_end-image_crop_start};{ocr_end-ocr_start};{label['code']};{extracted_text}\n")

        if i == 99:
            break

global_end = time.perf_counter()

global_time = global_end-global_start

print(f"Total execution lasted {round(global_time, 3)} seconds.")
print(f"Average seconds for image: {round(global_time/(i+1)*1000, 3)}ms.")