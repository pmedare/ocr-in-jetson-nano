import time
import json
import easyocr
import cv2

dataset_path = "/mnt/DADES2/STELA/data/"
dataset = "/mnt/DADES/home/pmedina/tfg/data/dataset.json"

with open(dataset, 'r') as file:
    data = json.load(file)

# Initialize model
reader = easyocr.Reader(['en'], gpu=True)

global_start = time.perf_counter()
with open('/mnt/DADES/home/pmedina/tfg/src/raw_runs/logs/easyOCR_server_GPU.txt', 'w') as file:
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
        results = reader.readtext(f'/mnt/DADES/home/pmedina/tfg/data/preprocessed_images/{id_}.jpg', detail=0)
        extracted_text = "".join(results)
        ocr_end = time.perf_counter()

        overall_end = time.perf_counter()

        file.write(f"{id_};{overall_end-overall_start};{image_crop_end-image_crop_start};{ocr_end-ocr_start};{label['code']};{extracted_text}\n")

        if i == 99:
                break

global_end = time.perf_counter()

global_time = global_end-global_start

print(f"Total execution lasted {round(global_time, 3)} seconds.")
print(f"Average seconds for image: {round(global_time/i, 3)*1000}ms.")