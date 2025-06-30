import time
import json
import easyocr
import cv2

ids = [i for i in range(0, 13+1)] + \
      [50] + [i for i in range(56,59+1)] + \
      [i for i in range(66,75+1)] + \
      [i for i in range(78, 110+1)] + [i for i in range(113, 151)]

dataset_path = "/mnt/DADES2/STELA/data/"
dataset = "/home/jetson/tfg/data/dataset.json"

with open(dataset, 'r') as file:
    data = json.load(file)

# Initialize model
reader = easyocr.Reader(['en'], gpu=False)

print("Starting warmup.")
image = cv2.imread("/home/jetson/tfg/data/preprocessed_images/1103.jpg")
for i in range(35):
    result = reader.readtext(image, detail=0)
print("Warmup has been completed!")

global_start = time.perf_counter()
with open('/home/jetson/tfg/raw_runs/logs/easyOCR_jetson_GPU_noLite_clocks.txt', 'w') as file:
    counter = 0
    # file.write('id;time_duration;ground_truth;prediction\n')
    for i, (id_, label) in enumerate(data.items()):
        if int(id_) in ids:
            counter += 1

            # Measure inference time
            overall_start = time.perf_counter()

            image_crop_start = time.perf_counter()
            image = cv2.imread(("/home/jetson/tfg/" + '/'.join(label['img_path'].split('/')[-4:])))
            cropped_image = image[150:375, 125:700]

            path = f'/home/jetson/tfg/data/cropped_images/{id_}.jpg'
            cv2.imwrite(path, cropped_image)
            image_crop_end = time.perf_counter()

            ocr_start = time.perf_counter()
            results = reader.readtext(f'/home/jetson/tfg/data/cropped_images/{id_}.jpg', detail=0)
            extracted_text = "".join(results)
            ocr_end = time.perf_counter()

            overall_end = time.perf_counter()

            file.write(f"{id_};{overall_end-overall_start};{image_crop_end-image_crop_start};{ocr_end-ocr_start};{label['code']};{extracted_text}\n")

            if counter == 100:
                    break

global_end = time.perf_counter()

global_time = global_end-global_start

print(f"Total execution lasted {round(global_time, 3)} seconds.")
print(f"Average seconds for image: {round(global_time/i, 3)*1000}ms.")
