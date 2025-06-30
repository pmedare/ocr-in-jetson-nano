import torch
from torchvision.models.alexnet import alexnet
from torchvision.models import squeezenet1_0, squeezenet1_1
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import densenet121, densenet169, densenet201, densenet161
from torch2trt import torch2trt
import time

num_inferences = 50
num_warmup = 20

model_fns = [
    ("alexnet", alexnet),
    ("squeezenet1_0", squeezenet1_0),
    ("squeezenet1_1", squeezenet1_1),
    ("resnet18", resnet18),
    ("resnet34", resnet34),
    ("resnet50", resnet50),
    ("resnet101", resnet101),
    ("resnet152", resnet152),
    ("densenet121", densenet121),
    ("densenet169", densenet169),
    ("densenet201", densenet201),
    ("densenet161", densenet161),
]

x = torch.ones((1, 3, 224, 224)).cuda()

for name, fn in model_fns:
    print("\n" + "="*40)
    print(f"Benchmarking {name}")
    model = fn(pretrained=True).eval().cuda()
    # Convert to TensorRT
    print("Converting to TensorRT")
    model_trt = torch2trt(model, [x], fp16_mode=True)
    print("Benchmarking baseline")
    # Benchmark PyTorch
    with torch.no_grad():
        print("Warming up...")
        for _ in range(num_warmup):  # Warm up
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        print("Starting baseline inference...")
        for _ in range(num_inferences):
            model(x)
        torch.cuda.synchronize()
        t1 = time.time()
    print('PyTorch time:', t1 - t0)
    print('PyTorch FPS:', num_inferences/(t1-t0))

    # Benchmark TensorRT
    print("Benchmarking TensorRT")
    print("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):  # Warm up
            model_trt(x)
        torch.cuda.synchronize()
        t0 = time.time()
        print("Starting TensorRT inference.")
        for _ in range(num_inferences):
            model_trt(x)
        torch.cuda.synchronize()
        t1 = time.time()
    print('TensorRT time:', t1 - t0)
    print('TensorRT FPS:', num_inferences/(t1-t0))
    # Free up memory
    del model, model_trt
    torch.cuda.empty_cache()