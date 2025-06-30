# :rocket: Run EasyOCR inference and acceleration!

> :information_source: In this folder you may find the needed scripts to compute inference with EasyOCR package. You have two different folders available: 
- **Baseline:** contains all scripts for CPU and GPU execution without any kind of improvement.
- **Accelerated:** contains all scripts for TensorRT acceleration with tensor2trt. First the TensorRT engine must be built, then inference is done and results are logged. Notice that there are both FP32 and FP16 options. 

*INT8 is not implemented since the process was always killed for lack of free SWAP memory. Implementing this optimization, although not highly supported in Jetson Nano could be tested.*