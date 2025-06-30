# :rocket: Run PaddleOCR inference and acceleration!

> :information_source: In this folder you may find the needed scripts to compute inference with PaddleOCR package. You have two different folders available: 
- **Baseline:** contains all scripts for CPU and GPU execution without any kind of improvement.
- **Accelerated:** contains all scripts for TensorRT acceleration with tensor2trt. The TensorRT engine is built before inference, and inference is done in the same script. Notice that with PaddleOCR a single flag activation `use_trt=True` is enough to create the TensorRT engine.

*INT8 is not implemented since the process was always killed for lack of free SWAP memory. Implementing this optimization, although not highly supported in Jetson Nano could be tested.*