# Follow-up documentation and report
> :information_source: In this folder you will find all follow-up reports delivered over the TFG duration as well as the final TFG report.

## Final workplan distribution
The final workplan distribution is shown in the table below.

| **Phase/Task** | **Weeks** |
|---|---|
| **Phase 1: Setup & Preparation** | **(Weeks 1-3)** |
| - Server environment setup (CUDA, cuDNN, PyTorch, OpenCV...) | Week 1 |
| - Dataset preparation and augmentation (random rotations, scaling, noise addition) | Week 2 |
| - Implementation of image preprocessing pipeline (Gaussian blur, adaptive thresholding...) | Weeks 3 |
| **Phase 2: Model Selection & Baseline Development** | **(Weeks 5-8)** |
| - Research and selection of OCR models (EasyOCR, PaddleOCR, Transformers...) | Week 5 |
| - Implementation of baseline models on the server | Weeks 6-7 |
| - Evaluation of baseline models (character recognition accuracy, inference time) | Week 8 |
| **Phase 3: Jetson Nano Deployment & Adaptation** | **(Weeks 9-15)** |
| - Jetson Nano environment setup (JetPack SDK, TensorRT installation) | Week 9 |
| - Containers creation for Jetson Nano | Week 10 |
| - Model adaptation for Jetson Nano | Weeks 11-13 |
| - Initial testing and debugging on Jetson Nano | Week 14-15 |
| **Phase 4: Model Optimization** | **(Weeks 15-17)** |
| - Model conversion to TensorRT format | Week 15 |
| - Testing of quantization and pruning | Week 15 |
| - TensorRT models' adaptation on Jetson Nano | Weeks 16-17 |
| - Performance metrics extraction on server and Jetson Nano | Week 17 |
| **Phase 5: Performance Evaluation & Finalization** | **(Weeks 18-20)** |
| - Comprehensive performance evaluation on Jetson Nano (accuracy, inference time, model size) | Weeks 18-19 |
| - Comparison of Jetson Nano performance w.r.t. server baseline | Week 20 |
| - Writing final report and documentation | Week 20 |