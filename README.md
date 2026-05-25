# FPGA-Edge-IIoT-Intrusion-Detection
A hierarchical Intrusion Detection System for IIoT edge devices on the PYNQ-Z2 FPGA. Features a cascading pipeline (Linear Regression, Conv-AE, Spatial Attention CNN) compiled via Xilinx FINN for high-efficiency threat detection. 

This project presents a multi-stage, conditionally executed Intrusion Detection System (IDS) optimized for Industrial IoT (IIoT) security. Deployed on the Xilinx PYNQ-Z2 FPGA architecture, the design addresses the severe computational bottleneck of running continuous deep learning models at the edge. Rather than relying on a single monolithic classifier, the system implements a dynamically gated hierarchical pipeline connected via AXI-Stream interfaces. Early-stage, lightweight hardware filters process continuous network traffic, isolating benign data and invoking a computationally intensive Convolutional Neural Network (CNN) exclusively for highly suspicious samples.

Hierarchical Inference Pipeline
The dataflow architecture operates across three cascaded stages to balance high throughput with accurate threat classification:
<br>
Stage 1: Custom Verilog Linear Regression (LR) Filter: The primary gate is a custom-designed Verilog IP utilizing 61 parallel DSP multipliers to evaluate incoming feature vectors at a rate of one sample per clock cycle. Operating with hardwired Q16.16 fixed-point weights, this stage acts as an ultra-low-latency anomaly scorer, successfully filtering 85.3% of all network traffic with a hardware latency of just ~0.20 ms.
<br>
Stage 2: Quantized Convolutional Autoencoder (Conv-AE): Traffic that breaches the Stage 1 threshold is passed to an 8-bit quantized Conv-AE. Trained to reconstruct normal IIoT traffic profiles, this symmetric encoder-decoder module computes a Mean Squared Error (MSE) anomaly score, intercepting an additional 13.0% of ambiguous data.
<br>
Stage 3: Spatial Attention CNN: Samples exceeding the Conv-AE threshold trigger the final classification stage. A two-layer CNN augmented with a Spatial Attention module generates a 15-class probability distribution to identify 14 specific attack types (e.g., DDoS, MITM). Spatial Attention was strategically selected for its ability to produce channel-wise scalar weights, translating highly discriminative spatial mapping into a single, FPGA-efficient multiply-accumulate dataflow stage.
<br>
# Hardware Implementation & FINN Deployment
To map these complex neural networks to the Zynq-7000 SoC fabric, the models undergo Quantization-Aware Training (QAT) via the Brevitas PyTorch library, compressing weights and activations to 8-bit unsigned integers. The quantized models are then exported to ONNX format and compiled into highly unrolled dataflow IPs using the Xilinx FINN framework. Due to the extreme logic density of the FINN-generated modules, the components are time-multiplexed in software on the PYNQ-Z2, leveraging the cascade gating to maintain high average throughput without exhausting physical LUT resources.
<br>
# Performance & Optimization Metrics
Tested against the comprehensive 61-feature Edge-IIoTset benchmark, this cascaded methodology effectively mitigates the performance-efficiency trade-off. The system achieves a pipeline-level binary detection accuracy of ~90% with a precision of 92.03%. By filtering the vast majority of benign inputs at the hardware level, the computationally expensive CNN is triggered on less than 2.0% of total network traffic. This dynamic execution reduces the weighted average pipeline latency to just ~0.254 ms, demonstrating a highly optimized architecture for real-time edge IIoT deployments.
