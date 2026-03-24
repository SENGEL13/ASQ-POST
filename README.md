# 🧠 ASQ & POST: A Synergistic Framework for Adaptive and Non-Uniform Quantization

Official PyTorch implementation of  
**"ASQ & POST: A Synergistic Framework for Adaptive and Non-Uniform Quantization"**  
submitted to Neurocomputing (2025)*.  

---

## 🚀 Overview

Quantization-Aware Training (QAT) often suffers from a rigidity problem—quantization parameters optimized on the training set fail to adapt to dynamic activation distributions during inference.  
This repository introduces two complementary components designed to overcome this limitation:

### 🔹 Adaptive Step-Size Quantization (**ASQ**)
A dynamic quantization mechanism that adjusts activation step-sizes via a lightweight two-layer adapter:

$$
s_a = s \times \beta
$$

where $\beta$ is computed from activation statistics.  
ASQ enhances generalization across diverse and unseen distributions while adding negligible computational overhead.

### 🔹 Power-of-Square-Root-of-Two (**POST**)
A non-uniform, hardware-friendly quantization scheme replacing the rigid Power-of-Two (POT) grid with a √2-based exponential grid.  
Implemented using a compact LUT (Look-Up Table), POST achieves higher representational fidelity with minimal storage (<0.002 % of model parameters).

---

## 🧩 Features

- ✅ Dynamic quantization for activations (ASQ)  
- ✅ √2-based non-uniform quantization for weights (POST)  
- ✅ Plug-and-play PyTorch modules for CNN and Transformer layers  
- ✅ Full training and evaluation pipelines for ImageNet and CIFAR-10  
- ✅ Reproducible results for ResNet-18/34 and MobileNet-V2  

---

## 🛠️ Installation

```bash
git clone https://github.com/SENGEL13/ASQ-POST.git
cd ASQ-POST
conda create -n asqpost python=3.10
conda activate asqpost
pip install -r requirements.txt
```

---

## 你好 👋

欢迎使用 ASQ & POST！本项目是面向神经网络量化的自适应非均匀量化框架的官方 PyTorch 实现。  
如有任何问题或建议，欢迎提交 Issue 或 Pull Request。

---
