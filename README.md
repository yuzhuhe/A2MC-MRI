# A²MC-MRI: Anatomy-Aware Deep Unrolling for Task-Oriented Acceleration of Multi-Contrast MRI

## 🔍 Description

**A²MC-MRI** is an anatomy-aware, unrolling-based deep network designed for accelerated multi-contrast MRI (MC-MRI) reconstruction. Unlike traditional methods that prioritize overall image quality, A²MC-MRI follows a **task-oriented** strategy, focusing on enhancing specific **Targets of Interest (TOIs)**—such as subcortical structures or specific pathologies—that are strictly relevant to downstream clinical objectives.

### Key Innovations:
* **Anatomy-Aware Denoising Prior**: Integrates a segmentation network (**P-net**) to identify TOIs, enabling refined constraints on the discrepancy between the reconstructed and denoised images within those specific regions.
* **Learnable Group Sparsity**: Minimizes the $l_{2,1}$-norm in a high-dimensional semantic space to capture intrinsic correlations and facilitate information fusion across different contrasts (e.g., T1, T2, and FLAIR).
* **Model-Unrolled Architecture**: Unfolds an iterative reconstruction algorithm into a multi-stage deep network ($T=6$) to improve interpretability and generalizability.
* **Task-Oriented k-Space Sampling**: Concurrently learns multi-contrast sampling patterns tailored to specific clinical needs and anatomical targets.

---

## 🧭 Framework Overview

The A²MC-MRI network consists of cascaded stages, each containing three pivotal modules:

1. **Denoising Module ($D_w$)**: A lightweight U-Net that produces images free of alias and artifacts from the previous stage.
2. **Anatomy-Aware Data-Consistency Module ($A^2\mathcal{DC}$)**: Incorporates the P-net to localize TOIs and applies refined constraints to enhance imaging quality in clinically relevant areas.
3. **Group Sparsity Module (GS)**: Enhances cross-contrast complementarity by capturing joint sparsity features in the informative semantic space.

![A2MC-MRI Architecture](assets/framework.png)

---

## 📦 Requirements & Dependencies

* **OS**: Ubuntu 20.04
* **GPU**: NVIDIA RTX 3090Ti (or compatible)
* **Environment**: Python 3.10+, PyTorch
---

## 🗂 Data Preparation

The framework was evaluated on three major datasets under high acceleration ratios (8x and 10x):

| Dataset | Contrasts | Target of Interest (TOI) | Field Strength |
| :--- | :--- | :--- | :--- |
| **M4Raw** | T1WI, T2WI, FLAIR | Brain subcortical regions | 0.3T |
| **fastMRI** | PDWI, FS-PDWI | Knee lesion / meniscus regions | 1.5/3.0T |
| **In-house** | T1WI, T2WI, FLAIR | Whole brain tissue | 3.0T |

---

## 🚀 Training & Usage

### Loss Function
The network is trained using a composite loss function to balance reconstruction fidelity and segmentation accuracy:
$\mathcal{L}(\Theta)$

### Training Details
* **Optimizer**: Adam
* **Learning Rate**: $1 \times 10^{-4}$
* **Batch Size**: 4
* **Stages ($T$)**: 6

## 🔗 Citation
If you use this code or method in your research, please cite the following paper:
**Paper:** [Anatomy-Aware Deep Unrolling for Task-Oriented Acceleration of Multi-Contrast MRI](https://ieeexplore.ieee.org/document/10994324)
**Code:** [https://github.com/ladderlab-xjtu/A2MC-MRI](https://github.com/ladderlab-xjtu/A2MC-MRI)
```bibtex
@ARTICLE{10994324,
  author={He, Yuzhu and Lian, Chunfeng and Xiao, Ruyi and Ju, Fangmao and Zou, Chao and Xu, Zongben and Ma, Jianhua},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Anatomy-Aware Deep Unrolling for Task-Oriented Acceleration of Multi-Contrast MRI}, 
  year={2025},
  volume={44},
  number={9},
  pages={3832-3844},
  keywords={Image reconstruction;Imaging;Magnetic resonance imaging;Image segmentation;Image quality;Noise reduction;Trajectory optimization;Training;Protocols;Pathology;MRI;multi-contrast;image reconstruction;task-oriented imaging;model-based deep learning},
  doi={10.1109/TMI.2025.3568157}}
