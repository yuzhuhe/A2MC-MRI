# A2MC-MRI
# A²MC-MRI: Anatomy-Aware Deep Unrolling for Task-Oriented Acceleration of Multi-Contrast MRI

## 🔍 Description

[cite_start]**A²MC-MRI** is an anatomy-aware, unrolling-based deep network designed for accelerated multi-contrast MRI (MC-MRI) reconstruction[cite: 11]. [cite_start]Unlike traditional methods that prioritize overall image quality, A²MC-MRI is a **task-oriented** framework that focuses on enhancing specific **Targets of Interest (TOIs)**—such as specific pathologies or anatomical regions—to better serve downstream clinical needs[cite: 11, 49].

The framework uniquely integrates:
* [cite_start]**Learnable Group Sparsity**: Captures intrinsic correlations across different contrasts in a high-dimensional semantic space[cite: 13, 51].
* [cite_start]**Anatomy-Aware Denoising Prior**: Utilizes a segmentation network to provide critical location information, enabling specialized denoising for TOIs[cite: 13, 14, 52].
* [cite_start]**Joint Learning**: The unrolled network is trained in tandem with k-space sampling patterns to optimize imaging for specific clinical tasks[cite: 15, 53].

---

## 🧭 Framework Overview

[cite_start]The A²MC-MRI network consists of $T=6$ cascaded stages[cite: 196, 260]. [cite_start]Each stage contains three pivotal modules[cite: 196, 201]:
1. [cite_start]**Denoising Module ($D_w$)**: Employs a lightweight U-Net to produce artifact-free images[cite: 202, 204].
2. [cite_start]**Anatomy-Aware Data-Consistency Module ($A^2\mathcal{DC}$)**: Incorporates a pre-trained segmentation network (**P-net**) to identify TOIs and refine reconstruction fidelity[cite: 202, 206, 216].
3. [cite_start]**Group Sparsity Module (GS)**: Enhances cross-contrast information fusion within a meaningful semantic space[cite: 202, 227].

![A2MC-MRI Architecture](assets/framework.png)

---

## 📦 Requirements

* **OS**: Ubuntu 20.04 [cite: 259]
* [cite_start]**GPU**: NVIDIA RTX 3090Ti [cite: 259]
* [cite_start]**Environment**: Python 3.x, PyTorch [cite: 259]
* **Dependencies**: NumPy, SciPy, Matplotlib, Nibabel, etc.

---

## 🗂 Data Preparation

The model has been comprehensively evaluated on three datasets[cite: 239]:

| Dataset | Contrasts | Target of Interest (TOI) |
| :--- | :--- | :--- |
| **M4Raw** (0.3T) | T1WI, T2WI, FLAIR | Brain subcortical regions [cite: 241, 244] |
| **fastMRI** (1.5/3T) | PDWI, FS-PDWI | Knee lesion / meniscus regions [cite: 246, 249, 304] |
| **In-house** (3.0T) | T1WI, T2WI, FLAIR | Whole brain tissue [cite: 250, 252] |

---

## 🚀 Usage

### Training
The network uses a composite loss function to balance image fidelity and segmentation accuracy[cite: 232]:
$$\mathcal{L}(\Theta) = \mathcal{L}_{consistency} + \alpha\mathcal{L}_{constraint} + \beta\mathcal{L}_{anatomy} + \gamma\mathcal{L}_{dice}$$ [cite: 233]

To start training, run:
```bash
python train.py --dataset M4Raw --acceleration 10
