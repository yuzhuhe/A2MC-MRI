# A²MC-MRI: Anatomy-Aware Deep Unrolling for Task-Oriented Acceleration of Multi-Contrast MRI

## 🔍 Abstract
[cite_start]Multi-contrast magnetic resonance imaging (MC-MRI) is crucial in clinical practice but is often hindered by long scanning times and the isolation between image acquisition and downstream clinical needs[cite: 7, 8]. [cite_start]We propose an anatomy-aware unrolling-based deep network, dubbed **A²MC-MRI**, offering promising interpretability and learning capacity for fast MC-MRI[cite: 11]. [cite_start]By integrating learnable group sparsity with an anatomy-aware denoising prior, the model enhances concurrent MC-MRI of specific targets of interest (TOIs)[cite: 13]. [cite_start]Comprehensive evaluations demonstrate state-of-the-art performance in reconstruction under high acceleration rates, featuring notable enhancements in TOI imaging quality[cite: 16].

---

## 🧭 Framework Overview
[cite_start]The A²MC-MRI network consists of $T=6$ cascaded stages, each containing three pivotal modules[cite: 196, 260]:

1. [cite_start]**Denoising Module ($D_w$)**: Utilizes a lightweight U-Net architecture to produce images free of alias and artifacts[cite: 204].
2. [cite_start]**Anatomy-Aware Data-Consistency Module ($A^2\mathcal{DC}$)**: Refines the data-consistency estimation by incorporating a pre-trained segmentation network (**P-net**) to localize TOIs via probability maps[cite: 212, 216].
3. [cite_start]**Group Sparsity Module (GS)**: Employs $l_{2,1}$-norm-based group sparsity in a high-dimensional semantic space to fuse complementary information across various contrasts[cite: 222, 227].

---

## 📦 Requirements & Dependencies
* [cite_start]**OS**: Ubuntu 20.04 [cite: 259]
* [cite_start]**GPU**: NVIDIA RTX 3090Ti [cite: 259]
* [cite_start]**Environment**: Python, PyTorch [cite: 259, 261]
* **Key Libraries**: `numpy`, `scipy`, `monai`, `nibabel`, `einops`, `transformers`

---

## 🗂 Datasets & TOI Definitions
[cite_start]The model was evaluated on three datasets under high acceleration ratios (8x and 10x)[cite: 239, 258]:

| Dataset | Contrasts | Target of Interest (TOI) |
| :--- | :--- | :--- |
| **M4Raw** | T1WI, T2WI, FLAIR | [cite_start]Brain subcortical regions [cite: 242, 244] |
| **fastMRI** | PDWI, FS-PDWI | [cite_start]Knee lesion / meniscus regions [cite: 246, 249] |
| **In-house** | T1WI, T2WI, FLAIR | [cite_start]Whole brain tissue [cite: 250, 252] |

---

## 🚀 Training & Optimization
### Loss Function
[cite_start]The network is trained using a composite loss function[cite: 232, 233]:
$$\mathcal{L}(\Theta) = \mathcal{L}_{consistency} + \alpha\mathcal{L}_{constraint} + \beta\mathcal{L}_{anatomy} + \gamma\mathcal{L}_{dice}$$
* [cite_start]**$\mathcal{L}_{consistency}$**: Global fidelity to ground-truth[cite: 233].
* [cite_start]**$\mathcal{L}_{constraint}$**: Ensures the inverse transform constraint[cite: 234].
* [cite_start]**$\mathcal{L}_{anatomy}$**: Refines reconstruction specifically within TOIs[cite: 234].
* [cite_start]**$\mathcal{L}_{dice}$**: Promotes accurate segmentation within the reconstructed images[cite: 235].

### Hyperparameters
* [cite_start]**Epochs**: 100 [cite: 261]
* [cite_start]**Batch Size**: 4 [cite: 261]
* [cite_start]**Learning Rate**: $1 \times 10^{-4}$ (Adam optimizer) [cite: 261]
* [cite_start]**Stages ($T$)**: 6 [cite: 260]

---

## 📊 Experimental Results
* [cite_start]**State-of-the-Art**: A²MC-MRI consistently surpasses competing methods like MT-Trans and MC-J-MoDL in PSNR and SSIM[cite: 311].
* [cite_start]**TOI Improvement**: Achieved a **1.16 dB** TOI-PSNR improvement for T2WI reconstruction on the M4Raw dataset (10x) compared to MC-J-MoDL[cite: 346].
* [cite_start]**Efficiency**: Moderate parameter count (6.71M) with high efficacy compared to purely data-driven models[cite: 449, 455].

---

## 🔗 Citation
If you use this code or method in your research, please cite the following paper:

```bibtex
@article{he2024anatomy,
  title={Anatomy-Aware Deep Unrolling for Task-Oriented Acceleration of Multi-Contrast MRI},
  author={He, Yuzhu and Lian, Chunfeng and Xiao, Ruyi and Ju, Fangmao and Zou, Chao and Wang, Fan and Xu, Zongben and Ma, Jianhua},
  journal={IEEE Transactions on Medical Imaging},
  volume={XX},
  number={XX},
  pages={XXXX--XXXX},
  year={2024},
  publisher={IEEE}
}
