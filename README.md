# DeAltHDR: Learning HDR Video Reconstruction from Degraded Alternating Exposure Sequences (ICLR 2026)

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://zhang-shuohao.github.io/DeAltHDR/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://zhang-shuohao.github.io/DeAltHDR/)
[![OpenReview](https://img.shields.io/badge/OpenReview-forum-green.svg)](https://openreview.net/forum?id=buzIPnGxA8)

---

## 📝 Abstract

High dynamic range (HDR) video can be reconstructed from low dynamic range (LDR) sequences with alternating exposures. However, most existing methods overlook the degradations (e.g., noise and blur) in LDR frames, focusing only on the brightness and position differences between them. To address this gap, we propose **DeAltHDR**, a novel framework for high-quality HDR video reconstruction from degraded sequences. Our framework addresses two key challenges. First, noisy and blurry contents complicate inter-frame alignment. To tackle this, we propose a **flow-guided masked attention mechanism** that leverages optical flow for a dynamic sparse cross-attention computation, achieving superior performance while maintaining efficiency. Notably, its controllable attention ratio allows for adaptive inference costs. Second, the lack of real-world paired data hinders practical deployment. We overcome this with a **two-stage training paradigm**: the model is first pre-trained on our newly introduced synthetic paired dataset and subsequently fine-tuned on unlabeled real-world videos via a proposed self-supervised method. Experiments show our method outperforms state-of-the-art ones. Code and data will be available at [https://zhang-shuohao.github.io/DeAltHDR/](https://zhang-shuohao.github.io/DeAltHDR/).

---

## 🔍 Method Overview

![Method Overview](../assets/overview.png)

Overview of our framework. Figure (a) illustrates the processing of the t-th frame in DeAltHDR, where the model uses the other 2 neighboring frames for assistance. Taking the alignment from the (t-1)-th frame to the t-th frame as an example, figure (b) shows how **Flow-Guided Mask Attention Alignment (FGMA)** works.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🏋️ Training

Edit `options/DeAltHDR.yml` to set your dataset path:

```yaml
dir_data: ['/path/to/your/dataset/']
```

Then run:

```bash
python train_dealthdr.py -opt options/DeAltHDR.yml --launcher none
```

---

## 🧪 Testing

```bash
python test_dealthdr.py -opt options/DeAltHDR.yml
```

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zhang2026dealthdr,
  title={DeAltHDR: Learning HDR Video Reconstruction from Degraded Alternating Exposure Sequences},
  author={Zhang, Shuohao and Zhang, Zhilu and Xu, Rongjian and Wu, Xiaohe and Zuo, Wangmeng},
  booktitle={ICLR},
  year={2026}
}
```
