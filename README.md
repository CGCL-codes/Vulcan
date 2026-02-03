# 💫 Vulcan · Class-Specific ViT Derivation 💫

## 🌀 What is Vulcan?
<div align="center">
  <img src="img/overview.jpg" style="width:67%;">
</div>

🚀 Vulcan is a novel approach for deriving compact, class-specific Vision Transformers (ViTs) tailored for resource-constrained edge devices. 
🎯 Given a pre-trained [base ViT](https://github.com/facebookresearch/deit/blob/main/README_deit.md), Vulcan can derive a lightweight ViTs that focus on recognizing the target classes.

## 📂 Project Structure
| Folder/File            | Description                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| `src/data`             | Stores sub-task definitions and intermediate experimental results.                                 |
| `src/dataset`          | Dataset loading, processing, and augmentation utilities.                                           |
| `src/engine`           | Core training and evaluation pipelines.                                                            |
| `src/method`           | Core implementations of Vulcan, including CCNC and TNNR losses, adaptive configuration, and structured pruning. |
| `src/model`            | Model definitions and loading utilities for ViT and Swin backbones.                                |
| `src/scripts`          | Shell scripts for running Vulcan experiments with different models, tasks, and pruning configurations.          |
| `src/utils`            | General utility functions for profiling, FLOPs/parameter calculation, memory analysis, and training support.    |
| `src/main.py`          | Main entry point to run Vulcan, including post-training and pruning.                               |

## 🚀 Quick Start
1. **Clone the repository**

First, clone the NuWa project to your local machine:
```bash
git clone https://github.com/xxx/vulcan.git
cd vulcan/scripts
```
2. **Install required dependencies**
3. **Run the pipeline**
```bash
./vulcan_base.sh
```

## 📎 Citation

If you find this code useful, please cite our paper:

```bibtex
@article{wei2026vulcan,
  title={Vulcan: Crafting compact class-specific Vision Transformers for edge intelligence},
  author={Wei, Ziteng and He, Qiang and Chen, Feifei and Duan, Ranjie and Li, Xiaodan and Li, Bin and Chen, Yuefeng and Xue, Hui and Jin, Hai and Yang, Yun},
  journal={International Conference on Learning Representations},
  year={2026}
}
```
