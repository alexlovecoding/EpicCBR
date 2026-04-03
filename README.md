# EpicCBR

PyTorch implementation for **EpicCBR: Multi-view Contrastive Learning for Bundle Recommendation**.

## Environment

- OS: Ubuntu 18.04 or higher
- Python >= 3.7.11
- CUDA (tested): 10.2
- PyTorch >= 1.9.0

## Getting Started

### Installation

1. Clone this repository.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

To train EpicCBR on the NetEase dataset using GPU 0, run:

```bash
python train.py -g 0 -m EpicCBR -d NetEase
```

You can specify the GPU id and dataset using command line arguments. All hyper-parameters are configured in `config.yaml`.

### Command Line Arguments

- `-g`, `--gpu`: GPU id to use (default: 0)
- `-d`, `--dataset`: Dataset name (e.g., NetEase, iFashion)
- `-m`, `--model`: Model name (must be `EpicCBR`)
- `-i`, `--info`: Additional info to append to log file names

### Output

- Logs, checkpoints, and TensorBoard summaries will be saved in the corresponding directories under `./log/`, `./checkpoints/`, and `./runs/`.


## Citation

If you use this code for your research, please cite the original paper.

```bibtex
@inproceedings{li2026epiccbr,
  title={EpicCBR: Item-Relation-Enhanced Dual-Scenario Contrastive Learning for Cold-Start Bundle Recommendation},
  author={Li, Yihang and Liu, Zhuo and Wei, Wei},
  booktitle={Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  pages={377--386},
  year={2026}
}
```
---

For any questions or issues, please open an issue in this repository.
