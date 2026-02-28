# DeAltHDR

Official PyTorch implementation of **DeAltHDR** (ICLR 2026).

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

Edit `options/DeAltHDR.yml` to set your dataset path:
```yaml
dir_data: ['/path/to/your/dataset/']
```

Then run:
```bash
python train_dealthdr.py -opt options/DeAltHDR.yml --launcher none
```

## Testing

```bash
python test_dealthdr.py -opt options/DeAltHDR.yml
```

## Citation

```bibtex
@inproceedings{zhang2026dealthdr,
  title={DeAltHDR: Learning HDR Video Reconstruction from Degraded Alternating Exposure Sequences},
  author={Zhang, Shuohao and Zhang, Zhilu and Xu, Rongjian and Wu, Xiaohe and Zuo, Wangmeng},
  booktitle={ICLR},
  year={2026}
}
```
