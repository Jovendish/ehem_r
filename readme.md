## Requirments
- python3.10+
- pytorch3d https://github.com/facebookresearch/pytorch3d
- lightning
- open3d
- torchac

```
conda create -n ehem python=3.10 -y
conda activate ehem
pip install lightning rich open3d torchac
```

### Train & Test data preproc

```
cd utils & python prepare_data.py
```

### Training

```
python train.py
```
