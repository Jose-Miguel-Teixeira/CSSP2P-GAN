# Leveraging Adversarial Learning for Pathological Fidelity in Virtual Staining (DGM4MICCAI, 2025)
Teixeira J., Klöckner P., Montezuma D., Fraga J., Horlings H. M., Cardoso J. S.
and Oliveira S. P., “Leveraging Adversarial Learning for Pathological Fidelity
in Virtual Staining.” In Deep Generative Models: 5th MICCAI Workshop -
DGM4MICCAI, 2025.

For contact please use one of the following e-mail address:
- joset4259@gmail.com
- up202006243@up.pt

## Requirements
In this project we used python3.11 and a NVIDIA A100 GPU.

### Prepare Virtual Environment
1. Clone Repository
```bash
# Clone Repository
git clone https://github.com/Jose-Miguel-Teixeira/CSSP2P-GAN.git
cd CSSP2P-GAN
```
2. Prepare virtual environment

**Option 1:** Virtual Environment using pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

**Option 2:** Virtual Environment using conda
```bash
conda env create -f environment.yaml
conda activate stains
```

### Download the HER2match Dataset
The HER2mathc dataset is available at: https://zenodo.org/records/15797050.

```bash
# Download the tiles
wget 'https://zenodo.org/record/15797050/files/tiles.zip?download=1' -O HER2match_tiles.zip
unzip HER2match_tiles.zip -d HER2match_tiles
```

## How to Run
### Train Models
- BCE GAN
```bash
python3 train_gan.py \
    --config-name BCEGAN_config.yaml
    ++discriminator.input_nc=3 \
    ++train.condition_discriminator=false
```

- cBCE GAN
```bash
python3 train_gan.py --config-name BCEGAN_config.yaml
```

- P2P GAN
```bash
python3 train_gan.py --config-name P2PGAN_config.yaml
```

- CSSP2P GAN
```bash
python3 train_gan.py --config-name CSSP2PGAN_config.yaml
```

**Note:**
- The hyperparameters used to train the models are located in the `conf` folder. To reproduce the results from the article, it is recommended to retain the default settings. However, feel free to modify these parameters to conduct further experiments.
- If you want to use WandbLogger, please input your entity name on the configuration file `conf > logger > WandbLogger.yaml`
- To re-train [Pyramid Pix2Pix](https://github.com/bupt-ai-cz/BCI) and [ASP](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE), please refer to their original code repository.

### Inference on Validation Set

### Inference on Test Set


## Citation
If you use this code for your research, please cite our paper **Leveraging Adversarial Learning for Pathological Fidelity in Virtual Staining**:
```bibtext
@inproceedings{teixeira2025,
  author    = {Teixeira, J. and Klöckner, P. and Montezuma, D. and Fraga, J. and Horlings, H. M. and Cardoso, J. S. and Oliveira, S. P.},
  title     = {Leveraging Adversarial Learning for Pathological Fidelity in Virtual Staining},
  booktitle = {Proceedings of the 5th MICCAI Workshop on Deep Generative Models (DGM4MICCAI)},
  year      = {2025}
}
```

## Acknowledgments
This project uses code from the [AdaptiveSupervisedPatchNCE repository](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE) by Fangda Li (2023), which is licensed under a custom license. The following files contain a copy or a modified version of the original code:
- models/networks.py
- models/discriminator.py