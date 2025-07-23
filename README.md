# Leveraging Adversarial Learning for Pathological Fidelity in Virtual Staining (DGM4MICCAI, 2025)
Teixeira J., Klöckner P., Montezuma D., Fraga J., Horlings H. M., Cardoso J. S.
and Oliveira S. P., “Leveraging Adversarial Learning for Pathological Fidelity
in Virtual Staining.” In Deep Generative Models: 5th MICCAI Workshop -
DGM4MICCAI, 2025.

**Abstract**
Breast cancer is the leading cause of cancer-related deaths among women, making early detection crucial. In addition to evaluating tumor biopsies using H\&E staining, the traditional pathology workflow uses HER2 immunohistochemical staining to classify invasive carcinoma. This is a costly and labor-intensive technique, for which virtual staining, as an image-to-image translation task, emerges as a promising alternative. Although recent, this is an emerging field of research with 64\% of published studies just in 2024. Most studies use publicly available datasets of H\&E and IHC pairs from consecutive tissue sections. Recognizing the training challenges, many authors develop complex virtual staining models based on conditional Generative Adversarial Networks but ignore the impact of adversarial loss on the quality of virtual staining. Furthermore, overlooking the issues of model evaluation, they claim improved performance based on metrics such as SSIM and PSNR, which we argue are not sufficiently robust to evaluate the quality of virtually stained images. In this article, we developed CSSP2P GAN, which we demonstrate to achieve heightened pathological fidelity through a blind pathological expert evaluation. Furthermore, while iteratively developing our model, we study the impact of the adversarial loss and demonstrate its crucial role in the quality of virtually stained images. Finally, while comparing our model with seminal works in the field, we underscore the limitations of the currently used evaluation metrics and demonstrate the superior performance of CSSP2P GAN.

For contact, please use one of the following e-mail address:
- joset4259@gmail.com
- up202006243@up.pt

## Requirements and Setup
In this project, we used Python 3.11 and an NVIDIA A100-SXM4-40GB GPU.

### Clone Repository
```bash
# Clone Repository
git clone https://github.com/Jose-Miguel-Teixeira/CSSP2P-GAN.git
cd CSSP2P-GAN
```
### Prepare virtual environment

**Option 1:** pip Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

**Option 2:** conda Environment
```bash
conda env create -f environment.yaml
conda activate stains
```

### Download the HER2match Dataset
1. The HER2mathc dataset is available at: https://zenodo.org/records/15797050.

```bash
# Download the tiles
wget 'https://zenodo.org/record/15797050/files/tiles.zip?download=1' -O HER2match_tiles.zip
unzip HER2match_tiles.zip -d HER2match_tiles
```

2. Modify the `conf/datamodule.yaml` file by inserting the full path to the dataset in the following parameters:
- train_dataroot
- test_dataroot

## How to Run
### Train Models
- BCE GAN
```bash
python3 train_gan.py \
    --config-name BCEGAN_config.yaml \
    ++discriminator.input_nc=3 \
    ++train.condition_discriminator=false \
    ++hydra.run.dir=outputs/BCEGAN/$(date +%Y_%m_%d_%H_%M_%S) \
    ++general.experiment_name=BCEGAN_$(date +%Y_%m_%d_%H_%M_%S)
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

>**Note:**
>- The hyperparameters used to train the models are located in the `conf` folder. To reproduce the results from the article, it is recommended to retain the default settings. However, feel free to modify these parameters to conduct further experiments.
>- If you want to use WandbLogger, please input your entity name on the configuration file `conf > logger > WandbLogger.yaml`
>- To re-train [Pyramid Pix2Pix](https://github.com/bupt-ai-cz/BCI) and [ASP](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE), please refer to their original code repository.

### Inference
After training the models, you can predict and evaluate the quality of the virtually stained images following these steps:
1. Navigate to the evaluate directory:
```bash
cd evaluate
```
**Inference on the Validation Set**

1. Predict on the Validation Set.

Before running `predict_gan.py`, modify the `conf/predict_gan.yaml` file by updating the following parameters:
- `module`: `<model_name>`  
  → Choose one of: `BCE GAN`, `P2P GAN`, or `CSSP2P GAN`
- `train.resume_from_checkpoint`: `<path/to/checkpoint>`  
  → Path to the model checkpoint you want to evaluate
- `discriminator_input_nc`: `6`  
  → Use `3` if evaluating **BCE GAN without H&E conditioning**
- `train.condition_discriminator`: `true`  
  → Set to `false` if evaluating **BCE GAN without H&E conditioning**
- `hydra.run.dir`: `<path/to/output>`  
  → Directory to store outputs and logs for this run
- `general.experiment_name`: `<experiment_name>` *(optional)*  
  → Name of the experiment for logging

> Note: You can omit `general.experiment_name` if you don’t need to label the run explicitly.

```bash
python3 predict_gan.py
```

2. Run the metrics evaluation script.
```bash
python3 evaluate.py \
    ++predictions_dir=<path to your predictions> \
    ++phase=val \
    ++HE_dir=../../HER2match_tiles/HE/val
    ++target_dir=../../HER2match_tiles/IHC/val
```

**Inference on the Test Set**
1. Predict on the Test Set.

Before running `test_gan.py`, modify the `conf/test_gan.yaml` file by updating the following parameters:
- `module`: `<model_name>`  
  → Choose one of: `BCE GAN`, `P2P GAN`, or `CSSP2P GAN`
- `train.resume_from_checkpoint`: `<path/to/checkpoint>`  
  → Path to the model checkpoint you want to evaluate
- `discriminator_input_nc`: `6`  
  → Use `3` if evaluating **BCE GAN without H&E conditioning**
- `train.condition_discriminator`: `true`  
  → Set to `false` if evaluating **BCE GAN without H&E conditioning**
- `hydra.run.dir`: `<path/to/output>`  
  → Directory to store outputs and logs for this run
- `general.experiment_name`: `<experiment_name>` *(optional)*  
  → Name of the experiment for logging

> Note: You can omit `general.experiment_name` if you don’t need to label the run explicitly.

```bash
python3 test_gan.py
```

2. Run the metrics evaluation script.
```bash
python3 evaluate.py \
    ++predictions_dir=<path to your predictions> \
    ++phase=test \
    ++HE_dir=../../HER2match_tiles/HE/test \
    ++target_dir=../../HER2match_tiles/IHC/test
```

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
