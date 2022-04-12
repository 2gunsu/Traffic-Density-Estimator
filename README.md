# Traffic Density Estimator

## Environments
I have tested the code in the following environment.  
| OS                 | Python       | Pytorch      | CUDA         | GPU                   | NVIDIA Driver |
| ------------------ | ------------ | ------------ | ------------ | --------------------- | ------------- |
| Ubuntu 18.04.5 LTS | 3.7.13       | 1.9.1        | 11.1         | NVIDIA RTX A6000      | 470.57.02     |

## Preparations
```bash
# [Step 1]: 
# Create new conda environment and activate.
# Set [ENV_NAME] freely to any name you want. (Please exclude the brackets.)
conda create --name [ENV_NAME] python=3.7
conda activate [ENV_NAME]


# [Step 2]: Install some packages using 'requirements.txt' in the repository.
pip install -r requirements.txt


# [Step 3]: Install Pytorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html


# [Step 4]: Install Detectron2
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

## Datasets

## Usages
### Training

### Evaluation

### Test on Single Image

### Test on Multi Images


## Qualitative Results


## Quantitative Results

