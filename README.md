# Traffic Density Estimator

## Environments
I have tested the code in the following environment.
| OS                 | Python       | Pytorch      | CUDA         | GPU                   | NVIDIA Driver |
| :----------------: | :----------: | :----------: | :----------: | :-------------------: | :-----------: |
| Ubuntu 18.04.5 LTS | 3.7.13       | 1.9.1        | 11.1         | NVIDIA RTX A6000      | 470.57.02     |

## Preparations
```bash
# [Step 1]: Create new conda environment and activate.
#           Set [ENV_NAME] freely to any name you want. (Please exclude the brackets.)
conda create --name [ENV_NAME] python=3.7
conda activate [ENV_NAME]


# [Step 2]: Install some packages using 'requirements.txt' in the repository.
pip install -r requirements.txt


# [Step 3]: Install Pytorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html


# [Step 4]: Install Detectron2
#           Detectron2 is a platform for object detection, segmentation and other visual recognition tasks.
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

## Datasets
### DOTA: A Large-scale Dataset for Object Detection in Aerial Images [[Paper](https://arxiv.org/abs/1711.10398)] [[Site](https://captain-whu.github.io/DOTA/dataset.html)]
You can download pre-processed DOTA dataset in this **[link](https://drive.google.com/file/d/1NPdqu3CQWEX6639OV5c6Tletb3lN7eci/view?usp=sharing)** directly. (7.3GB)  
Some image samples are shown below.  

<img src="https://user-images.githubusercontent.com/59532188/163018184-4f25fb8b-137f-4b96-9cbd-ef7d7d8377e8.png" width=250 height=250> <img src="https://user-images.githubusercontent.com/59532188/163018231-54cb510f-2bc8-4503-b2be-6edcbe13a77b.png" width=250 height=250> <img src="https://user-images.githubusercontent.com/59532188/163018270-8fa3fb7c-36ff-4aec-8d46-c7eecbf3f875.png" width=250 height=250>

The structure of downloaded data is as follows.

```
DOTA.zip
|-- Train
|   |-- Label.json
|   `-- Image
|       |-- Image_00000.png
|       |-- Image_00001.png
|       |-- Image_00002.png
|       `-- ...
|-- Test
|   |-- Label.json
|   `-- Image
|       |-- Image_00042.png
|       |-- Image_00055.png
|       |-- Image_00060.png
|       `-- ...
|-- Val
|   |-- Label.json
|   `-- Image
|       |-- Image_00066.png
|       |-- Image_00125.png
|       |-- Image_00130.png
|       `-- ...
`-- Mini
    |-- Label.json
    `-- Image
        |-- Image_00066.png
        |-- Image_00125.png
        |-- Image_00130.png
        `-- ...
```


## Usages
### Pretrained Weights
Download the config files and pretrained weights from the table below.
| Trained Dataset    | With Denoiser | Noise Type  | Backbone             | Config File  | Weight File |
| :----------------: | :-----------: | :---------: | :------------------: | :----------: | :---------: |
| DOTA               |       X       |      -      | ResNeXt-101-FPN      | Download     | Download    |
| DOTA               |       O       |  Gaussian   | ResNeXt-101-FPN      | Download     | Download    |

### Training

### Evaluation

### Test on Single Image

### Test on Multi Images
