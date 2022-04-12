# Noise-Robust Traffic Density Estimator
**Project page for Capstone Design 1(2020/03 ~ 2020/06) and Capstone Design 2(2020/09 ~ 2020/12).**

Traffic density is estimated using the mask of the vehicle extracted from satellite image through Mask R-CNN.  
By attaching the denosing network to the Mask R-CNN, the above process can be more robust to noise in image.

#### [Step 1] Remove Noise from Image
<img src="https://user-images.githubusercontent.com/59532188/163022985-740996b6-e343-4679-a7b3-85d1429dc5b2.png" width=430> <img src="https://user-images.githubusercontent.com/59532188/163023357-d5608e80-6582-451c-b123-b4c17dbe77da.gif" width=430>

#### [Step 2] Extract Vehicle Masks from Clean Image 
<img src="https://user-images.githubusercontent.com/59532188/163021696-13087f11-c695-48e5-a5db-135c510ae804.png" width=430> <img src="https://user-images.githubusercontent.com/59532188/163021732-42898c6b-4d13-4220-9fac-6efbc724975e.gif" width=430>

#### [Step 3] Convert Vehicle Masks to Density Map
<img src="https://user-images.githubusercontent.com/59532188/163023971-d2baf397-8be6-48ef-b2f7-798f6494510e.png" width=300> <img src="https://user-images.githubusercontent.com/59532188/163022216-e50c9657-b0b4-4fa4-8705-890f18ea17b9.png" width=300> <img src="https://user-images.githubusercontent.com/59532188/163023718-031d3ab0-e1fc-47aa-9fbf-2abfb4e8cdba.png" width=300>


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

# [Step 3]: Install Pytorch v1.9.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# [Step 4]: Install Detectron2 v0.5
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
Not added yet

### Evaluation
Not added yet

### Test on Single Image
Not added yet

### Test on Multi Images
Not added yet

## Quantitative Results
<img src="https://user-images.githubusercontent.com/59532188/163020764-9802fc98-9a13-474f-9f48-89480bdbcbd9.png" width=600>

## References
[1] Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. **["Noise2Void - Learning Denoising from Single Noisy Images."](https://arxiv.org/abs/1811.10980)** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

