# Noise-Robust Traffic Density Estimator
**Project page for Capstone Design 1(2020/03 ~ 2020/06) and Capstone Design 2(2020/09 ~ 2020/12).**

Traffic density is estimated using the mask of the vehicle extracted from satellite image through **Mask R-CNN**.  
By attaching the denosing network to the Mask R-CNN, the above process can be more robust to noise in image.

#### [Step 1] Remove Noise from Image
<img src="https://user-images.githubusercontent.com/59532188/163022985-740996b6-e343-4679-a7b3-85d1429dc5b2.png" width=400> <img src="https://user-images.githubusercontent.com/59532188/163023357-d5608e80-6582-451c-b123-b4c17dbe77da.gif" width=400>

#### [Step 2] Extract Vehicle Masks from Clean Image 
<img src="https://user-images.githubusercontent.com/59532188/163021696-13087f11-c695-48e5-a5db-135c510ae804.png" width=400> <img src="https://user-images.githubusercontent.com/59532188/163021732-42898c6b-4d13-4220-9fac-6efbc724975e.gif" width=400>

#### [Step 3] Convert Vehicle Masks to Density Map (Click to enlarge images)
<img src="https://user-images.githubusercontent.com/59532188/163023971-d2baf397-8be6-48ef-b2f7-798f6494510e.png" width=270> <img src="https://user-images.githubusercontent.com/59532188/163022216-e50c9657-b0b4-4fa4-8705-890f18ea17b9.png" width=270> <img src="https://user-images.githubusercontent.com/59532188/163023718-031d3ab0-e1fc-47aa-9fbf-2abfb4e8cdba.png" width=270>


## Environments
We have tested the code in the following environment.
| OS                 | Python       | Pytorch      | CUDA         | GPU                   | NVIDIA Driver |
| :----------------: | :----------: | :----------: | :----------: | :-------------------: | :-----------: |
| Ubuntu 18.04.5 LTS | 3.7.13       | 1.9.1        | 11.1         | NVIDIA RTX A6000      | 470.57.02     |

## Preparations
```bash
# [Step 1]: Create new conda environment and activate.
#           Set [ENV_NAME] freely to any name you want. (Please exclude the brackets.)
conda create --name [ENV_NAME] python=3.7
conda activate [ENV_NAME]

# [Step 2]: Clone the repository.
git clone https://github.com/2gunsu/Traffic-Density-Estimator
cd Traffic-Density-Estimator

# [Step 3]: Install some packages using 'requirements.txt' in the repository.
pip install -r requirements.txt

# [Step 4]: Install Pytorch v1.9.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# [Step 5]: Install Detectron2 v0.5
#           Detectron2 is a platform for object detection, segmentation and other visual recognition tasks.
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

## Datasets
### DOTA: A Large-scale Dataset for Object Detection in Aerial Images [[Paper](https://arxiv.org/abs/1711.10398)] [[Site](https://captain-whu.github.io/DOTA/dataset.html)]
There are several classes in this dataset, but we only used the classes belonging to the vehicle among them.  
You can download pre-processed DOTA dataset in this **[link](https://drive.google.com/file/d/1NPdqu3CQWEX6639OV5c6Tletb3lN7eci/view?usp=sharing)** directly. (7.3GB)   
Some image samples and its corresponding annotations are shown below.  

<img src="https://user-images.githubusercontent.com/59532188/163309746-53a443f6-fe61-4130-8d48-e01c6ad03549.png" width=250 height=250> <img src="https://user-images.githubusercontent.com/59532188/163309757-05551156-bd10-4703-a279-79076305841f.png" width=250 height=250> <img src="https://user-images.githubusercontent.com/59532188/163309777-67790e26-5ca2-41cc-818b-8d1b83f481a6.png" width=250 height=250>  
<img src="https://user-images.githubusercontent.com/59532188/163309767-2d615a45-b9d9-4912-b97b-811ae746844d.png" width=250 height=250> <img src="https://user-images.githubusercontent.com/59532188/163309775-6b797342-596e-4f15-89c9-cfeb200999bf.png" width=250 height=250> <img src="https://user-images.githubusercontent.com/59532188/163309763-a09480c2-1b6d-47a7-bfed-1290469c0bb5.png" width=250 height=250>  


## Usages
### Pretrained Weights
Download the config files and pretrained weights from the table below.
| Trained Dataset    | With Denoiser | Noise Type  | Backbone             | Config File  | Weight File |
| :----------------: | :-----------: | :---------: | :------------------: | :----------: | :---------: |
| DOTA               |       X       |      -      | ResNeXt-101-FPN      | [Download](https://drive.google.com/file/d/1ty3IxMOi8TCIf9z_VdXNEWebq3vCXxPG/view?usp=sharing)     | [Download](https://drive.google.com/file/d/1FJK3iQhVtRMAWPrMreSEaUJ-3WBXOV4O/view?usp=sharing)    |


### Training
Please check more detailed parameters in `train.py` and follow the script below for a quick start.  
The data path(`--train_path` or `--val_path`) must contain an `Image` folder and a `Label.json` file.  
Skip the training if you only want to use the pretrained model.  

```bash
python train.py --train_path    [str]   # Directory of training data
                --val_path      [str]   # Directory of validation data
                --output_dir    [str]   # Output directory
                --backbone_arch [str]   # Select one in ['R50-FPN', 'R101-FPN', 'X101-FPN'].
                                        # 'R' denotes for ResNet, 'X' denotes for ResNeXt
                --gpu_id        [int]   # Index of the GPU to be used for training
                --epochs        [int]   
                --batch_size    [int]
```

### Evaluation
```bash
python evaluation.py --eval_path      [str]   # Directory of evaluation data
                     --config_file    [str]   # Path of config file (.yaml)
                     --weight_file    [str]   # Path of weight file (.pth)
                     --gpu_id         [int]   # Index of the GPU to be used for evaluation
```

### Inference on Single Image
Follow the script below to test  general sized images.  
```bash
python test.py --config_file    [str]   # Path of config file (.yaml)
               --weight_file    [str]   # Path of weight file (.pth)
               --conf_score     [float] # Confidence threshold for inference
               --gpu_id         [int]   # Index of the GPU to be used for inference
               --image_file     [str]   # Path of single image file
               --save_dir       [str]
```

But if you want to process very high resolution images, follow the script below.  
The script below splits the large image into smaller patches determined by `--grid_size`, inferences them individually, and merges them back together.  
```bash
python test.py --config_file    [str]   # Path of config file (.yaml)
               --weight_file    [str]   # Path of weight file (.pth)
               --conf_score     [float] # Confidence threshold for inference
               --gpu_id         [int]   # Index of the GPU to be used for inference
               --image_file     [str]   # Path of large-sized image file
               --save_dir       [str]

               --grid_split             # [Optional]
               --grid_size      [int]   # [Optional] Determine the size of patches
```

### Inference on Multiple Images
```bash
python test.py --config_file    [str]   # Path of config file (.yaml)
               --weight_file    [str]   # Path of weight file (.pth)
               --conf_score     [float] # Confidence threshold for inference
               --gpu_id         [int]   # Index of the GPU to be used for inference
               --image_dir      [str]   # Directory which contains multiple images
               --save_dir       [str]

               --grid_split             # [Optional]
               --grid_size      [int]   # [Optional] Determine the size of patches
```

## Quantitative Results
<img src="https://user-images.githubusercontent.com/59532188/163020764-9802fc98-9a13-474f-9f48-89480bdbcbd9.png" width=600>  


## Qualitative Results  
Please click to enlarge the image.  

#### (1) Daejeon, South Korea
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/1_1.png"><img src="https://user-images.githubusercontent.com/59532188/163596595-cd7095da-32a0-4a71-8118-ed374b7d9858.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/1_2.png"><img src="https://user-images.githubusercontent.com/59532188/163596596-25575371-bc63-40c3-95cc-274355b82943.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/1_3.png"><img src="https://user-images.githubusercontent.com/59532188/163596598-6aaf03b1-64de-4692-9722-879145584708.png" width=250 height=250></a>

#### (2) Incheon, South Korea
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/2_1.png"><img src="https://user-images.githubusercontent.com/59532188/163596601-e43f64d7-405b-4aeb-8c64-08452b39b169.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/2_2.png"><img src="https://user-images.githubusercontent.com/59532188/163596603-e929ece1-7eda-4868-a099-f06549c9f52d.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/2_3.png"><img src="https://user-images.githubusercontent.com/59532188/163596606-82d30f98-d711-4b62-be3e-f2541b8ee9cf.png" width=250 height=250></a>

#### (3) Seoul, South Korea
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/3_1.png"><img src="https://user-images.githubusercontent.com/59532188/163596683-3825f759-4e32-477f-bd4f-b14e45c3b07e.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/3_2.png"><img src="https://user-images.githubusercontent.com/59532188/163596686-ac1a0853-b6bd-492e-897f-26a679e34a37.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/3_3.png"><img src="https://user-images.githubusercontent.com/59532188/163596689-a9b6b77f-157c-4be0-979e-f31dafabbfe2.png" width=250 height=250></a>

#### (4) Busan, South Korea
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/4_1.png"><img src="https://user-images.githubusercontent.com/59532188/163596690-d48820de-3063-4927-b387-2ba2156db1eb.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/4_2.png"><img src="https://user-images.githubusercontent.com/59532188/163596691-8dfd6535-ca81-4f3e-b9f5-54b6ba68ba3a.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/4_3.png"><img src="https://user-images.githubusercontent.com/59532188/163596696-7d640562-a52e-4874-b274-d646c134b1d3.png" width=250 height=250></a>

#### (5) New York, United States of America
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/5_1.png"><img src="https://user-images.githubusercontent.com/59532188/163596763-a578e239-b25e-47da-b9b6-59a1ab0e36de.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/5_2.png"><img src="https://user-images.githubusercontent.com/59532188/163596765-d137f3d2-5e3f-4645-8a71-ff529b6ee6c8.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/5_3.png"><img src="https://user-images.githubusercontent.com/59532188/163596766-1fddd017-3c3d-43e4-b0bf-46fd4818fa43.png" width=250 height=250></a>

#### (6) Shanghai, China
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/6_1.png"><img src="https://user-images.githubusercontent.com/59532188/163596770-701d3605-e1a2-49a0-ab0d-bc605136033d.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/6_2.png"><img src="https://user-images.githubusercontent.com/59532188/163596776-0b0f855d-8f26-4607-98b2-6f8e50a7b5e0.png" width=250 height=250></a>
<a href="https://2gunsu.synology.me:8090/github_images/traffic-density-estimator/origin/6_3.png"><img src="https://user-images.githubusercontent.com/59532188/163596778-ae64e43a-ca14-4394-99ff-660d2d36ceba.png" width=250 height=250></a>




## References
[1] Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. **["Noise2Void - Learning Denoising from Single Noisy Images."](https://arxiv.org/abs/1811.10980)** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.  
[2] K. He, G. Gkioxari, P. Dollár and R. Girshick, **["Mask R-CNN,"](https://ieeexplore.ieee.org/document/8237584)** 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980-2988, doi: 10.1109/ICCV.2017.322.  

