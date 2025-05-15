# HAFNet: Hierarchical Attention Fusion Network for Infrared Small Target Detection

**We have submitted the paper for review and will make the code available after publication.**

## Network
HAFNet is an improved U-Net architecture from two aspects: feature extraction backbone and skip connections, aiming to enhance detection ability and generalization performance.
## Recommended Environment
 - [ ] python  3.11.7
 - [ ] pytorch 2.2.1
 - [ ] torchvision 0.17.1

## Datasets
**Our project has the following structure:**
  ```
  ├───dataset/
  │    ├── NUAA-SIRST
  │    │    ├── image
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── train_NUAA-SIRST.txt
  │    │    │── train_NUAA-SIRST.txt
  │    ├── IRSTD-1K
  │    │    ├── image
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── train_IRSTD-1K.txt
  │    │    ├── train_IRSTD-1K.txt
  │    ├── ...  
  ```
<be>

## Results
#### Visualization results
![outline](image/Visualization.jpg)
#### Grad-CAM results
![outline](image/Grad-CAM.jpg)
#### 3D Visualization Results
![outline](image/3D.jpg)

#### Quantitative Results on NUAA-SIRST, IRSTD-1K and NUDT-SIRST

| Dataset         | IoU (x10(-2)) | nIoU (x10(-2)) | Pd(x10(-2))| Fa (x10(-6))|  F (x10(-2))| 
| ------------- |:-------------:|:-------------:|:-----:|:-----:|:-----:|
| NUAA-SIRST    | 79.19  | 81.00  |  97.72 | 14.06 | 88.39 |
| IRSTD-1K      | 67.95  | 69.23  |  94.61 | 10.57 | 80.91 |
| NUDT-SIRST    | 96.28  | 96.11  |  99.26 | 1.79  | 98.10 |
| [[weights]](https://drive.google.com/drive/folders/1MhZbgE0pjOmyEuOaIs11tvivFw2ViMCh?usp=drive_link)|

*This code is highly borrowed from [SCTransNet](https://github.com/xdFai/SCTransNet). Thanks to Shuai Yuan.

*The overall repository style is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.








