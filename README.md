# UNet-Attention-removecloud  ![Static Badge](https://img.shields.io/badge/GPU-Nvidia-%2376B900?logo=nvidia) ![Static Badge](https://img.shields.io/badge/Anaconda-Env-%2344A833?logo=anaconda) ![Static Badge](https://img.shields.io/badge/IDE-Pycharm-%23000000?logo=pycharm) ![Static Badge](https://img.shields.io/badge/Operating_System-Linux-%23FCC624?logo=linux)


![1_rtTzjxWmhnNGKkebU_grYQ](https://user-images.githubusercontent.com/46363139/126754697-b4740533-fe1d-46be-9901-f420bc8ed8df.jpeg)

This is the code repository to train and record inference with patch-based UNet3D with attention mechanism models for image restoration under thin cloud conditions.😎


# Train
```
cd main

python train.py --config_file='../experiments/RSUNet/config.yaml'
```

the epoch number is set as 10000 epoches running on a Nvidia A100 GPU

# Test
```
python test.py --config_file='../experiments/RSUNet/config.yaml'
```

# Measure PSNR and SSIM
```
cd /cloud/remove_cloud_codes/lib/measures
python get_psnr_ssim.py --config_file='../experiments/RSUNet/config.yaml'
```

get_psnr_ssim.py is able to generate 2 new xls file after running ( {}.xls and {}_mean.xls respectively. Rename the xls file in line 136 using
```
xl_name ='unet3d_2023'
```
another xls file {}_mean.xls can be changed by name in line 125 

```
workbook.save('{}/{}_mean.xls'.format(img_path,xl_name))
```

# Dataset

A real-world multispectral dataset for thin cloud removal task, called LC8MS-TCR, from Landsat-8 OLI/TIRS satellite data. LC8MS-TCR is composed of 1420 paired real cloudy and cloud-free optical images used for training and testing cloud removal networks that comprises representative scenes with almost full multispectral information (9 bands from all 11 bands), containing different types of real-life thin clouds with their unique characteristic signature in 9 channels.

# Configuration Path

UNet as the instance:

## Model Path
```
cd /home/yub3/cloud/remove_cloud_codes/lib/model/generator/unet.py
```

## Config path
```
/home/yub3/cloud/remove_cloud_codes/experiments/RSUNet/config.yaml
# UNet PATH/ changing epoch number 
```





# Acknowledgement
The program is trained and tested on a A100 NVIDIA GPU on the server of Department of Computer Sceince in Aberystwyth University.
