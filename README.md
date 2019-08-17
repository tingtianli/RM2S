# RM2S
### Codes for the paper "Single-Image Reflection Removal via a Two-Stage Background Recovery Process"
![cover](cover.PNG)

### Codes are implemented on pytorch>=0.2.1

### How to use
#### Training:
- Training process uses images from [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) for reflection image simulation.
- The network for the first stage process: 
```
python train_1st_stage.py --imgs_dir your_voc12_path/VOC2012/JPEGImages/
```
- The network for the 2nd stage process: 
```
python train_2nd_stage.py --imgs_dir your_voc12_path/VOC2012/JPEGImages/
```
Trained models will be saved in the folder ./model_para
#### Testing:

- A single image demo:
```
python one_img_demo.py --img_dir one_image_path(e.g. imgs/1.png) --net_ini_pkl path_to_trained_models/Net_1st_stage.pkl --netG_img_pkl path_to_trained_models/Net_2nd_stage.pkl
```
- Evaluate on [SIR2 benchmark](http://rose1.ntu.edu.sg/Datasets/sir2Benchmark.asp):
```
python benchmark_imgs_process.py --bechmark_dir SIR2_dataset_path --net_ini_pkl path_to_trained_models/Net_1st_stage.pkl --netG_img_pkl path_to_trained_models/Net_2nd_stage.pkl
```
- [Pre-trained models](https://connectpolyu-my.sharepoint.com/:f:/g/personal/15900416r_connect_polyu_hk/EpjHAPgDdfxIhQSb4BkYOWABhllxLZG5BgflQG-CXfHR7A?e=0Hzosa) are provided <br/>
-  (K-means threshold coeficents 0.5 and 0.5 can show better perceptual performance. But 0.2 and 0.8 can give better background fidelity)
## Citation
```
@Article{li2019rm2s,
  author    = {Li, Tingtian and Lun, Daniel P.K.},
  title     = {Single-Image Reflection Removal via a Two-Stage Background Recovery Process},
  journal   = {IEEE Signal Processing Letters},
  year      = {2019},
}
```
