# Sym-Parameterized Dynamic Inference for Mixed-Domain Image Translation

This repository is official Pytorch implementations of SGN in the following paper.

Simyung Chang, SeongUk Park, John Yang and Nojun Kwak, "Sym-Parameterized Dynamic Inference for Mixed-Domain Image Translation", ICCV2019,  [arXiv](https://arxiv.org/abs/1811.12362)

This code is built on [Pytorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN) and tested on Pytorch 0.4.1.



<img src='figs/main.png' align="center" width=800>

Dynamic Inference for Mixed-Domain Image Translation.



<img src='figs/SP.png' align="center" width=400>

The concept of sym-parameter.



<img src='figs/SGN.png' align="center" width=800>

Overall Structure of SGN for Three Different Losses.

#### Video translation result of SGN : [SGN-Video](https://youtu.be/i1XsGEUpfrs)



## Train

### Prepare training data

```
./download_dataset <dataset_name>
```

<dataset_name> for the models in the paper.

Model 1: vangogh2photo

Model 2: ukiyoe2photo

Model 3 : summer2winter_yosemite, monet2photo



### Train 

```
python train.py --dataroot <data_dir> --style_image <image_file>
```

To train the Model 1

```
python train.py --dataroot 'datasets/vangogh2photo/' --style_image 'images/style-images/udnie.jpg'
```



## Test

The pre-trained models can be downloaded from [Google Drive](https://drive.google.com/open?id=137-61DU9u05lF7bOfwn1ptW-KJw23rGW).

Copy the models in pretrained/



### Test

```
python test.py --dataroot <data_dir> --generator <trained_file>
```

Test Model1 with pre-trained check point.

```
python test.py --dataroot 'datasets/vangogh2photo' --generator 'pretrained/Model1.pth'
```



## Citation
```
@article{chang2018image,
  title={Image Translation to Mixed-Domain using Sym-Parameterized Generative Network},
  author={Chang, Simyung and Park, SeongUk and Yang, John and Kwak, Nojun},
  journal={arXiv preprint arXiv:1811.12362},
  year={2018}
}
```



## Acknowledgments
Our code is built on [Pytorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN).
