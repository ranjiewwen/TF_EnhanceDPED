## TF_EnhanceDPED project

- Tensorflow implement of image enhancement base on dped.
- First reimplementation of ICCV 2017 paper "[DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks](https://arxiv.org/pdf/1704.02470.pdf)".

- Seconde reimplementation [Range Scaling Global U-Net for Perceptual Image Enhancement on Mobile Devices](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Huang_Range_Scaling_Global_U-Net_for_Perceptual_Image_Enhancement_on_Mobile_ECCVW_2018_paper.pdf). Join the PRIM2018 Challenge on Perceptual Image Enhancement on Smartphones (Track B: Image Enhancement) http://ai-benchmark.com/challenge.html#challenge . which is champion plan. it modify the generate network and loss to make the image enhancement result better. 

- Third i will add image quality assessment model to guide image enhancement. use iqa model extract generate image loss(Subjective representation loss and Subjective score loss),which will make the generated image more in line with the subjective perception of the human eye. There i ues [end-to-end optimized deep neural network (MEON)](https://ece.uwaterloo.ca/~zduanmu/tip2018biqa/) this models.

### Prerequisites

- Python + scipy, numpy packages
- TensorFlow (>=1.0.1) + CUDA CuDNN
- Nvidia GPU

### File tree

```bash
├── data
│   ├── dped -> /home/***/datasets/dped/
│   ├── __init__.py
│   ├── load_dataset.py
│   └── pretrain_models
├── demo
├── experiments
│   ├── config
│   └── logs
├── loss
│   ├── GAN_loss.py
│   ├── __init__.py
│   ├── other_loss.py
│   └── vgg_loss.py
├── metrics
│   ├── __init__.py
│   ├── psnr.py
│   └── ssim.py
├── net
│   ├── __init__.py
│   └── resnet.py
├── README.md
├── tools
│   ├── inference.py
│   └── train.py
└── utils
    ├── __init__.py
    ├── logger.py
    └── utils.py

```

- Refactored file structure is easy to expand.
- Net, data, loss, metrics are folders for network, data import, loss definition and metrics respectively.
- Demo, utils folder is used to provide some visualization, helper function folder.
- `experiments/config` will provide the parameter parse, you can modify the hyperparameter from this file.
- The DPED_* folder is automatically generated for each experiment, which stores the checkpoint file and the visually enhanced patch image.
- The `experiments/logs/` folder saves the event files in the training and validation phases, which can be visualized by commands `tensorboard --lodir logs`.


### First steps congfig

- Download the pre-trained VGG-19 model and put it into pretrain_models/ folder.
    - It should use the weight provided by the author, when use the matconvnet offical weight it will get error.
- Download DPED dataset (patches for CNN training) and extract it into dped/ folder.
    - This folder should contain three subolders: sony/, iphone/ and blackberry/

- Modify the config configuration file in the experiments.

### Tain

- Use this command to trian the models `python tools/train_baseline.py --<parameter>`.

### Test

- [aiff22/ai-challenge](https://github.com/aiff22/ai-challenge)

### Coming soon optimization

- Training is to read a part of the data into the memory for training. Each training can re-import a part of the data and load it into the memory.
- So each training load the data is very slow, next step will optimization. 
- Will use Moving average loss.

### some useful paper
- Range Scaling Global U-Net for Perceptual Image Enhancement on Mobile Devices,Jie Huang, Pengfei Zhu, Mingrui Geng, Jiewen Ran, Xingguang Zhou, Chen Xing, Pengfei Wan, Xiangyang Ji.
- TALEBI H, MILANFAR P. NIMA: Neural Image Assessment[J]. IEEE Transactions on Image Processing, 2018,27(8):3998-4011.
- TALEBI H, MILANFAR P. Learned perceptual image enhancement, 2018[C]. IEEE, 2018
- CHOI J, KIM J, CHEON M, et al. Deep Learning-based Image Super-Resolution Considering Quantitative and Perceptual Quality[J]. 2018.
- HUA W, XIA Y. Low-Light Image Enhancement Based on Joint Generative Adversarial Network and Image Quality Assessment, 2018[C]. IEEE, 2018.
- End-to-End Blind Image Quality Assessment Using Deep Neural Networks .Kede Ma, Wentao Liu, Kai Zhang, Zhengfang Duanmu, Zhou Wang, and Wangmeng Zuo.IEEE Transactions on Image Processing (TIP), vol. 27, no. 3, pp. 1202-1213, Mar. 2018.

### Thanks

- 2019/01/02 init the repository.
- Thanks offical code [DPED](https://github.com/aiff22/DPED)!
- [tf-perceptual-eusr](https://github.com/idearibosome/tf-perceptual-eusr)
