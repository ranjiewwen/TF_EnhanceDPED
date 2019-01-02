## TF_EnhanceDPED project

- Tensorflow implement of image enhancement base on dped.
- First reimplementation of ICCV 2017 paper "[DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks](https://arxiv.org/pdf/1704.02470.pdf)" .
- Second will modify the generate network and loss to make the image enhancement result better. 

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
│   ├── config.py
│   ├── DPED_model_20190101-19:25
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

- Use this command to trian the models `python tools/train.py --<parameter>`.

### Coming soon optimization

- Training is to read a part of the data into the memory for training. Each training can re-import a part of the data and load it into the memory.
- So each training load the data is very slow, next step will optimization. 
- Will use Moving average loss.

### Thanks

- 2019/01/02 init the repository.
- Thanks offical code [DPED](https://github.com/aiff22/DPED) !



