# ERA S9 Assignment

This repository contains the code for the S9 assignment of the ERA course. The aim of the assignment is to write a new network that

    1. has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
    2. total RF must be more than 44
    3. one of the layers must use Depthwise Separable Convolution
    4. one of the layers must use Dilated Convolution
    5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    6. use albumentation library and apply:
        a. horizontal flip
        b. shiftScaleRotate
        c. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
    7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
    8. make sure you're following code-modularity (else 0 for full assignment)
    9. upload to Github

## Pre-requisites

The code is written in Python 3.10.11. It is recommended to use a virtual environment to run the code to avoid dependency issues. Try to use Google Colab or Kaggle to run the code as they provide free access to GPUs. If you are running the code on your local machine, make sure you install the virtual environment before running the code.

### Installing the Virtual Environment

It is advised to install Anaconda to manage the virtual environment. Anaconda can be downloaded from [here](https://www.anaconda.com/products/individual). Once installed, the virtual environment can be created using the following command:

```bash
conda create -n era python=3.10.11
```

### Activating the Virtual Environment

The virtual environment needs to be activated before running the code. This can be done using the following command:

```bash
conda activate era
```

## Installation

1. Clone the repository using the following command:

    ```bash
    git clone https://github.com/ChikkaUdayaSai/ERA-S9-Assignment
    ```

2. Navigate to the repository directory:

    ```bash
    cd ERA-S9-Assignment
    ```

3. Install the dependencies using the following commnad:

    ```bash
    pip install -r requirements.txt
    ```

Note: If you are using Google Colab or Kaggle, you can skip the above step as the dependencies are already installed in the environment. But it is advised to check the versions of the dependencies before running the code.

The code uses PyTorch and Torchvision for fetching the MNIST dataset and training the model. An additional dependency, Matplotlib, is used for plotting the training and validation losses. Finally, the Torchsummary package is used to visualize the model architecture.

We are now ready to run the code with the following versions of the dependencies:

- **PyTorch: 2.0.1**
- **Torchvision: 0.15.2**
- **Matplotlib: 3.7.1**
- **Torchsummary: 1.5.1**
- **Albumentations: 1.2.1**


## Solution

The entire assignment was modularized into different python modules and the runner code is present in the Session9.ipynb notebook. The following modules were created:

1. model.py - Contains the model architecture
2. utils.py - Contains the helper functions for training and testing the model
3. dataset.py - Contains the custom dataset class for loading the CIFAR10 dataset

The model architecture is as follows:
    
```python

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 16, 32, 32]           2,304
              ReLU-6           [-1, 16, 32, 32]               0
       BatchNorm2d-7           [-1, 16, 32, 32]              32
           Dropout-8           [-1, 16, 32, 32]               0
            Conv2d-9           [-1, 48, 32, 32]             480
           Conv2d-10           [-1, 16, 32, 32]             784
        Depthwise-11           [-1, 16, 32, 32]               0
             ReLU-12           [-1, 16, 32, 32]               0
      BatchNorm2d-13           [-1, 16, 32, 32]              32
          Dropout-14           [-1, 16, 32, 32]               0
           Conv2d-15           [-1, 32, 32, 32]           4,608
             ReLU-16           [-1, 32, 32, 32]               0
      BatchNorm2d-17           [-1, 32, 32, 32]              64
          Dropout-18           [-1, 32, 32, 32]               0
           Conv2d-19           [-1, 32, 32, 32]           9,216
             ReLU-20           [-1, 32, 32, 32]               0
      BatchNorm2d-21           [-1, 32, 32, 32]              64
          Dropout-22           [-1, 32, 32, 32]               0
           Conv2d-23           [-1, 32, 28, 28]           9,216
             ReLU-24           [-1, 32, 28, 28]               0
      BatchNorm2d-25           [-1, 32, 28, 28]              64
          Dropout-26           [-1, 32, 28, 28]               0
           Conv2d-27           [-1, 64, 28, 28]          18,432
             ReLU-28           [-1, 64, 28, 28]               0
      BatchNorm2d-29           [-1, 64, 28, 28]             128
          Dropout-30           [-1, 64, 28, 28]               0
           Conv2d-31           [-1, 64, 28, 28]          36,864
             ReLU-32           [-1, 64, 28, 28]               0
      BatchNorm2d-33           [-1, 64, 28, 28]             128
          Dropout-34           [-1, 64, 28, 28]               0
           Conv2d-35           [-1, 64, 24, 24]          36,864
             ReLU-36           [-1, 64, 24, 24]               0
      BatchNorm2d-37           [-1, 64, 24, 24]             128
          Dropout-38           [-1, 64, 24, 24]               0
           Conv2d-39           [-1, 64, 24, 24]          36,864
             ReLU-40           [-1, 64, 24, 24]               0
      BatchNorm2d-41           [-1, 64, 24, 24]             128
          Dropout-42           [-1, 64, 24, 24]               0
           Conv2d-43           [-1, 48, 24, 24]          27,648
             ReLU-44           [-1, 48, 24, 24]               0
      BatchNorm2d-45           [-1, 48, 24, 24]              96
          Dropout-46           [-1, 48, 24, 24]               0
           Conv2d-47           [-1, 32, 24, 24]          13,824
             ReLU-48           [-1, 32, 24, 24]               0
      BatchNorm2d-49           [-1, 32, 24, 24]              64
          Dropout-50           [-1, 32, 24, 24]               0
AdaptiveAvgPool2d-51             [-1, 32, 1, 1]               0
           Conv2d-52             [-1, 16, 1, 1]             512
             ReLU-53             [-1, 16, 1, 1]               0
      BatchNorm2d-54             [-1, 16, 1, 1]              32
          Dropout-55             [-1, 16, 1, 1]               0
           Conv2d-56             [-1, 10, 1, 1]             160
================================================================
Total params: 199,200
Trainable params: 199,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.49
Params size (MB): 0.76
Estimated Total Size (MB): 12.26
----------------------------------------------------------------
```

The model was trained for 29 epochs with a batch size of 128 and a learning rate of 0.01. The model achieved a test accuracy of 85.07% at the end of 29 epochs.

The losses and accuracies for the training and test sets are as follows:

![Losses and Accuracies](.\assets\losses_and_accuracies.png)

The misclassified images are as follows:

![Misclassified Images](.\assets\misclassified_images.png)