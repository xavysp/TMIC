 # TMIC
Tiny Model for Image Classification
## Introduction

Nearly all SOTA edge detection models based on deep learning incorporate an image classifier CNN architecture as their backbone.
However, only a few deep learning-based models have been specifically designed and implemented from scratch for edge detection,
such as DexiNed, TIN and [TEED](https://github.com/xavysp/TEED) to name a few, which is named Only Edge Detection Models (OEDM).
Designing a deep learning model specifically for edge detection offers the advantage of eliminating the need for transfer
learning, simplifying the training process, and reducing training time. However, these models have not been as widely adopted
as those adapted from image classification. Consequently, there is limited evidence of the
successful application of edge detection models to other computer vision tasks, such as image classification, despite the
theoretical compatibility and effectiveness of transitioning from image classification to edge detection.

The objective of this work is to examine the performance of edge detection CNN architecture in the context of image classification.

## Dataset

The re-implemented model from TEED, named TMIC, considered two popular datasets MNIST and FASHION-MNIST.


## Training and testing setup
```
To train and test TMIC you just need to execute:
tmic_classifier-SMISH-TEED.ipynb

```

## Aknowledgement
