# Welcome to Awesome On-device AI
![Awesome](https://awesome.re/badge.svg) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ysyisyourbrother/awesome-mlsys-mobile/pulls)

A curated list of awesome projects and papers for AI on **Mobile/IoT/Edge** devices. Everything is continuously updating. Welcome contribution!

## Contents

- [Papers](#papers)
  - [Learning on Devices](#1-learning-on-devices)
  - [Inference on Devices](#2-inference-on-devices)
  - [Models for Mobile](#3-model-for-mobile)
- [Open Source Projects](#open-source-projects)
- [Contribute](#Contribute)

## Papers

### 1. Learning on Devices

#### a. Memory Efficient Learning

- [POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging](https://proceedings.mlr.press/v162/patil22b/patil22b.pdf) by Patil et al., ICML 2022
- [Melon: breaking the memory wall for resource-efficient on-device machine learning](https://xumengwei.github.io/files/MobiSys22-Melo.pdf) by Qipeng Wang et al., MobiSys 2022
- [Sage: Memory-efficient DNN Training on Mobile Devices](https://dl.acm.org/doi/abs/10.1145/3498361.3539765) by In Gim et al., MobiSys 2022

#### b. Learning Acceleration

- [Mandheling: Mixed-Precision On-Device DNN Training with DSP Offloading](http://arxiv.org/abs/2206.07509) by Daliang Xu et al., MobiCom 2022

#### c. Learning on Mobile Cluster

- [Eco-FL: Adaptive Federated Learning with Efficient Edge Collaborative Pipeline Training](https://ssl.linklings.net/conferences/icpp/icpp2022_program/views/includes/files/pap117s3-file1.pdf) by Shengyuan Ye et al., ICPP 2022
- [EDDL: A Distributed Deep Learning System for Resource-limited Edge Computing Environment](https://buzhangy.github.io/publication/eddl-sec21.pdf) by Pengzhan Hao et al., SEC 2021



### 2. Inference on Devices

#### a. Collaborative Inference

- [CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices](https://chrisplus.me/assets/pdf/mobisys22-CoDL.pdf) by Fucheng Jia et al., MobiSys 2022
- [Distributed Inference with Deep Learning Models across Heterogeneous Edge Devices](https://iqua.ece.toronto.edu/papers/chenghao-infocom22.pdf) by Chenghao hu et al., InfoCom 2022
- [Coedge: Cooperative dnn inference with adaptive workload partitioning over heterogeneous edge devices](https://ieeexplore.ieee.org/abstract/document/9296560/) by Liekang Zeng et al., TON 2020
- [μLayer: Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization](https://dl.acm.org/doi/abs/10.1145/3302424.3303950) by Youngsok Kim et al., EuroSys 2019
- [DeepThings: Distributed Adaptive Deep Learning Inference on Resource-Constrained IoT Edge Clusters](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8493499) by zhuoran Zhao et al., TCAD 2018
- [Modnn: Local distributed mobile computing system for deep neural network](https://ieeexplore.ieee.org/abstract/document/7927211/) by Jiachen Mao et al., DATE 2017

#### b. Latency Prediction for Inference

- [nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices](https://dl.acm.org/doi/abs/10.1145/3458864.3467882) by Li Lyna Zhang et al., MobiSys 2021



### 3. Model for Mobile

#### a. Lightweight Model

- [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/pdf/2004.02984.pdf) by Zhiqing Sun et al., ACL 2020
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf) by Mingxing Tan et al., ICML 2019
- [Shufflenet: An extremely efficient convolutional neural network for mobile devices](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html) by Xiangyu Zhang et al., CVPR 2018
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf) by Mark Sandler et al., CVPR 2018



## Open Source Projects

#### 1. DL Framework on Mobile

- [Tensorflow Lite: Deploy machine learning models on mobile and edge devices](https://www.tensorflow.org/lite) by Google.
- [TensorflowJS: A WebGL accelerated JavaScript library for training and deploying ML models](https://github.com/tensorflow/tfjs) by Google.

#### 2. Inference Deployment

- [TensorRT: A C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators](https://github.com/NVIDIA/TensorRT) by Nvidia.
- [MACE: a deep learning inference framework optimized for mobile heterogeneous computing platforms](https://github.com/XiaoMi/mace) by XiaoMi.
- [MNN: A Universal and Efficient Inference Engine](https://github.com/alibaba/MNN) by Alibaba.

#### 3. Compilation

- [TVM: Open deep learning compiler stack for cpu, gpu and specialized accelerators](https://github.com/apache/tvm) by Tianqi Chen et al.




## Contribute

All contributions to this repository are welcome. Open an [issue](https://github.com/ysyisyourbrother/awesome-mlsys-mobile/issues) or send a [pull request](https://github.com/ysyisyourbrother/awesome-mlsys-mobile/pulls).
