# Welcome to Awesome On-device AI
![Awesome](https://awesome.re/badge.svg) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ysyisyourbrother/awesome-mlsys-mobile/pulls)

A curated list of awesome projects and papers for AI on ___Mobile/IoT/Edge___ devices. Everything is continuously updating. ___Welcome contributions! Feel free to add not just papers, but also new sections, or adjust the existing content.___

## Contents

- [Papers/Tutorial](#papers/tutorial)
  - [Training on Devices](#1-Training-on-devices)
  - [Inference on Devices](#2-inference-on-devices)
  - [Mobile AI Applications](#3-Mobile-AI-Applications)
  - [Survey and Tutorial](#4-survey-and-tutorial)
- [Open Source Projects](#open-source-projects)
- [Contribute](#Contribute)

## Papers/Tutorial

### 1. Training on Devices

#### 1.1 Memory Efficient Training

- [ICML'22] POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging. by Patil et al. [[paper](https://proceedings.mlr.press/v162/patil22b/patil22b.pdf)]
- [NeruIPS'22] On-Device Training Under 256KB Memory. by Ji Lin, Song Han et al. [[paper](https://arxiv.org/pdf/2206.15472.pdf)]
- [MobiSys'22] Melon: breaking the memory wall for resource-efficient on-device machine learning. by Qipeng Wang et al. [[paper](https://xumengwei.github.io/files/MobiSys22-Melo.pdf)]
- [MobiSys'22] Sage: Memory-efficient DNN Training on Mobile Devices. by In Gim et al. 2022 [[paper](https://dl.acm.org/doi/abs/10.1145/3498361.3539765)]

#### 1.2 Training Acceleration

- [NeurIPS'25] LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades.
- [SenSys'24] PieBridge: Fast and Parameter-Efficient On-Device Training via Proxy Networks. By Wangsong Yin.
- [MobiCom'22] Mandheling: Mixed-Precision On-Device DNN Training with DSP Offloading. by Daliang Xu et al. [[paper](http://arxiv.org/abs/2206.07509)]

#### 1.3 Training on Mobile Cluster

- [TPDS'26] Resource-Efficient Personal Large Language Models Fine-Tuning with Collaborative Edge Computing. by Shengyuan Ye et al. [[paper](https://ieeexplore.ieee.org/abstract/document/11355763/)]
- [ASPLOS'24] SoCFlow: Efficient and Scalable DNN Training on SoC-Clustered Edge Servers. by Daliang Xu et al.
- [ATC'24] More is Different: Prototyping and Analyzing a New Form of Edge Server with Massive Mobile SoCs. by Li Zhang et al.
- [ATC'24] High-density Mobile Cloud Gaming on Edge SoC Clusters. by Li Zhang et al.
- [ATC'24] FwdLLM: Efficient Federated Finetuning of Large Language Models with Perturbed Inferences. by Mengwei Xu et al.
- [WWW'24] Towards Energy-efficient Federated Learning via INT8-based Training on Mobile DSPs. by Jinliang Yuan et al.
- [ICPP'24] Pluto and Charon: A Time and Memory Efficient Collaborative Edge AI Framework for Personal LLMs Fine-Tuning. by Bei Ouyang et al.  [[paper](https://dl.acm.org/doi/pdf/10.1145/3673038.3673043)]
- [MobiCom'24] Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices. by Shengyuan Ye et al. [[paper](https://dl.acm.org/doi/pdf/10.1145/3636534.3649363)]
- [MobiCom'23] Federated Few-shot Learning for Mobile NLP. by Dongqi Cai et al.
- [MobiCom'23] Efficient Federated Learning for Modern NLP. by Dongqi Cai et al.
- [ICPP'22] Eco-FL: Adaptive Federated Learning with Efficient Edge Collaborative Pipeline Training. by Shengyuan Ye et al. [[paper](https://ssl.linklings.net/conferences/icpp/icpp2022_program/views/includes/files/pap117s3-file1.pdf)] [[code](https://github.com/ysyisyourbrother/Federated-Learning-Research.git)]
- [SEC'21] EDDL: A Distributed Deep Learning System for Resource-limited Edge Computing Environment. by Pengzhan Hao et al. [[paper](https://buzhangy.github.io/publication/eddl-sec21.pdf)]
- [MobiSys'21 Workshop] Towards Ubiquitous Learning: A First Measurement of On-Device Training Performance. by Dongqi Chai, Mengwei Xu et al. [[paper](https://dl.acm.org/doi/10.1145/3469116.3470009)]

### 2. Inference on Devices

#### 2.1 Collaborative Inference
- [TMC'25] Resource-Efficient Collaborative Edge Transformer Inference with Hybrid Model Parallelism. by Shengyuan Ye et al. [[paper](https://ieeexplore.ieee.org/abstract/document/11017462?casa_token=hFum-boeJ00AAAAA:u8q5IdqmZSizapYu4-zRQ7dDUdAUkzujHmSqnaVhsE_1Q2r74RWoa1WOLe0LLP5wi95ZXLKdpA)]
- [INFOCOM'25] Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices. by Shengyuan Ye et al. [[paper](https://arxiv.org/pdf/2504.08242?)]
- [ICCAD'25] Mitigating Resource Contention for Responsive On-device Machine Learning Inferences. by Minsung Kim et al. [[paper](https://ieeexplore.ieee.org/abstract/document/11240707)]
- [INFOCOM'24] Galaxy: A Resource-Efficient Collaborative Edge AI System for In-situ Transformer Inference. by Shengyuan Ye et al. [[paper](https://arxiv.org/pdf/2405.17245?)]
- [ICSOC'23] Niagara: Scheduling DNN Inference Services on Heterogeneous Edge Processors. by Daliang Xu et al.
- [MobiSys'23] NN-Stretch: Automatic Neural Network Branching for Parallel Inference on Heterogeneous Multi-Processors. by USTC & Microsoft. [[paper](https://www.microsoft.com/en-us/research/uploads/prod/2023/05/stretch_mobisys23-6462ea7a63d9e.pdf)]
- [MobiSys'22] CoDL: efficient CPU-GPU co-execution for deep learning inference on mobile devices. by Fucheng Jia et al. [[paper](https://chrisplus.me/assets/pdf/mobisys22-CoDL.pdf)]
- [InfoCom'22] Distributed Inference with Deep Learning Models across Heterogeneous Edge Devices. by Chenghao hu et al. [[paper](https://iqua.ece.toronto.edu/papers/chenghao-infocom22.pdf)]
- [TON'20] Coedge: Cooperative dnn inference with adaptive workload partitioning over heterogeneous edge devices. by Liekang Zeng et al. [[paper](https://ieeexplore.ieee.org/abstract/document/9296560)]
- [ICCD'20] A distributed in-situ CNN inference system for IoT applications. by Jiangsu Du et al. [[paper](https://ieeexplore.ieee.org/abstract/document/9283504/)]
- [TPDS'20] Model Parallelism Optimization for Distributed Inference via Decoupled CNN Structure. by Jiangsu Du et al. [[paper](https://ieeexplore.ieee.org/document/9275375/)]
- [EuroSys'19] μLayer: Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization. by Youngsok Kim et al. [[paper](https://dl.acm.org/doi/abs/10.1145/3302424.3303950)]
- [TCAD'18] DeepThings: Distributed Adaptive Deep Learning Inference on Resource-Constrained IoT Edge Clusters. by zhuoran Zhao et al. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8493499)]
- [DATE'17] Modnn: Local distributed mobile computing system for deep neural network. by Jiachen Mao et al. [[paper](https://ieeexplore.ieee.org/abstract/document/7927211/)]


#### 2.2 Inference Acceleration

- [ASPLOS'25] Fast On-device LLM Inference with NPUs. by Daliang Xu et al.
- [TMC'24] EdgeLLM: Fast On-Device LLM Inference With Speculative Decoding. by Daliang Xu et al.
- [MobiCom'24] FlexNN: Efficient and Adaptive DNN Inference on Memory-Constrained Edge Devices. by Xiangyu Li et al.
- [MobiSys'24] Empowering In-Browser Deep Learning Inference on Edge Through Just-In-Time Kernel Optimization.
- [MobiSys'23] Boosting DNN Cold Inference on Edge Devices. by Rongjie Yi et al.
- [MobiSys'23] ConvReLU++: Reference-based Lossless Acceleration of Conv-ReLU Operations on Mobile CPU. by Shanghai Jiao Tong University [[paper](https://yuanchun-li.github.io/static/files/MobiSys23_ConvReLU++.pdf)]
- [MobiSys'22] Band: coordinated multi-DNN inference on heterogeneous mobile processors. by Seoul National University et al. [[paper](https://dl.acm.org/doi/10.1145/3498361.3538948)]
- [MobiSys'21] nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices. by Li Lyna Zhang et al. [[paper](https://dl.acm.org/doi/abs/10.1145/3458864.3467882)]


### 3. Mobile AI Applications

#### 3.1 Mobile GUI Agent
- [ICLR'26] UIShift: Enhancing VLM-based GUI Agents through Self-supervised Reinforcement Learning.
- [MobiSys'25] AutoDroid-V2: Boosting SLM-based GUI Agents via Code Generation. by Hao Wen et al.
- [UIST'24] LlamaTouch: A Faithful and Scalable Testbed for Mobile UI Task Automation.
- [Arxiv'24] MobileViews: A Large-Scale Mobile GUI Dataset.
- [MobiCom'24] AutoDroid: LLM-powered Task Automation in Android. by Hao Wen et al.


#### 3.2 Mobile Visual and Multimodal Tasks
- [INFOCOM'26] Venus: An Efficient Edge Memory-and-Retrieval System for VLM-based Online Video Understanding. by Shengyuan Ye et al. [[paper](https://arxiv.org/pdf/2512.07344)]
- [NC'25] Ubiquitous Memory Augmentation via Mobile Multimodal Embedding System. by Dongqi Cai et al.
- [NSDI'25] Region-based Content Enhancement for Efficient Video Analytics at the Edge.
- [Arxiv'24] InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions.
- [Arxiv'23] MobileVLM : A Fast, Strong and Open Vision Language Assistant for Mobile Devices.
- [MobiSys'22] Approximate Query Service on Autonomous IoT Cameras. [[paper](https://xumengwei.github.io/files/MobiSys20-Elf.pdf)]
- [MobiCom'18] DeepCache: Principled Cache for Mobile Deep Vision. by Mengwei Xu et al. [[paper](https://arxiv.org/abs/1712.01670)]

#### 3.3 Mobile NLP/Speech

- [NeurIPS'24] SILENCE: Protecting Privacy in Offloaded Speech Understanding on Wimpy Devices. by Dongqi Cai et al.
- [Ubicomp'18] DeepType: On-Device Deep Learning for Input Personalization Service with Minimal Privacy Concern. by Mengwei Xu et al. [[paper](https://dl.acm.org/doi/10.1145/3287075)]
- [Arxiv 2018] Federated learning for mobile keyboard prediction. by Google [[paper](https://arxiv.org/abs/1811.03604)]

### 4. Survey and Tutorial

- [Arxiv'24] Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security. by Yuanchun Li.
- [CSUR'24] A Survey of Resource-efficient LLM and Multimodal Foundation Models. by Mengwei Xu et al. [[paper](https://xumengwei.github.io/files/CSUR24-efficientllm.pdf)]
- [CVPR'23 Tutorial] Efficient Neural Networks: From Algorithm Design to Practical Mobile Deployments. by Snap Research [[paper](https://snap-research.github.io/efficient-nn-tutorial/)]


## Open Source Projects

### 1. DL Framework on Mobile

- mllm: Fast Multimodal LLM on Mobile Devices. by BUPT Team. [[code](https://github.com/UbiquitousLearning/mllm)]
- Tensorflow Lite: Deploy machine learning models on mobile and edge devices. by Google. [[code](https://www.tensorflow.org/lite)]
- TensorflowJS: A WebGL accelerated JavaScript library for training and deploying ML models. by Google. [[code](https://github.com/tensorflow/tfjs)]
- MNN: A Universal and Efficient Inference Engine. by Alibaba. [[code](https://github.com/alibaba/MNN)]
- TensorRT: A C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators. by Nvidia. [[code](https://github.com/NVIDIA/TensorRT)]
- TVM: Open deep learning compiler stack for cpu, gpu and specialized accelerators. by Tianqi Chen et al. [[code](https://github.com/apache/tvm)]
- MACE: a deep learning inference framework optimized for mobile heterogeneous computing platforms. by XiaoMi. [[code](https://github.com/XiaoMi/mace)]
- NCNN: a high-performance neural network inference framework optimized for the mobile platform. by Tencent. [[code](https://github.com/Tencent/ncnn)]

### 2. Audio
- FluidAudio: Local audio AI SDK for Apple platforms with ASR, speaker diarization, VAD, and TTS. Optimized for Apple Neural Engine. by FluidInference. [[code](https://github.com/FluidInference/FluidAudio)]

## Contribute

All contributions to this repository are welcome. Open an [issue](https://github.com/ysyisyourbrother/awesome-mlsys-mobile/issues) or send a [pull request](https://github.com/ysyisyourbrother/awesome-mlsys-mobile/pulls).
