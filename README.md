If you have any questions, feel free to ask!  :)  
*To ensure the correctness of the experimental results, please run DCNv3 in FuxiCTR==2.0.1 or the latest version.*

This model was formerly known as DCNv3: Towards Next Generation Deep Cross Network for CTR Prediction

# FCN: Fusing Exponential and Linear Cross Network for Click-Through Rate Prediction
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-kdd12)](https://paperswithcode.com/sota/click-through-rate-prediction-on-kdd12?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-kkbox)](https://paperswithcode.com/sota/click-through-rate-prediction-on-kkbox?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-ipinyou)](https://paperswithcode.com/sota/click-through-rate-prediction-on-ipinyou?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-avazu)](https://paperswithcode.com/sota/click-through-rate-prediction-on-avazu?p=dcnv3-towards-next-generation-deep-cross)


![image](https://github.com/user-attachments/assets/0a40f76b-0784-47d4-9e06-985d85540460)


## Model Overview
<div align="center">
    <img src="https://github.com/user-attachments/assets/6f0479ce-edb2-4ad1-92a0-7ec9aeeb5f2d" alt="DCNv3" />
</div>

![image](https://github.com/user-attachments/assets/11d6850f-5eec-4865-b5a7-4abbc2088508)



## Requirements
python>=3.6  
pytorch>=1.10  
fuxictr==2.0.1 or lastest  
PyYAML  
pandas  
scikit-learn  
numpy  
h5py  
tqdm  

## Experiment results
![image](https://github.com/user-attachments/assets/4be97744-3f3f-4840-96f1-b1e6c3435d51)
![image](https://github.com/user-attachments/assets/7128efd9-ea98-4d1d-8773-ce529ce4377f)



## Datasets
Get the datasets from https://github.com/reczoo/Datasets

## Hyperparameter settings and logs
Get the result from ./checkpoints

## Acknowledgement
This implementation is based on FuxiCTR and BARS. Thanks for their sharing and contribution.  
BARS: https://github.com/openbenchmark  
FuxiCTR: https://github.com/xue-pai/FuxiCTR

## Citation
If you find our code helpful for your research, please cite the following paper:

```bibtex
@article{li2024dcnv3,
  title={DCNv3: Towards Next Generation Deep Cross Network for CTR Prediction},
  author={Li, Honghao and Zhang, Yiwen and Zhang, Yi and Li, Hanwei and Sang, Lei and Zhu, Jieming},
  journal={arXiv preprint arXiv:2407.13349},
  year={2024}
}
```


