If you have any questions, feel free to ask!  :)  
*To ensure the correctness of the experimental results, please run FCN in FuxiCTR==2.0.1.*

This model was formerly known as __DCNv3: Towards Next Generation Deep Cross Network for CTR Prediction__

A new version of the paper and code update will be available soon...

# FCN: Fusing Exponential and Linear Cross Network for Click-Through Rate Prediction
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-criteo)](https://paperswithcode.com/sota/click-through-rate-prediction-on-criteo?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-kdd12)](https://paperswithcode.com/sota/click-through-rate-prediction-on-kdd12?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-kkbox)](https://paperswithcode.com/sota/click-through-rate-prediction-on-kkbox?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-ipinyou)](https://paperswithcode.com/sota/click-through-rate-prediction-on-ipinyou?p=dcnv3-towards-next-generation-deep-cross)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dcnv3-towards-next-generation-deep-cross/click-through-rate-prediction-on-avazu)](https://paperswithcode.com/sota/click-through-rate-prediction-on-avazu?p=dcnv3-towards-next-generation-deep-cross)


![image](https://github.com/salmon1802/FCN/blob/master/fig/benchmark.png)


## Model Overview
<div align="center">
    <img src="https://github.com/salmon1802/FCN/blob/master/fig/architecture.png" alt="FCN" />
</div>

## Requirements
python>=3.6  
pytorch>=1.10  
fuxictr==2.0.1  
PyYAML  
pandas  
scikit-learn  
numpy  
h5py  
tqdm  

## Experiment results
![image](https://github.com/salmon1802/FCN/blob/master/fig/performance.png)

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
@article{li2025fcn,
  title={FCN: Fusing Exponential and Linear Cross Network for Click-Through Rate Prediction},
  author={Li, Honghao and Zhang, Yiwen and Zhang, Yi and Li, Hanwei and Sang, Lei and Zhu, Jieming},
  journal={arXiv preprint arXiv:2407.13349},
  year={2025}
}
```


