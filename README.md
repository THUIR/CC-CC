# CC-CC

This is our implementation for the paper:

*Shaoyun Shi, Min Zhang, Xinxing Yu, Yongfeng Zhang, Bin Hao, Yiqun Liu and Shaoping Ma. 2019. [Adaptive Feature Sampling for Recommendation with Missing Content Feature Values](https://dl.acm.org/doi/abs/10.1145/3357384.3357942) 
In CIKM'19.*

**Please cite our paper if you use our codes. Thanks!**

Author: Shaoyun Shi (shisy13 AT gmail.com)

```
@inproceedings{shi2019adaptive,
  title={Adaptive Feature Sampling for Recommendation with Missing Content Feature Values},
  author={Shaoyun Shi, Min Zhang, Xinxing Yu, Yongfeng Zhang, Bin Hao, Yiqun Liu and Shaoping Ma},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={1451--1460},
  year={2019},
  organization={ACM}
}
```



## Environments

Python 3.6.7

Packages: See in [requirements.txt](https://github.com/THUIR/CC-CC/blob/master/requirements.txt)

```
tensorflow==1.4.1
pandas==0.23.4
scipy==1.1.0
tqdm==4.28.1
numpy==1.15.4
scikit_learn==0.21.3
```



## Datasets

The processed datasets is in [./dataset](https://github.com/THUIR/CC-CC/blob/master/dataset).

- **ml-100k**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/). 
- **Zhihu**: The origin dataset can be found [here](https://www.biendata.com/competition/CCIR2018/). 


## Example to run the codes		

```
# CC-CC with Adaptive-Feature-Sampling
> cd CC-CC/src/
> python main.py --model_name FSCCCC --dataset ml100k-r-i30-u30-f10 --optimizer Adagrad --l2 1e-4 --cs_ratio 0.2 --fs_ratio 0.2 --fs_mode afs --lr 5e-2 --random_seed 2018 --gpu 1
```