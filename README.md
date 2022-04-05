# Empowering Product Search with Collaborative Filtering

We implement a product search model equipped with collaborative filtering. 

In addition, this repo also implements several typical product search baselines, including QL, UQL, LSE, HEM, AEM and Transearch.

We do not add ALSTP in this repo due its strict setting. 

The aim of this repo is to build a universal product search environment with same pre-processing steps, deep learning toolkit, and hyper-parameters.
Please contribute freely when you find any parts conflict!

Code contributor: Yangyang Guo (60%) and [Xiangkun Yin](https://github.com/PTYin) (40%).

## Evaluation Protocols
This repo supports two kinds of evaluation protocols: all test set (default one) and 1 vs 99. See the preprocessing in [neg_candidate.py](preprocess/neg_candidate.py).

Note that there are very few lines requiring commenting and uncommenting for 1 vs 99 testing.

## Prerequisites
    * python==3.7
    * pytorch==1.10.1 

## Dataset
In present (by Apr. 2022), the exclusive datasets for product search is [Amazon](http://jmcauley.ucsd.edu/data/amazon/index_2014.html).

Unfortunately, we do not have any released real-world datasets involving user-submitted queries and their corresponding bought product. 
In view of this, we follow the LSE paper to extract the queries ourselves. 

Download the Amazon dataset and put the subset in the path according to the ```data_path``` in [param.py](./src/params.py).

## Pre-processing

Perform pre-processing:

```
    PYTHONPATH=$PYTHONPATH:./src python preprocess/process.py --dataset Office_Products 
    ```

For QL and UQL, run an additional script to make some temp files:

```
    PYTHONPATH=$PYTHONPATH:./src python src/QL/preprocess.py --dataset Office_Products 
    ```

For TransearchText, run the pre-trained Doc2Vec model:

```
    PYTHONPATH=$PYTHONPATH:./src python preprocess/doc2vec.py --dataset Office_Products 
    ```

## Model Training and Testing
```
    PYTHONPATH=$PYTHONPATH:./src python src/anymodel --dataset Office_Products
    ```

## Citation
If you want to use this code, please cite the papers below:
```
@inproceedings{transearch,
  author    = {Yangyang Guo and
               Zhiyong Cheng and
               Liqiang Nie and
               Xin{-}Shun Xu and
               Mohan S. Kankanhalli},
  title     = {Multi-modal Preference Modeling for Product Search},
  booktitle = {ACM Multimedia Conference},
  pages     = {1865--1873},
  publisher = {ACM},
  year      = {2018}
}
@article{alstp,
  author    = {Yangyang Guo and
               Zhiyong Cheng and
               Liqiang Nie and
               Yinglong Wang and
               Jun Ma and
               Mohan S. Kankanhalli},
  title     = {Attentive Long Short-Term Preference Modeling for Personalized Product
               Search},
  journal   = {ACM Transactions on Information Systems},
  volume    = {37},
  number    = {2},
  pages     = {19:1--19:27},
  year      = {2019}
}
```
