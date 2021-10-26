# A Biased Graph Neural Network Sampler with Near-Optimal Regret
The implementation and expremental code of Thanos graph neural network sampler [See paper](https://arxiv.org/abs/2103.01089).

## Datasets
- Chameleon
- Squirrel
- CoraFull
- ogbn-arxiv
- ogbn-product

## Model 
- GCN
- GAT

## Dependencies
- tensorflow==1.13.1
- torch==1.7.0
- ogb==1.2.3
- dgl==0.5.3
- cython==0.29.21
- tensorboardX==2.1

## Prepare
```bash
pip3 install -r requirement.txt
python3 cython_sampler/setup.py build_ext -i
mkdir log
```

## How to run
```bash
python3 train.py --logdir <log_folder> --dataset <dataset> --sampler <sampler> --model <model> --sample_interval <DeltaT> --neighbor_limit <k> --gamma 0.4 --eta 0.01 --epochs 200 --hidden1 256 --batchsize 256 --learning_rate 0.001 --dropout 0.1 
```
+ \<log_folder\>: The folder to save TB file and running data under the root folder './log'.
+ dataset: CoraFull / ogbn-arxiv / ogbn-products
+ sampler: BanditSampler / Thanos 
+ model: GCN / GAT
+ sample_interval: $ \Delta_{T} $, the interval for reinitialization of sampler. Set -1 for BanditSampler to turn off the reinitialization. Setting 0 will reinitialize sampler every epoch.
+ neighbor_limit: k, the number of neighbor to be sampled. 
+ gamma: $ \gamma $
+ eta: $ \eta $
+ hidden1: the dimension of hidden embedding
+ epochs: the number of epoches 

### Command Example for BanditSampler
```bash
python3 train.py --logdir log_corafull --dataset CoraFull --sampler BanditSampler  --model GCN --logger_name 1 --sample_interval -1 --neighbor_limit 3 --gamma 0.4 --etas 0.01 --epochs 300 --hidden1 256 --batchsize 256 --learning_rate 0.001 --dropout 0.1 --noadd_selfloop 
```
```bash
python3 train.py --logdir log_corafull --dataset CoraFull --sampler BanditSampler  --model GAT --logger_name 1 --sample_interval -1 --neighbor_limit 3 --gamma 0.4 --etas 0.01 --epochs 300 --hidden1 256 --batchsize 256 --learning_rate 0.001 --dropout 0.1 --add_selfloop 
```

### Command Example for Thanos and Thanos.M
```bash
python3 train.py --logdir log_corafull --dataset CoraFull --sampler Thanos  --model GCN --logger_name 1 --sample_interval 2000 --neighbor_limit 3 --gamma 0.2 --etas 0.1 --epochs 300 --hidden1 256 --batchsize 256 --learning_rate 0.001 --dropout 0.1 --noadd_selfloop 
```
```bash
python3 train.py --logdir log_corafull --dataset CoraFull --sampler Thanos  --model GAT --logger_name 1 --sample_interval 2000 --neighbor_limit 3 --gamma 0.2 --etas 0.1 --epochs 300 --hidden1 256 --batchsize 256 --learning_rate 0.001 --dropout 0.1 --add_selfloop 
```

### Check Results
```bash
tensorboard --logdir log/<log_folder>
```

## Reference 
```bash
@article{zhang2021biased,
  title={A biased graph neural network sampler with near-optimal regret},
  author={Zhang, Qingru and Wipf, David and Gan, Quan and Song, Le},
  journal={arXiv preprint arXiv:2103.01089},
  year={2021}
}
```
