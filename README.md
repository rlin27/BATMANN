# BATMANN: A Binarized MANN for Few-Shot Learning

This is a PyTorch implementation of the MANN described in [Robust high-dimensional memory-augmented
neural networks](https://doi.org/10.1038/s41467-021-22364-0). In addition, we provide a binary version MANN, whose controller is trained as a binary neural network (BNN) in an end-to-end way, and the feature vectors stored in the key memory are binarized as well.

## Running Codes
In this code, you can run the MANN on omniglot dataset, obtaining a full-precision or a binarized mature Controller. We provide scripts in ``` ./scripts ``` and the checkpoints in ``` ./log ``` , which lead to easy running of our codes. 

### Installation
This code is tested on both PyTorch 1.2 (cuda 11.2).
```
git clone https://github.com/RuiLin0212/BATMANN.git
pip install -r requirements.txt
```
### Learn and Evaluate a Controller
We provide the scripts to learn a full-precision and a binary controller in ```./scripts``` , respectively. You can modify the ```--data_dir```, and simply run ```sh ./scripts/full_precision.sh``` / ```sh ./scripts/binary.sh```. Then you can get mature controllers for 5-way 1-shot, 20-way 5-shot, and 100-way 5-shot problems. Or you can modify more arguments according to your needs and specific problems. For omniglot dataset, it is worth nothing that the following requirements should be satiesfied：
+ num_shot + pool_query_train + pool_val_train <= 20
+ pool_query_train >= batch_size_train
+ pool_val_train >= val_num_train
+ num_shot + pool_query_test <= 20
+ pool_query_test >= batch_size_test 

```
python main.py \
--log_dir [The path to store the training log file.] \
--data_dir [The absolute path to the dataset.] \
--input_channel [Number of input channel of the samples.] \
--feature_dim [The dimension of the feature vectors.] \
--class_num [m in the m-way n-shot problem.] \
--num_shot [n in the m-way n-shot problem.] \
--pool_query_train [Number of samples in each class to sample the queries in the training phase.] \
--pool_val_train [Number of samples in each class to sample the validation samples in the training phase.] \
--batch_size_train [Number of queries in each class in the training phase.] \
--val_num_train [Number of validation samples in each class in the training phase] \
--pool_query_test [Number of samples in each class to sample the queries in the inference phase.] \
--batch_size_test [Number of queries in each class in the inference phase.] \
--train_episode [Number of episode during training.] \
--log_interval [Number of intervals to log the training process.] \
--val_episode [Number of episode during validation.] \
--val_interval [Number of intrvals to do validation.] \
--test_episode [Number of episode during inference.] \
--learning_rate [Initial learning rate for the optimizer.] \
--quantization [Do binarized training or not.] \
--test_only [Use pretrained parameters to do inference directly or not.] \
--pretrained_dir [The path to the pretrained parameters.] \
--gpu [ID of the GPU to use]
```

### Inference Only
For the ease of reproducibility, we also provide the checkpoints for mature Controller. To do inference directly, you can modify ```--data_dir``` and ```---pretrained_dir```, then run ```sh ./scripts/check_pretrained.sh```. Or you can modify more arguments:

```
python main.py \
--log_dir [The path to store the training log file.] \
--data_dir [The absolute path to the dataset.] \
--input_channel [Number of input channel of the samples.] \
--feature_dim [The dimension of the feature vectors.] \
--class_num [m in the m-way n-shot problem.] \
--num_shot [n in the m-way n-shot problem.] \
--pool_query_test [Number of samples in each class to sample the queries in the inference phase.] \
--batch_size_test [Number of queries in each class in the inference phase.] \
--test_episode [Number of episode during inference.] \
--quantization [Do binarized training or not.] \
--test_only [Use pretrained parameters to do inference directly or not.] \
--pretrained_dir [The path to the pretrained parameters.] \
--gpu [ID of the GPU to use]
```

## Codes Structure
The two figures below illustrate the relations among different functions, which also help understand how the MANN work.

### Data Loading
![](./fig/dataloader.png)

### Learn & Inference
![](./fig/main.png)

## Experimental Results

| Problem | Full-Precision | Binary |
|:---:|:---:|:----:|
| 5-way 1-shot | ~ | 94.40% |
| 20-way 5-shot | 97.64% | 95.11% |
| 100-way 5-shot |  | ~ |

More results will be available soon.


## Acknowledgement
This code is ispired by [LearningToCompare_FSL](https://github.com/floodsung/LearningToCompare_FSL). We thanks for this open-source implementations.


