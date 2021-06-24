############# Exp 1 #############
# Learn: real-value + softabs
# Infer: bipolar + sim_approx
#################################
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--log_dir ./log_ablation/log_exp1 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_train 10 \
--pool_val_train 5 \
--batch_size_train 5 \
--val_num_train 3 \
--pool_query_test 15 \
--batch_size_test 5 \
--train_episode 10000 \
--log_interval 100 \
--val_episode 250 \
--val_interval 500 \
--test_episode 1000 \
--learning_rate 0.0001 \
--quantization_learn No \
--quantization_infer 1 \
--rotation_update 1 \
--a32 1 \
--test_only 0 \
--pretrained_dir None \
--sim_cal cos_softabs \
--binary_id 1 \
--gpu 0,1 \

############# Exp 2 #############
# Learn: real-value + softabs
# Infer: binary + sim_approx
#################################
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--log_dir ./log_ablation/log_exp2 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_train 10 \
--pool_val_train 5 \
--batch_size_train 5 \
--val_num_train 3 \
--pool_query_test 15 \
--batch_size_test 5 \
--train_episode 10000 \
--log_interval 100 \
--val_episode 250 \
--val_interval 500 \
--test_episode 1000 \
--learning_rate 0.0001 \
--quantization_learn No \
--quantization_infer 1 \
--rotation_update 1 \
--a32 1 \
--test_only 0 \
--pretrained_dir None \
--sim_cal cos_softabs \
--binary_id 2 \
--gpu 0,1 \

############# Exp 3 #############
# Learn: real-value + softmax
# Infer: bipolar + sim_approx
#################################
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--log_dir ./log_ablation/log_exp3 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_train 10 \
--pool_val_train 5 \
--batch_size_train 5 \
--val_num_train 3 \
--pool_query_test 15 \
--batch_size_test 5 \
--train_episode 10000 \
--log_interval 100 \
--val_episode 250 \
--val_interval 500 \
--test_episode 1000 \
--learning_rate 0.0001 \
--quantization_learn No \
--quantization_infer 1 \
--rotation_update 1 \
--a32 1 \
--test_only 0 \
--pretrained_dir None \
--sim_cal cos_softmax \
--binary_id 1 \
--gpu 0,1 \

############# Exp 4 #############
# Learn: real-value + softmax
# Infer: binary + sim_approx
#################################
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--log_dir ./log_ablation/log_exp4 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_train 10 \
--pool_val_train 5 \
--batch_size_train 5 \
--val_num_train 3 \
--pool_query_test 15 \
--batch_size_test 5 \
--train_episode 10000 \
--log_interval 100 \
--val_episode 250 \
--val_interval 500 \
--test_episode 1000 \
--learning_rate 0.0001 \
--quantization_learn No \
--quantization_infer 1 \
--rotation_update 1 \
--a32 1 \
--test_only 0 \
--pretrained_dir None \
--sim_cal cos_softmax \
--binary_id 2 \
--gpu 0,1 \


############# Exp 5 #############
# Learn: XNOR + softabs
# Infer: bipolar + sim_approx
#################################
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--log_dir ./log_ablation/log_exp5 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_train 10 \
--pool_val_train 5 \
--batch_size_train 5 \
--val_num_train 3 \
--pool_query_test 15 \
--batch_size_test 5 \
--train_episode 10000 \
--log_interval 100 \
--val_episode 250 \
--val_interval 500 \
--test_episode 1000 \
--learning_rate 0.0001 \
--quantization_learn XNOR \
--quantization_infer 1 \
--rotation_update 1 \
--a32 1 \
--test_only 0 \
--pretrained_dir None \
--sim_cal cos_softabs \
--binary_id 1 \
--gpu 0,1 \


############# Exp 6 #############
# Learn: XNOR + softmax
# Infer: bipolar + sim_approx
#################################
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--log_dir ./log_ablation/log_exp6 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_train 10 \
--pool_val_train 5 \
--batch_size_train 5 \
--val_num_train 3 \
--pool_query_test 15 \
--batch_size_test 5 \
--train_episode 10000 \
--log_interval 100 \
--val_episode 250 \
--val_interval 500 \
--test_episode 1000 \
--learning_rate 0.0001 \
--quantization_learn XNOR \
--quantization_infer 1 \
--rotation_update 1 \
--a32 1 \
--test_only 0 \
--pretrained_dir None \
--sim_cal cos_softmax \
--binary_id 1 \
--gpu 0,1 \


############# Exp 7 #############
# Learn: RBNN + softabs
# Infer: bipolar + sim_approx
#################################
CUDA_VISIBLE_DEVICES=0,1 python main_RBNN.py \
--log_dir ./log_ablation/log_exp7 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_train 10 \
--pool_val_train 5 \
--batch_size_train 5 \
--val_num_train 3 \
--pool_query_test 15 \
--batch_size_test 5 \
--train_episode 10000 \
--log_interval 100 \
--val_episode 250 \
--val_interval 500 \
--test_episode 1000 \
--learning_rate 0.0001 \
--quantization_learn RBNN \
--quantization_infer 1 \
--rotation_update 1 \
--a32 1 \
--test_only 0 \
--pretrained_dir None \
--sim_cal cos_softabs \
--binary_id 1 \
--gpu 0,1 \

############# Exp 8 #############
# Learn: RBNN + softmax
# Infer: bipolar + sim_approx
#################################
CUDA_VISIBLE_DEVICES=0,1 python main_RBNN.py \
--log_dir ./log_ablation/log_exp8 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_train 10 \
--pool_val_train 5 \
--batch_size_train 5 \
--val_num_train 3 \
--pool_query_test 15 \
--batch_size_test 5 \
--train_episode 10000 \
--log_interval 100 \
--val_episode 250 \
--val_interval 500 \
--test_episode 1000 \
--learning_rate 0.0001 \
--quantization_learn RBNN \
--quantization_infer 1 \
--rotation_update 1 \
--a32 1 \
--test_only 0 \
--pretrained_dir None \
--sim_cal cos_softmax \
--binary_id 1 \
--gpu 0,1 







