##################################
## Learn: real-value + softabs
## Infer: bipolar + sim_approx
##################################
CUDA_VISIBLE_DEVICES=2,3 python main.py \
--log_dir [Log Path] \
--data_dir [Data Path] \
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
--gpu 0,1