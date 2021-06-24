# check the full-precision one
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--log_dir ./log_check_pretrained/full_precision_100_5 \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 100 \
--num_shot 5 \
--pool_query_test 15 \
--batch_size_test 4 \
--test_episode 1000 \
--quantization 0 \
--test_only 1 \
--pretrained_dir ./log_full_precision_100_5/model_best.pth \
--gpu 0,1,2,3 \

## check the binary one
#python main.py \
#--log_dir ./log_check_pretrained/binary_100_5 \
#--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
#--input_channel 1 \
#--feature_dim 512 \
#--class_num 100 \
#--num_shot 5 \
#--pool_query_test 15 \
#--batch_size_test 4 \
#--test_episode 1000 \
#--quantization 1 \
#--test_only 1 \
#--pretrained_dir ./log_binary_100_5/model_best.pth \
#--gpu 2,3