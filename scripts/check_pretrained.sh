# check the full-precision one
python main.py \
--log_dir ./log_check_pretrained/full_precision \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_test 15 \
--batch_size_test 4 \
--test_episode 1000 \
--quantization 0 \
--test_only 1 \
--pretrained_dir ./log_full_precision/model_best.pth \
--gpu 2,3 \

# check the binary one
python main.py \
--log_dir ./log_check_pretrained/binary \
--data_dir /mnt/nfsdisk/jier/4_MANN/mann_hdv/data \
--input_channel 1 \
--feature_dim 512 \
--class_num 20 \
--num_shot 5 \
--pool_query_test 15 \
--batch_size_test 4 \
--test_episode 1000 \
--quantization 1 \
--test_only 1 \
--pretrained_dir ./log_binary/model_best.pth \
--gpu 2,3