nohup bash -c \
'for db in wikihow cooking win2k
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --agent_mode eas \
    --gpu_rate 0.20 \
    --save_weight 1 \
    --actionDB $db \
    --context_num 1 \
    --is_test 1 \
    --result_dir "test_exclussive"
done' \
> cn1_eas.out 2>&1 &
