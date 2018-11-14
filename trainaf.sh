nohup bash -c \
'for db in wikihow cooking win2k
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --agent_mode af \
    --gpu_rate 0.20 \
    --save_weight 1 \
    --actionDB $db \
    --context_num 1 \
    --is_test 0 \
    --result_dir "fixed_words"
done' \
> cn1_af.out 2>&1 &
