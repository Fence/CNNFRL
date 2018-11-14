start=$(date +%s)
for((i=1; i<=3; i++))
do
    for dim in 8 16 32
    do
        CUDA_VISIBLE_DEVICES=9 python main.py \
        --gpu_fraction 0.2 \
        --train_mode grid \
        --metric acc \
        --agent_name full \
        --image_dim $dim \
        --autolen True \
        --hist_len 2 \
        --result_dir "test_"$i 
    done
done
end=$(date +%s)
echo -e "\n\nTotal time cost: $((end - $start))s\n\n"
