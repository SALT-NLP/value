export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

HF_ORG="WillHeld"

for MODEL_NAME in roberta-base bert-base-uncased
do
    MODEL=$HF_ORG/${MODEL_NAME}-coqa
    echo $MODEL
    python coqa_exp/run_coqa_adapterhub.py \
	   --model_name_or_path $MODEL_NAME \
	   --dataset_name coqa \
	   --metric_for_best_model="eval_f1" \
	   --output_dir ./results_train_combined/$MODEL_NAME/coqa \
	   --max_seq_length 384 \
	   --load_dialect_from_hub \
	   --version_2_with_negative \
	   --per_device_train_batch_size 16 \
	   --learning_rate 2e-5 \
	   --weight_decay 0.1 \
	   --warmup_ratio 0.06 \
	   --num_train_epochs 10 \
	   --overwrite_output_dir \
	   --do_train \
	   --do_eval \
	   --evaluation_strategy "steps" \
	   --eval_steps 500 \
	   --save_total_limit 1 \
	   --load_best_model_at_end True \
	   --hub_model_id $MODEL \
	   --push_to_hub True \
	   --use_auth_token
done
