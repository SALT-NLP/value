export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

HF_ORG="SALT-NLP"

#TASKS="cola mnli qnli rte qqp sst2 stsb"
TASKS="cola qnli mnli"
#TASKS="rte sst2 stsb qqp"
#bert-base-uncased roberta-base
for MODEL_NAME in roberta-base
do
    for TASK_NAME in $TASKS
    do
	MODEL=$HF_ORG/${MODEL_NAME}-${TASK_NAME}-combined-value
	echo $MODEL
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results_train_combined/$MODEL_NAME/$TASK_NAME \
	       --max_seq_length 128 \
	       --per_device_train_batch_size 16 \
	       --learning_rate 2e-5 \
	       --weight_decay 0.1 \
	       --warmup_ratio 0.06 \
	       --num_train_epochs 10 \
	       --overwrite_output_dir \
	       --do_train \
	       --do_eval \
	       --push_to_hub True \
	       --combine_sae True \
	       --dialect "aave" \
	       --hub_model_id $MODEL \
	       --evaluation_strategy "steps" \
	       --eval_steps 500 \
	       --save_total_limit 1 \
	       --load_best_model_at_end True \
	       --hub_private_repo \
	       --use_auth_token
	
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL \
	       --task_name $TASK_NAME \
	       --output_dir ./results_full/$MODEL/$TASK_NAME \
	       --overwrite_output_dir \
	       --do_eval \
	       --hub_model_id=$MODEL \
	       --hub_private_repo \
	       --use_auth_token

	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL \
	       --task_name $TASK_NAME \
	       --output_dir ./results_full/$MODEL/${TASK_NAME}_aave \
	       --overwrite_output_dir \
	       --dialect="aave" \
	       --do_eval \
	       --hub_model_id=$MODEL \
	       --hub_private_repo \
	       --use_auth_token
    done
done
