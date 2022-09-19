export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

TASKS="cola mnli qnli rte qqp sst2 stsb"
for MODEL_NAME in bert-base-cased roberta-base
do
    for TASK_NAME in $TASKS
    do
	MODEL=WillHeld/${MODEL_NAME}-${TASK_NAME}
	echo $MODEL
	echo $TASK_NAME
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL \
	       --task_name $TASK_NAME \
	       --output_dir ./results_full/$MODEL_NAME/$TASK_NAME \
	       --overwrite_output_dir \
	       --do_eval

	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL \
	       --task_name $TASK_NAME \
	       --output_dir ./results_full/$MODEL_NAME/${TASK_NAME}_aave \
	       --overwrite_output_dir \
	       --dialect="aave" \
	       --do_eval
    done
done
