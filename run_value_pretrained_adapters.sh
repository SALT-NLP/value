export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')


TASKS="cola mnli qnli rte qqp sst2 stsb"
for MODEL_NAME in bert-base-uncased roberta-base
do
    for TASK_NAME in $TASKS
    do
	ADAPTER_ADDRESS=AdapterHub/${MODEL_NAME}-pf-${TASK_NAME}
	echo $TASK_NAME
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results/$MODEL_NAME/$TASK_NAME \
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --load_adapter $ADAPTER_ADDRESS \
	       --do_eval

	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results/$MODEL_NAME/${TASK_NAME}_aave \
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --dialect="aave" \
	       --load_adapter $ADAPTER_ADDRESS \
	       --do_eval
    done
done
