export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')


#TASKS="cola mnli qnli rte qqp sst2 stsb"
TASKS="cola"
declare -A PRETRAINED_ADAPTERS
PRETRAINED_ADAPTERS["cola"]="lingaccept/cola@ukp"
PRETRAINED_ADAPTERS["mnli"]="nli/multinli@ukp"
PRETRAINED_ADAPTERS["qnli"]="nli/qnli@ukp"
PRETRAINED_ADAPTERS["rte"]="nli/rte@ukp"
PRETRAINED_ADAPTERS["qqp"]="sts/qqp@ukp"
PRETRAINED_ADAPTERS["sst2"]="sentiment/sst-2@ukp"
PRETRAINED_ADAPTERS["stsb"]="sts/sts-b@ukp"

for MODEL_NAME in bert-base-uncased roberta-base
do
    for TASK_NAME in $TASKS
    do
	echo $TASK_NAME
	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results/$MODEL_NAME/$TASK_NAME \
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --load_adapter "lingaccept/cola@ukp" \
	       --do_eval

	python run_glue_adapterhub.py \
	       --model_name_or_path $MODEL_NAME \
	       --task_name $TASK_NAME \
	       --output_dir ./results/$MODEL_NAME/${TASK_NAME}_aave \
	       --overwrite_output_dir \
	       --adapter_config pfeiffer \
	       --train_adapter \
	       --dialect="aave" \
	       --load_adapter "lingaccept/cola@ukp" \
	       --do_eval
    done
done
