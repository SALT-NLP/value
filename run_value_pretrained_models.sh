export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

declare -A MODELS=(
    ["roberta-base-cola"]="JeremiahZ/roberta-base-cola" \
    ["roberta-base-mnli"]="JeremiahZ/roberta-base-mnli" \
    ["roberta-base-qnli"]="JeremiahZ/roberta-base-qnli" \
    ["roberta-base-rte"]="JeremiahZ/roberta-base-rte" \
    ["roberta-base-qqp"]="JeremiahZ/roberta-base-qqp" \
    ["roberta-base-sst2"]="JeremiahZ/roberta-base-sst2" \
    ["roberta-base-stsb"]="JeremiahZ/roberta-base-stsb" \
    ["bert-base-uncased-cola"]="gchhablani/bert-base-cased-finetuned-cola" \
    ["bert-base-uncased-mnli"]="gchhablani/bert-base-cased-finetuned-mnli" \
    ["bert-base-uncased-qnli"]="gchhablani/bert-base-cased-finetuned-qnli" \
    ["bert-base-uncased-rte"]="gchhablani/bert-base-cased-finetuned-rte" \
    ["bert-base-uncased-qqp"]="gchhablani/bert-base-cased-finetuned-qqp" \
    ["bert-base-uncased-sst2"]="gchhablani/bert-base-cased-finetuned-sst2" \
    ["bert-base-uncased-stsb"]="gchhablani/bert-base-cased-finetuned-stsb" \
    )

TASKS="cola mnli qnli rte qqp sst2 stsb"
for MODEL_NAME in bert-base-uncased roberta-base
do
    for TASK_NAME in $TASKS
    do
	MODEL=${MODELS[$MODEL_NAME-$TASK_NAME]}
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
