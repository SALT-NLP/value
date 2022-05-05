# VALUE: The VernAcular Language Understanding Evaluation benchmark 

This repository contains source code necessary to build the VALUE datasets.

Feel free to contact [Caleb Ziems](https://calebziems.com/) with any questions.

[[Paper]](https://arxiv.org/pdf/2204.03031.pdf) | [[Data Use Agreement]](https://forms.gle/9EpDtvfebXXhvfaV8)

## Setup
### Prerequisites: 
1. Create a virtual environment
```bash
conda create --name value python=3.7
conda activate value
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Install spaCy English pipeline and nltk wordnet
```python
python -m spacy download en_core_web_sm
python 
>>> import nltk
>>> nltk.download('wordnet')
>>> quit()
```

4. Download the datasets from the [GLUE benchmark](https://gluebenchmark.com/) where each task is a subdirectory of `data/GLUE`

```
python download_glue_data.py --data_dir "data/GLUE" --tasks all
```

### Build VALUE variants

**Note**: This can take a while to run. To create only a single task, replace the --all tag with the task-specific tag (e.g. --MNLI) in each of the following commands.

Move to src (```cd src```) and complete the following:

1. Build VALUE base variant with column for HTML tagging (to be used in MTurk validation)
```python
python -m src.build_value --all --VALUE 'data/VALUE' --lexical_mapping 'resources/sae_aave_mapping_dict.pkl' --morphosyntax --html --dialect aave
```

2. Build VALUE_no_morpho variant
```python
python -m src.build_value --all --VALUE 'data/VALUE' --lexical_mapping 'resources/sae_aave_mapping_dict.pkl' --html --dialect aave
```

3. Build VALUE_no_lex variant
```python
python -m src.build_value --all --VALUE 'data/VALUE' --morphosyntax --html --dialect aave
```

4. Build the VALUE_style_transfer variant by cloning the [style-transfer-paraphrase](https://github.com/martiansideofthemoon/style-transfer-paraphrase) repo and running the following for each task dataframe
```python
from style_paraphrase.inference_utils import GPT2Generator
import pandas as pd

paraphraser = GPT2Generator('pretrained_style_transfer/models/paraphraser_gpt2_large/', upper_length="same_5")
paraphraser.modify_p(top_p=0.6)
sae_to_aave = GPT2Generator('pretrained_style_transfer/models/cds_models/aae', upper_length="same_5")
sae_to_aave.modify_p(top_p=0.6)

df = pd.read_csv('path/to/specific/task') # FILL THIS PATH IN
converted = []
batch_size = 32
for i in range(int(len(df)/batch_size)+1):
    sub_df = df.iloc[batch_size*(i):batch_size*(i+1)].copy()
    for col in df.columns:
        if (('sentence' in col) or ('question' in col)) and ('parse' not in col):
            sub_df = sub_df[[type(c)==str for c in sub_df[col].values]].copy()
            consider = sub_df[col].values
            para, prob = paraphraser.generate_batch(consider)
            aave, prob = sae_to_aave.generate_batch(para)
            sub_df[col+'-glue'] = consider
            sub_df[col] = aave
            converted.append(sub_df)
            
converted_df = pd.concat(converted)
converted_df.to_csv('path/to/transformed/task', sep='\t') # FILL THIS PATH IN
```