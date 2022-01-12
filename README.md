# Multilingual Pre-training with Universal Dependency Learning

## Introduction

The pre-trained language model (PrLM) demonstrates domination in downstream natural language processing tasks, in which multilingual PrLM takes advantage of language universality to alleviate the issue of limited resources for low-resource languages. Despite its successes, the performance of multilingual PrLM is still unsatisfactory, when multilingual PrLMs only focus on plain text and ignore obvious universal linguistic structure clues. Existing PrLMs have shown that monolingual linguistic structure knowledge may bring about better performance. Thus we propose a novel multilingual PrLM that supports both explicit universal dependency parsing and implicit language modeling. Syntax in terms of universal dependency parse serves as not only pre-training objective but also learned representation in our model, which brings unprecedented PrLM interpretability and convenience in downstream task use. Our model outperforms two popular multilingual PrLM, multilingual-BERT and XLM-R, on cross-lingual natural language understanding (NLU) benchmarks and linguistic structure parsing datasets, demonstrating the effectiveness and stronger cross-lingual modeling capabilities of our approach.
![The model architecture of UD-PrLM](https://github.com/KAI-SHU/UDPrLM/blob/main/figure/qqq.PNG)

## Pre-training UD-PrLM

### Pre-process Data

You need to download the corpus for training Mask Language Modeling (MLM) corresponding to each baseline model, such as Wikipedias for m-BERT and CommonCrawl for XLM-R<sub>base</sub> and XLM-R<sub>large</sub>, and you need to download Universal Dependencies Treebanks. Run the following code to generate UD training data from the original Universal Dependencies Treebanks (already downloaded to path `UD_TREEBANKS_PATH`) and save it to `UD_TREEBANKS_PATH`.

```sh
$ git clone https://github.com/KAI-SHU/UDPrLM && cd UDPrLM
$ python transformers/examples/udlm/preprocess_training_data/join_ud_conllu.py
    --input_dir UD_TREEBANKS_PATH \
    --output_dir UD_JOIN_PATH
```

Taking UD-BERT as an example, run the following code to generate MLM training data from the original corpus (already downloaded to path `MULTILINGUAL_WIKIPEDIAS_DATA_PATH`) and save it to `UDBERT_MLM_JOIN_PATH`.

```sh
$ python transformers/examples/udlm/preprocess_training_data/join_mlm.py
    --input_dir MULTILINGUAL_WIKIPEDIAS_DATA_PATH \
    --output_dir UDBERT_MLM_JOIN_PATH \
    --suffix .text
```

### Training

```sh
$ cd transformers
$ pip install --editable .

# for UD-BERT
$ python -u examples/udlm/run_ud_lm.py \
    --output_dir UDBERT_SAVE_PATH \
    --dataset_cache_dir CACHE_PATH \
    --config_name BertConfig \
    --tokenizer_name BertTokenizer \
    --data_dir UD_JOIN_PATH \
    --train_file train.conll \
    --eval_file dev_tiny.conll \
    --convert_strategy 2 \
    --special_label APP \
    --mlm_train_file UDBERT_MLM_JOIN_PATH/mlm_train.txt \
    --mlm_validation_file UDBERT_MLM_JOIN_PATH/mlm_dev.txt \
    --max_seq_length 512 \
    --parsing_max_seq_length 256 \
    --warmup_ratio 0.1 \
    --learning_rate 0.00003 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps 600000 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy 0 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --max_grad_norm 5.0 \
    --per_device_eval_batch_size 6 \
    --per_device_parsing_eval_batch_size 4 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --labels UD_JOIN_PATH/labels.txt \
    --model_type bert \
    --parsing_training_prob 0.8 \
    --use_ud_repr 

# for UD-XLM-R
$ python -u examples/udlm/run_ud_lm.py \
    --output_dir UDXLMR_SAVE_PATH \
    --dataset_cache_dir CACHE_PATH \
    --config_name XLMRobertaConfig \
    --tokenizer_name XLMRobertaTokenizer \
    --data_dir UD_JOIN_PATH \
    --train_file train.conll \
    --eval_file dev_tiny.conll \
    --convert_strategy 2 \
    --special_label APP \
    --mlm_train_file UDXLMR_MLM_JOIN_PATH/mlm_train.txt \
    --mlm_validation_file UDXLMR_MLM_JOIN_PATH/mlm_dev.txt \
    --max_seq_length 512 \
    --parsing_max_seq_length 256 \
    --warmup_ratio 0.1 \
    --learning_rate 0.00003 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps 600000 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --save_strategy 0 \
    --save_steps 2000 \
    --save_total_limit 100 \
    --max_grad_norm 5.0 \
    --per_device_eval_batch_size 4 \
    --per_device_parsing_eval_batch_size 4 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2} \
    --do_train \
    --do_eval \
    --labels UD_JOIN_PATH/labels.txt \
    --model_type roberta \
    --parsing_training_prob 0.8 \
    --use_ud_repr \
    --fp16
```

### Evaluate Syntactic Parsing Ability 

Take UD-BERT as an example:

```sh
$ python -u examples/udlm/run_ud_lm.py \
    --model_name_or_path UDBERT_SAVE_PATH \
    --data_dir UD_TREEBANKS_PATH/UD_Bulgarian-BTB \
    --eval_file bg_btb-ud-test.conllu \
    --do_eval \
    --parsing_only \
    --model_type bert \
    --special_label APP \
    --labels UD_JOIN_PATH/labels.txt \
    --convert_strategy 2 \
    --output_dir EVAL_RESULT_PATH \
    --parsing_max_seq_length 100 \
    --per_device_eval_batch_size 4 \
    --per_device_parsing_eval_batch_size 4
```

## Call the Library

At the end of the pre-training process, the UD-PrLM is saved in the output directory you specified, and an UD-PrLM can be loaded and used by calling the `udlm` library.

```sh
$ from transformers import AutoTokenizer
$ from udlm import UDBertConfig, UDBertModel

$ udbert = UDBertModel.from_pretrained('UDBERT_SAVE_PATH', \
	config=UDBertConfig.from_pretrained('UDBERT_SAVE_PATH', output_hidden_states=True))

$ from transformers import AutoTokenizer
$ from udlm import UDXLMRobertaConfig, UDXLMRobertaModel

$ udxlmr = UDXLMRobertaModel.from_pretrained('UDXLMR_SAVE_PATH', \
	config=UDXLMRobertaConfig.from_pretrained('UDXLMR_SAVE_PATH', output_hidden_states=True))
```

## Example: use `udlm` for parsing

### Introduce Supar

We follow **Supar** ([Zhang et al.](https://github.com/yzhangcs/parser)), a Python package that contains many state-of-the-art syntactic/semantic parsers, as well as highly parallel implementations of several effective structured prediction algorithms. We add support for multilingual universal syntactic dependency pretrained model (UDPrLM) including UD-BERT, UD-XLM-R<sub>base</sub> and UD-XLM-R<sub>large</sub> using `udlm` library.

### Installation

```sh
$ cd parser
$ python setup.py install
```

### Syntactic Dependency Parsing

Below are examples of training `biaffine` dependency parsers

```sh
# m-BERT (baseline)
$ python -u -m supar.cmds.biaffine_dep train -b -d 0 -c biaffine-dep-lang -p model  \
    -f tag char bert  \
    --train ud/lang/train.conllx  \
    --dev ud/lang/dev.conllx  \
    --test ud/lang/test.conllx  \
    --n_bert_layers 4 \
    --bert_type bert \
    --bert bert-base-multilingual-case  \
    --tree --punct --freeze
# UD-BERT
$ python -u -m supar.cmds.biaffine_dep train -b -d 0 -c biaffine-dep-lang -p model  \
    -f tag char bert  \
    --train ud/lang/train.conllx  \
    --dev ud/lang/dev.conllx  \
    --test ud/lang/test.conllx  \
    --n_bert_layers 5 \
    --bert_type udbert \
    --bert the-path-of-udbert/  \
    --tree --punct --freeze  \
    --fusion_ud
```

The option -c controls where to load predefined configs, you can either specify a local file path or the same short name as a pretrained model. --fusion_ud ensures that syntactic information in PrLM is used to strengthen the final representation. --punct makes the evaluate and test scores include punctuation, so we use this parameter on UD dataset but not on PTB and CTB. When using XLM-R<sub>base</sub> or XLM-R<sub>large</sub> as the pre-training language model for baseline, you need to set --bert_type to `xlm-roberta` and --bert to `xlm-roberta-base` or `xlm-roberta-large`. When using UD-XLM-R<sub>base</sub> or UD-XLM-R<sub>large</sub> as the pre-training language model, you need to set --bert_type to `udxlmr` and --bert to the path of corresponding UDPrLM.

To evaluate:

```sh
python -u -m supar.cmds.biaffine_dep evaluate -d 0 -p biaffine-dep-lang --data data/lang-test.conllx --tree --proj --punct  \
```
--tree and --proj ensures to output well-formed and projective trees respectively.

### Syntactic Constituency Parsing

Below are examples of training `crf` constituency parsers
```sh
# m-BERT (baseline)
$ python -u -m supar.cmds.crf_con train -b -d 0 -c crf_con-lang -p model  \
    -f char bert  \
    --train spmrl/lang/train.pid  \
    --dev spmrl/lang/dev.pid  \
    --test spmrl/lang/test.pid  \
    --n_bert_layers 4 \
    --bert_type bert \
    --bert bert-base-multilingual-case  \
    --mbr --freeze
# UD-BERT
$ python -u -m supar.cmds.crf_con train -b -d 0 -c crf_con-lang -p model  \
    -f char bert  \
    --train ud/lang/lang-train.conllx  \
    --dev ud/lang//lang-dev.conllx  \
    --test ud/lang/lang-test.conllx  \
    --n_bert_layers 5 \
    --bert_type udbert \
    --bert the-path-of-udbert/  \
    --mbr --freeze  \
    --fusion_ud
```

Specifying --mbr to perform MBR decoding often leads to consistent improvement.

## Citation

```sh
@inproceedings{
sun2021multilingual,
title={Multilingual Pre-training with Universal Dependency Learning},
author={Kailai Sun and Zuchao Li and hai zhao},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=5JIAKpVrmZK}
}
```
