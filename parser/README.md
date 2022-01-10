### Multilingual Pre-training with Universal Dependency Learning
## Introduction

The pre-trained language model (PrLM) demonstrates domination in downstream natural language processing tasks, in which multilingual PrLM takes advantage of language universality to alleviate the issue of limited resources for low-resource languages. Despite its successes, the performance of multilingual PrLM is still unsatisfactory, when multilingual PrLMs only focus on plain text and ignore obvious universal linguistic structure clues. Existing PrLMs have shown that monolingual linguistic structure knowledge may bring about better performance. Thus we propose a novel multilingual PrLM that supports both explicit universal dependency parsing and implicit language modeling. Syntax in terms of universal dependency parse serves as not only pre-training objective but also learned representation in our model, which brings unprecedented PrLM interpretability and convenience in downstream task use. Our model outperforms two popular multilingual PrLM, multilingual-BERT and XLM-R, on cross-lingual natural language understanding (NLU) benchmarks and linguistic structure parsing datasets, demonstrating the effectiveness and stronger cross-lingual modeling capabilities of our approach.

![Image text](https://github.com/KAI-SHU/UDPrLM/blob/main/parser/figure/20211017_udbert_arch.pdf)
## Introduction

We follow **Supar** ([Zhang et al.](https://github.com/yzhangcs/parser)), a Python package that contains many state-of-the-art syntactic/semantic parsers, as well as highly parallel implementations of several effective structured prediction algorithms. We add support for multilingual universal syntactic dependency pretrained model (UDPrLM) including UD-BERT, UD-XLM-R<sub>base</sub> and UD-XLM-R<sub>large</sub>.

## Installation

```sh
$ git clone https://github.com/KAI-SHU/UDPrLM && cd UDPrLM/parser
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
 
 Different from conventional evaluation manner of executing `EVALB`, we internally integrate python code for constituency tree evaluation. As different treebanks do not share the same evaluation parameters, it is recommended to evaluate the results in interactive mode.
 To evaluate English and Chinese models:
```py
>>> Parser.load('crf-con-en').evaluate('ptb/test.pid',
                                       delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                                       equal={'ADVP': 'PRT'},
                                       verbose=False)
(0.21318972731630007, UCM: 50.08% LCM: 47.56% UP: 94.89% UR: 94.71% UF: 94.80% LP: 94.16% LR: 93.98% LF: 94.07%)
>>> Parser.load('crf-con-zh').evaluate('ctb7/test.pid',
                                       delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                                       equal={'ADVP': 'PRT'},
                                       verbose=False)
(0.3994724107416053, UCM: 24.96% LCM: 23.39% UP: 90.88% UR: 90.47% UF: 90.68% LP: 88.82% LR: 88.42% LF: 88.62%)
```

To evaluate the multilingual model:
```py
>>> Parser.load('crf-con-xlmr').evaluate('spmrl/eu/test.pid',
                                         delete={'TOP', 'ROOT', 'S1', '-NONE-', 'VROOT'},
                                         equal={},
                                         verbose=False)
(0.45620645582675934, UCM: 53.07% LCM: 48.10% UP: 94.74% UR: 95.53% UF: 95.14% LP: 93.29% LR: 94.07% LF: 93.68%)
```
