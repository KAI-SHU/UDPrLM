import torch
import argparse
import json
import os
import collections
import shutil
from torch import nn
import sys

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Convert UDXLMR Model Type')
    parser.add_argument("--input_json", type=str, help="Pretrained UDXLMR Model Config Input Path")
    parser.add_argument("--input_labels", type=str, help="Pretrained UDXLMR Model Config Input Path")
    parser.add_argument("--convert_strategy", type=int, default=2, help="Conver strategy")
    parser.add_argument("--special_label", type=str, default="APP", help="Special label")
    parser.add_argument("--output_json", type=str, help="Pretrained UDXLMR Model Config Output Path")
    args = parser.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as fin:
        config = json.load(fin)

    if '_name_or_path' in config.keys():
        del config['_name_or_path']

    config['architectures'][0] = 'UDXLMRModel'

    labels = []
    with open(args.input_labels, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if len(line) > 0:
                labels.append(line)
    
    assert len(labels) == len(config['id2label']) and len(config['id2label']) == len(config['label2id'])

    config['num_ud_labels'] = len(labels)

    config['convert_strategy'] = args.convert_strategy
    assert args.special_label in labels
    config['special_label'] = args.special_label

    config['model_type'] = 'udxlmr'

    del config['id2label']
    del config['label2id']

    config['id2udtype'] = {idx:item for idx, item in enumerate(labels)}
    config['udtype2id'] = {item:idx for idx, item in enumerate(labels)}

    with open(args.output_json, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(config, indent=4, sort_keys=True))