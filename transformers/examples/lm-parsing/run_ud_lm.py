import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import re
import json
import math
import os
import sys
from datasets import load_dataset
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertTokenizer,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    set_seed,
)

from training_args import TrainingArguments
import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from utils_dependency_parsing import convert_examples_to_features, get_labels, save_labels, read_examples_from_file, write_conll_examples
from trainer import Trainer

from configuration.configuration_bert import BertForDependencyParsingConfig
from configuration.configuration_roberta import XLMRobertaForDependencyParsingConfig
from transformers.trainer_utils import EvaluationStrategy, LoggingStrategy, SchedulerType
from modeling.modeling_bert import BertWithParsingModel
from modeling.modeling_roberta import XLMRobertaWithParsingModel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertForDependencyParsingConfig, BertWithParsingModel, BertTokenizer),
    "roberta": (XLMRobertaForDependencyParsingConfig, XLMRobertaWithParsingModel, XLMRobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    

def load_examples(args, file_name, tokenizer, is_training=False, postags=None, labels=None, pad_postag='_', pad_label='_', convert_strategy=0, special_postag='_', special_label='_'):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset

    logger.info("Creating features from dataset file: %s", os.path.join(args.data_dir, file_name))

    examples = read_examples_from_file(args.data_dir, file_name, is_training=is_training, use_postag=args.use_postag)

    features = convert_examples_to_features(
        examples, args.parsing_max_seq_length, tokenizer,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        is_training=is_training,
        use_postag=args.use_postag,
        postag_list=postags,
        label_list=labels,
        pad_postag=pad_postag,
        pad_label=pad_label,
        convert_strategy=convert_strategy,
        special_postag=special_postag,
        special_label=special_label
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset

    # Convert to Tensors and build dataset
    all_example_ids = torch.tensor([f.index for f in features], dtype=torch.long)
    all_feat_ids = torch.tensor([f.feat_index for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.use_postag:
        all_postag_ids = torch.tensor([f.postag_ids for f in features], dtype=torch.long)
    if is_training:
        all_head_ids = torch.tensor([f.head_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    
    if args.use_postag and is_training:
        dataset = TensorDataset(all_example_ids, all_feat_ids, all_input_ids, all_input_mask, all_segment_ids, all_postag_ids, all_head_ids, all_label_ids)
    elif args.use_postag:
        dataset = TensorDataset(all_example_ids, all_feat_ids, all_input_ids, all_input_mask, all_segment_ids, all_postag_ids)
    elif is_training:
        dataset = TensorDataset(all_example_ids, all_feat_ids, all_input_ids, all_input_mask, all_segment_ids, all_head_ids, all_label_ids)
    else:
        dataset = TensorDataset(all_example_ids, all_feat_ids, all_input_ids, all_input_mask, all_segment_ids)
    
    return (dataset, examples, features)


def main():
    # get TrainingArgments
    training_parser = HfArgumentParser((TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = training_parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = training_parser.parse_args_into_dataclasses()[0]

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if not os.path.exists(args.output_dir) and args.do_train:
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    '''
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(device.type)
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args._n_gpu = args.n_gpu
    args.device = device
    '''

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args)

    # MLM Dataset
    if not args.parsing_only:
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            mlm_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir = args.dataset_cache_dir)
            if "validation" not in mlm_datasets.keys():
                mlm_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir = args.dataset_cache_dir,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                mlm_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir = args.dataset_cache_dir,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
        else:
            data_files = {}
            if args.mlm_train_file is not None:
                data_files["train"] = args.mlm_train_file
            if args.mlm_validation_file is not None:
                data_files["validation"] = args.mlm_validation_file
            extension = args.mlm_train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            mlm_datasets = load_dataset(extension, data_files=data_files, cache_dir = args.dataset_cache_dir)
        if args.do_train:
            column_names = mlm_datasets["train"].column_names
        else:
            column_names = mlm_datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

    # Prepare dependency parsing task
    postags = None
    if args.use_postag:
        if len(args.postags) > 0:
            postags = get_labels(args.postags)
        else:
            postags = get_labels(os.path.join(args.model_name_or_path, 'postags.txt'))
        assert postags[0] == '_', 'PAD postag must be in the position 0'
    
    if len(args.labels) > 0:
        labels = get_labels(args.labels)
    else:
        labels = get_labels(os.path.join(args.model_name_or_path, 'labels.txt'))
    assert labels[0] == '_', 'PAD label must be in the position 0'
    if args.convert_strategy > 0:
        if args.use_postag:
            assert args.special_postag in postags
        assert args.special_label in labels
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                        use_postag=args.use_postag, num_postags=len(postags) if postags is not None else 0,
                                        num_labels=len(labels))
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    logger.info(model)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))

    if args.fixed_xlmr_large_lm:
        logger.info('Fixed language model!')
        for k, p in model.named_parameters():
            if k.startswith('roberta.') \
                and not k.startswith('roberta.encoder.layer.23.') \
                and not k.startswith('roberta.encoder.layer.22.') and not k.startswith('roberta.encoder.layer.21.'): # update the last layer
                logger.info('fixed param: %s' % k)
                p.requires_grad = False
            else:
                p.requires_grad = True
                logger.info('gradient updated param: %s' % k)

    if args.fixed_xlmr_base_lm:
        logger.info('Fixed language model!')
        for k, p in model.named_parameters():
            if k.startswith('roberta.') \
                and not k.startswith('roberta.encoder.layer.11.') \
                and not k.startswith('roberta.encoder.layer.10.') and not k.startswith('roberta.encoder.layer.9.'): # update the last layer
                logger.info('fixed param: %s' % k)
                p.requires_grad = False
            else:
                p.requires_grad = True
                logger.info('gradient updated param: %s' % k)

    if args.fixed_bert_base_lm:
        logger.info('Fixed language model!')
        for k, p in model.named_parameters():
            if k.startswith('bert.') \
                and not k.startswith('bert.encoder.layer.11.') \
                and not k.startswith('bert.encoder.layer.10.') and not k.startswith('bert.encoder.layer.9.'): # update the last layer
                logger.info('fixed param: %s' % k)
                p.requires_grad = False
            else:
                p.requires_grad = True
                logger.info('gradient updated param: %s' % k)

    # Parsing Data
    if args.do_train:
        assert args.train_file
        train_data = load_examples(
                            args, args.train_file, tokenizer, 
                            is_training=True,
                            postags=postags, 
                            labels=labels, 
                            pad_postag='_',
                            pad_label='_',
                            convert_strategy=args.convert_strategy,
                            special_postag=args.special_postag,
                            special_label=args.special_label
                        )
    if args.do_eval:
        assert args.eval_file
        eval_data = load_examples(
                            args, args.eval_file, tokenizer, 
                            is_training=True,
                            postags=postags, 
                            labels=labels, 
                            pad_postag='_',
                            pad_label='_',
                            convert_strategy=args.convert_strategy,
                            special_postag=args.special_postag,
                            special_label=args.special_label
                        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warn(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.parsing_only:
        tokenized_datasets = {"train":None, "validation":None}
    elif args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = mlm_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not args.overwrite_cache,
        )
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = mlm_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            cache_file_names = {k: os.path.join(args.dataset_cache_dir, f'tokenized_{str(k)}.arrow') for k in mlm_datasets}
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            cache_file_names = {k: os.path.join(args.dataset_cache_dir, f'tokenized_and_grouped_{str(k)}.arrow') for k in tokenized_datasets}
        )

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"] if args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if args.do_eval else None,
        p_train_data=train_data if args.do_train else (None, None, None),
        p_eval_data=eval_data if args.do_eval else (None, None, None),
        postags=postags,
        p_labels=labels,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    last_checkpoint = None
    if args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif args.model_name_or_path is not None and os.path.isdir(args.model_name_or_path):
            checkpoint = args.model_name_or_path
        else:
            checkpoint = None
        if args.parsing_only:
            train_result = trainer.train_parsing_only(resume_from_checkpoint=checkpoint)
        else:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero() and not args.parsing_only:
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if args.do_eval and not args.parsing_only:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(args.output_dir, "eval_results_mlm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
    if args.do_eval and args.parsing_only:
        logger.info("*** Evaluate ***")
        results = trainer.parsing_evaluate()
        print(results)
    return results

if __name__ == "__main__":
    main()
