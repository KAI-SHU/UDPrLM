import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


def get_parsing_orders(heads):
    orders = []
    for idx in range(len(heads)):
        h = heads[idx]
        o = 1
        while h != 0:
            h = heads[h-1]
            o += 1
        orders.append(o)
    return orders


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, index, words, postags, orders, heads, labels):
        """Constructs a InputExample.
        Args:
            index: int. Index for the example.
            words: list. The words of the sequence.
            postags: (Optional) list. The postags of the sequence.
            heads: (Optional) list. The dependency heads of the sequence.
            labels: (Optional) list. The dependency labels for each word of the sequence.
        """
        self.index = index
        self.words = words
        self.postags = postags
        self.orders = orders
        self.heads = heads
        self.labels = labels

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, index, input_ids, input_mask, segment_ids, postag_ids, head_ids, label_ids, word_token_starts, word_token_ends, token_word_indexs):
        self.index = index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.postag_ids = postag_ids
        self.head_ids = head_ids
        self.label_ids = label_ids
        self.word_token_starts = word_token_starts
        self.word_token_ends = word_token_ends
        self.token_word_indexs = token_word_indexs


def read_examples_from_file(data_dir, file_name, is_training=True, use_postag=True, with_parsing_order=False):
    file_path = os.path.join(data_dir, file_name)
    example_index = 0
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        postags = []
        heads = []
        labels = []
        for line in f:
            if len(line.strip())==0:
                if words:
                    examples.append(InputExample(index=example_index,
                                                 words=words,
                                                 postags=postags if use_postag else None,
                                                 orders=get_parsing_orders(heads) if is_training and with_parsing_order else None,
                                                 heads=heads if is_training else None,
                                                 labels=labels if is_training else None))
                    example_index += 1
                    words = []
                    postags = []
                    heads = []
                    labels = []
            else:
                splits = line.strip().split("\t")
                words.append(splits[1])
                if use_postag:
                    postags.append(splits[4])
                if is_training:
                    heads.append(int(splits[6]))
                    labels.append(splits[7])
        if words:
            examples.append(InputExample(index=example_index,
                                                 words=words,
                                                 postags=postags if use_postag else None,
                                                 orders=get_parsing_orders(heads) if is_training and with_parsing_order else None,
                                                 heads=heads if is_training else None,
                                                 labels=labels if is_training else None))
    return examples


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        is_training=False,
        use_postag=False,
        postag_list=None,
        label_list=None,
        pad_postag='_',
        pad_label='_',
        convert_strategy=0,
        special_postag='_',
        special_label='_',
    ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    postag_map = {postag: i for i, postag in enumerate(postag_list)} if postag_list is not None else None
    label_map = {label: i for i, label in enumerate(label_list)} if label_list is not None else None

    if use_postag:
        assert postag_map is not None and pad_postag in postag_map.keys()
        pad_postag_id = postag_map[pad_postag]
        if convert_strategy > 0:
            assert special_postag in postag_map.keys()
            special_postag_id = postag_map[special_postag]
    if is_training:
        assert label_map is not None and pad_label in label_map.keys()
        pad_label_id = label_map[pad_label]
        if convert_strategy > 0:
            assert special_label in label_map.keys()
            special_label_id = label_map[special_label]
    
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        # ???????????????????????????????????????word????????????????????????subword??????????????????????????????
        # 0. ???word???????????????subword???head????????????word????????????subword???label?????????word??????
        # 1. ???word??????????????????subword???head????????????word????????????subword???label?????????word???????????????subword???head??????????????????subword???label??????????????????
        # 2. ???word??????????????????subword???head????????????word????????????subword???label?????????word???????????????subword???head??????????????????subword???label??????????????????
        
        tokens = []
        word_token_starts = []
        word_token_ends = []
        token_word_indexs = []
        for word_index, word in enumerate(example.words):
            word_tokens = tokenizer.tokenize(word)
            word_token_starts.append(len(tokens))
            tokens.extend(word_tokens)
            token_word_indexs.extend([word_index]*len(word_tokens))
            word_token_ends.append(len(tokens)-1)

        postag_ids = None
        head_ids = None
        label_ids = None
        
        if use_postag:
            postag_ids = []
            for word_index, postag in enumerate(example.postags):
                token_span_len = word_token_ends[word_index]-word_token_starts[word_index]+1
                if convert_strategy == 0:
                    postag_ids.extend([postag_map[postag]]*token_span_len)
                else:
                    postag_ids.extend([postag_map[postag]] + [special_postag_id]*(token_span_len-1))


        if is_training:
            if convert_strategy > 0:
                assert special_postag_id >= 0
                assert special_label_id >= 0
            head_ids = []
            label_ids = []
            for word_index, head in enumerate(example.heads):
                head = head - 1 # the truely word index
                label = example.labels[word_index]
                token_span_len = word_token_ends[word_index]-word_token_starts[word_index]+1
                if convert_strategy == 0:
                    if head == -1: # position for <ROOT>
                        head_ids.extend([-1]*token_span_len)
                    else:
                        head_ids.extend([word_token_starts[head]]*token_span_len)
                    label_ids.extend([label_map[label]]*token_span_len)
                elif convert_strategy == 1:
                    if head == -1: # position for <ROOT>
                        head_ids.append([-1]+[word_token_starts[word_index]]*(token_span_len-1))
                    else:
                        head_ids.append([word_token_starts[head]]+[word_token_starts[word_index]]*(token_span_len-1))
                    label_ids.extend([label_map[label]] + [special_label_id]*(token_span_len-1))
                else:
                    if head == -1: # position for <ROOT>
                        head_ids.append([-1]+[word_token_starts[word_index]+idx for idx in range(token_span_len-1)])
                    else:
                        head_ids.append([word_token_starts[head]]+[word_token_starts[word_index]+idx for idx in range(token_span_len-1)])
                    label_ids.extend([label_map[label]] + [special_label_id]*(token_span_len-1))

        
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        assert len(tokens) <= max_seq_length - special_tokens_count, (len(tokens), max_seq_length, special_tokens_count)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For dependency parsing tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens += [sep_token]
        token_word_indexs += [None]
        if use_postag:
            postag_ids += [pad_postag_id]
        if is_training:
            head_ids += [-1]
            label_ids += [pad_label_id]
        
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            token_word_indexs += [None]
            if use_postag:
                postag_ids += [pad_postag_id]
            if is_training:
                head_ids += [-1]
                label_ids += [pad_label_id]
        
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # position for <ROOT>, since we use [CLS] for ROOT
        root_index = None

        if cls_token_at_end: # will not change the index
            root_index = len(tokens)
            tokens += [cls_token]
            token_word_indexs += [None]
            if use_postag:
                postag_ids += [pad_postag_id]
            if is_training:
                head_ids += [-1]
                label_ids += [pad_label_id]
            segment_ids += [cls_token_segment_id]
        else: # index changed
            root_index = 0
            tokens = [cls_token] + tokens
            token_word_indexs = [None] + token_word_indexs
            if use_postag:
                postag_ids = [pad_postag_id] + postag_ids
            if is_training:
                head_ids = [-1] + head_ids
                label_ids = [pad_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

            # IMPORTANT!!!
            word_token_starts = [idx+1 for idx in word_token_starts]
            word_token_ends = [idx+1 for idx in word_token_ends]
            if is_training:
                head_ids = [idx+1 if (idx != -1 and idx is not None) else idx for idx in head_ids]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left: # index changed
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            token_word_indexs = ([None] * padding_length) + token_word_indexs
            if use_postag:
                postag_ids = ([pad_postag_id] * padding_length) + postag_ids
            if is_training:
                head_ids = ([-1] * padding_length) + head_ids
                label_ids = ([pad_label_id] * padding_length) + label_ids
            
            # IMPORTANT!!!
            root_index += padding_length
            word_token_starts = [idx+padding_length for idx in word_token_starts]
            word_token_ends = [idx+padding_length for idx in word_token_ends]
            if is_training:
                head_ids = [idx+padding_length if (idx != -1 and idx is not None) else idx for idx in head_ids]
        else: # will not change the index
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            token_word_indexs = token_word_indexs + ([None] * padding_length)
            if use_postag:
                postag_ids = postag_ids + ([pad_postag_id] * padding_length)
            if is_training:
                head_ids = head_ids + ([-1] * padding_length)
                label_ids = label_ids + ([pad_label_id] * padding_length)

        # replace -1 to truely root position
        if is_training:
            head_ids = [idx if idx != -1 else root_index for idx in head_ids]
        
        # IMPORTANT: word length change to +1, due to <ROOT> ([CLS] in this work)
        word_token_starts = [root_index] + word_token_starts
        word_token_ends = [root_index] + word_token_ends

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if use_postag:
            assert len(postag_ids) == max_seq_length
        if is_training:
            assert len(head_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("index: %s", example.index)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            if use_postag:
                logger.info("postag_ids: %s", " ".join([str(x) for x in postag_ids]))
            if is_training:
                logger.info("head_ids: %s", " ".join([str(x) for x in head_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        
        features.append(
                InputFeatures(index=example.index,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              postag_ids=postag_ids,
                              head_ids=head_ids,
                              label_ids=label_ids,
                              word_token_starts=word_token_starts,
                              word_token_ends=word_token_ends,
                              token_word_indexs=token_word_indexs
                              ))

    return features


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    return labels


def save_labels(path, labels):
    with open(path, "w") as fout:
        for item in labels:
            fout.write(item+"\n")


def write_conll_examples(words, postags, heads, labels, file_path):
    assert len(words) == len(heads) and len(heads) == len(labels)
    with open(file_path, 'w') as f:
        for i in range(len(words)):
            assert len(words[i]) == len(heads[i]) and len(heads[i]) == len(labels[i])
            for j in range(len(words[i])):
                f.write('{}\t{}\t_\t{}\t_\t_\t{}\t{}\t_\t_\n'.format(j+1, words[i][j], postags[i][j] if postags is not None else '_', heads[i][j], labels[i][j]))
            f.write('\n')

def load_examples(model_args, data_args,training_args,max_seq_length, tokenizer, is_training=False, postags=None, labels=None, pad_postag='_', pad_label='_', convert_strategy=0, special_postag='_', special_label='_'):
    if training_args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset

    logger.info("Creating features from dataset file: %s",  data_args.train_file)

    examples = read_examples_from_file(data_args.train_file, is_training=is_training, use_postag=args.use_postag, with_parsing_order=False)

    features = convert_examples_to_features_with_parsing_order(
        examples, max_seq_length, tokenizer,
        cls_token_at_end=bool(model_args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if model_args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(model_args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(model_args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if model_args.model_type in ["xlnet"] else 0,
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

    if training_args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset

    # Convert to Tensors and build dataset
    all_example_ids = torch.tensor([f.index for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.use_postag:
        all_postag_ids = torch.tensor([f.postag_ids for f in features], dtype=torch.long)
    if is_training:
        all_order_ids = torch.tensor([f.order_ids for f in features], dtype=torch.long)
        all_head_ids = torch.tensor([f.head_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    
    if data_args.use_postag and is_training:
        dataset = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids, all_postag_ids, all_order_ids, all_head_ids, all_label_ids)
    elif data_args.use_postag:
        dataset = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids, all_postag_ids)
    elif is_training:
        dataset = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids, all_order_ids, all_head_ids, all_label_ids)
    else:
        dataset = TensorDataset(all_example_ids, all_input_ids, all_input_mask, all_segment_ids)
    
    return (dataset, examples, features)