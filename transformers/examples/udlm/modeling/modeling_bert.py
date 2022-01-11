import logging
import math
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from configuration.configuration_bert import BertForDependencyParsingConfig
from transformers.models.bert.modeling_bert import (BertPreTrainedModel, BertModel, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer, BertForPreTrainingOutput, MaskedLMOutput)
from .dependency import BiaffineDependencyParsingHead
logger = logging.getLogger(__name__)


class BertWithParsingModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.parsing_head = BiaffineDependencyParsingHead(config)
        self.bert_layer = BertLayer(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        next_sentence_label=None,
        postag_ids=None, 
        head_ids=None, 
        label_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parsing=False,
        use_ud_repr=True
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if parsing:
            sequence_output = outputs[0]
            if head_ids is not None and label_ids is not None:
                parsing_out = self.parsing_head.loss(sequence_output, head_ids, label_ids, attention_mask)
            else:
                parsing_out = self.parsing_head.decode(sequence_output, attention_mask)
            return parsing_out
        else:
            sequence_output = outputs[0]
            if use_ud_repr:
                sequence_output = self.parsing_head.ud_repr(sequence_output, attention_mask)
                if input_ids is not None:
                    input_shape = input_ids.size()
                elif inputs_embeds is not None:
                    input_shape = inputs_embeds.size()[:-1]
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)
                sequence_output = self.bert_layer(
                                            sequence_output,
                                            extended_attention_mask,
                                            encoder_hidden_states,
                                            None,
                                            output_attentions,
                                        )[0]
            prediction_scores = self.cls(sequence_output)

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return MaskedLMOutput(
                loss=masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
