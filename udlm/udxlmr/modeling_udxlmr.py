import logging
import math
import os
import sys
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import (RobertaLayer, RobertaPreTrainedModel, RobertaModel, RobertaLMHead, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, RobertaClassificationHead)
from .configuration_udxlmr import UDXLMRobertaConfig
sys.path.append("..")
from supar.udlm.parsing import BiaffineDependencyParsingHead
logger = logging.getLogger(__name__)


class UDXLMRobertaModel(RobertaPreTrainedModel):
    config_class = UDXLMRobertaConfig
    def __init__(self, config):
        super().__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.parsing_head = BiaffineDependencyParsingHead(config)
        self.roberta_layer = RobertaLayer(config)
        self.use_ud_repr = True
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

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
        postag_ids=None, 
        head_ids=None, 
        label_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parsing=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
            if self.use_ud_repr:
                sequence_output = self.parsing_head.ud_repr(sequence_output, attention_mask)
                if input_ids is not None:
                    input_shape = input_ids.size()
                elif inputs_embeds is not None:
                    input_shape = inputs_embeds.size()[:-1]
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                extended_attention_mask: torch.Tensor = self.roberta.get_extended_attention_mask(attention_mask, input_shape, device)
                sequence_output = self.roberta_layer(
                                            hidden_states=sequence_output,
                                            attention_mask=extended_attention_mask,
                                            encoder_hidden_states=encoder_hidden_states,
                                            output_attentions=output_attentions,
                                        )[0]
                if self.config.output_hidden_states:
                    if not return_dict:
                        outputs[-1] = outputs[-1] + (sequence_output,)
                    else:
                        outputs['hidden_states'] = outputs['hidden_states'] + (sequence_output,)
            prediction_scores = self.lm_head(sequence_output)

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
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


class UDXLMRobertaForSequenceClassification(RobertaPreTrainedModel):
    config_class = UDXLMRobertaConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.parsing_head = BiaffineDependencyParsingHead(config)
        self.roberta_layer = RobertaLayer(config)
        self.use_ud_repr = True
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.use_ud_repr:
            sequence_output = self.parsing_head.ud_repr(sequence_output, attention_mask)
            if input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            extended_attention_mask: torch.Tensor = self.roberta.get_extended_attention_mask(attention_mask, input_shape, device)
            sequence_output = self.roberta_layer(
                                        hidden_states=sequence_output,
                                        attention_mask=extended_attention_mask,
                                        output_attentions=output_attentions,
                                    )[0]
            if self.config.output_hidden_states:
                if not return_dict:
                    outputs[-1] = outputs[-1] + (sequence_output,)
                else:
                    outputs['hidden_states'] = outputs['hidden_states'] + (sequence_output,)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class UDXLMRobertaForQuestionAnswering(RobertaPreTrainedModel):
    config_class = UDXLMRobertaConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.parsing_head = BiaffineDependencyParsingHead(config)
        self.roberta_layer = RobertaLayer(config)
        self.use_ud_repr = True
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.use_ud_repr:
            sequence_output = self.parsing_head.ud_repr(sequence_output, attention_mask)
            if input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            extended_attention_mask: torch.Tensor = self.roberta.get_extended_attention_mask(attention_mask, input_shape, device)
            sequence_output = self.roberta_layer(
                                        hidden_states=sequence_output,
                                        attention_mask=extended_attention_mask,
                                        output_attentions=output_attentions,
                                    )[0]
            if self.config.output_hidden_states:
                if not return_dict:
                    outputs[-1] = outputs[-1] + (sequence_output,)
                else:
                    outputs['hidden_states'] = outputs['hidden_states'] + (sequence_output,)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )