import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import (MLP, Biaffine)

class StructuralAttentionLayer(nn.Module):

    def __init__(self, config, projeciton=True):
        super().__init__()
        hidden_size = config.hidden_size
        if projeciton:
            self.projection = nn.Linear(hidden_size, config.projeciton)
            hidden_size = config.projeciton
        else:
            self.projection = None
        self.head_WV = nn.Parameter(torch.randn(config.num_ud_labels, hidden_size, hidden_size))
        self.dense = nn.Linear(hidden_size * config.num_ud_labels, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, x, s_arc, s_rel, mask):
        p_arc = F.softmax(s_arc, dim=-1) * mask.unsqueeze(-1)
        p_rel = F.softmax(s_rel, -1)
        A = p_arc.unsqueeze(-1) * p_rel
        if self.projection:
            x = self.projection(x)
        Ax = torch.einsum('bijk,bih->bihk', A, x)
        AxW = torch.einsum('bihk,khm->bihk', Ax, self.head_WV)
        AxW = AxW.flatten(2)
        x = self.dense(AxW) # b*L*D
        x = self.activation(x)
        return x


class UDReprFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BiaffineDependencyParsingHead(nn.Module):
    def __init__(self,
                 config,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=config.hidden_size, n_out=config.arc_space, dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=config.hidden_size, n_out=config.arc_space, dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.hidden_size, n_out=config.label_space, dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.hidden_size, n_out=config.label_space, dropout=config.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.arc_space, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=config.label_space, n_out=config.num_ud_labels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.structural_attn = StructuralAttentionLayer(config)
        self.ud_repr_fusion =  UDReprFusionLayer(config)

    def forward(self, sequence_output, mask):
        # apply MLPs to the output states
        arc_d = self.mlp_arc_d(sequence_output)
        arc_h = self.mlp_arc_h(sequence_output)
        rel_d = self.mlp_rel_d(sequence_output)
        rel_h = self.mlp_rel_h(sequence_output)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def loss(self, sequence_output, arcs, rels, mask, partial=False):
        r"""
        Args:
            sequence_output (~torch.LongTensor): ``[batch_size, seq_len, hidden state]``.
                The tensor of the output of transformers.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
        Returns:
            ~torch.Tensor:
                The training loss.
        """
        mask = mask.type(torch.bool)
        s_arc, s_rel = self(sequence_output, mask)
        """
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
        """
        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, sequence_output, mask, tree=True, proj=False):
        r"""
        Args:
            sequence_output (~torch.LongTensor): ``[batch_size, seq_len, hidden state]``.
                The tensor of the output of transformers.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        mask = mask.type(torch.bool)
        s_arc, s_rel = self(sequence_output, mask)
        '''
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds
        '''
        return s_arc, s_rel

    def ud_repr(self, sequence_output, mask):
        mask = mask.type(torch.bool)
        s_arc, s_rel = self(sequence_output, mask)
        ud_repr = self.structural_attn(sequence_output, s_arc, s_rel, mask)
        ud_repr = self.ud_repr_fusion(ud_repr, sequence_output)
        return ud_repr
