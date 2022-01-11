from overrides import overrides
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss
import numpy as np

class BiAAttention(nn.Module):
    '''
    Bi-Affine attention layer.
    '''

    def __init__(self, input_size_encoder, input_size_decoder, num_labels, biaffine=True, **kwargs):
        '''
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''
        super(BiAAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter('U', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_d)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.constant_(self.b, 0.)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        '''
        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]
        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        '''
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [num_label, input_size_decoder] * [batch, input_size_decoder, length_decoder]
        # the output shape is [batch, num_label, length_decoder]
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        # compute decoder part: [num_label, input_size_encoder] * [batch, input_size_encoder, length_encoder]
        # the output shape is [batch, num_label, length_encoder]
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        # output shape [batch, num_label, length_decoder, length_encoder]
        if self.biaffine:
            # compute bi-affine part
            # [batch, 1, length_decoder, input_size_decoder] * [num_labels, input_size_decoder, input_size_encoder]
            # output shape [batch, num_label, length_decoder, input_size_encoder]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
            # output shape [batch, num_label, length_decoder, length_encoder]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))

            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output


class BiLinear(nn.Module):
    '''
    Bi-linear layer
    '''
    def __init__(self, left_features, right_features, out_features, bias=True):
        '''
        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        '''
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        '''
        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]
        Returns:
        '''

        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], \
            "batch size of left and right inputs mis-match: (%s, %s)" % (left_size[:-1], right_size[:-1])
        batch = int(np.prod(left_size[:-1]))

        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)

        # output [batch, out_features]
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        # convert back to [batch1, batch2, ..., out_features]
        return output.view(left_size[:-1] + (self.out_features, ))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'in1_features=' + str(self.left_features) \
               + ', in2_features=' + str(self.right_features) \
               + ', out_features=' + str(self.out_features) + ')'


class CharCNN(nn.Module):
    """
    CNN layers for characters
    """
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels=None, activation='elu'):
        super(CharCNN, self).__init__()
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            ACT = nn.ELU
        else:
            ACT = nn.Tanh
        layers = list()
        for i in range(num_layers - 1):
            layers.append(('conv{}'.format(i), nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)))
            layers.append(('act{}'.format(i), ACT()))
            in_channels = hidden_channels
        layers.append(('conv_top', nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)))
        layers.append(('act_top', ACT()))
        self.act = ACT
        self.net = nn.Sequential(OrderedDict(layers))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)
            else:
                assert isinstance(layer, self.act)

    def forward(self, char):
        """

        Args:
            char: Tensor
                the input tensor of character [batch, sent_length, char_length, in_channels]

        Returns: Tensor
            output character encoding with shape [batch, sent_length, in_channels]

        """
        # [batch, sent_length, char_length, in_channels]
        char_size = char.size()
        # first transform to [batch * sent_length, char_length, in_channels]
        # then transpose to [batch * sent_length, in_channels, char_length]
        char = char.view(-1, char_size[2], char_size[3]).transpose(1, 2)
        # [batch * sent_length, out_channels, char_length]
        char = self.net(char).max(dim=2)[0]
        # [batch, sent_length, out_channels]
        return char.view(char_size[0], char_size[1], -1)


def decode_MST(energies, lengths, leading_symbolic=0, labeled=True):
    """
    decode best parsing tree with MST algorithm.
    :param energies: energies: numpy 4D tensor
        energies of each edge. the shape is [batch_size, num_labels, n_steps, n_steps],
        where the summy root is at index 0.
    :param masks: numpy 2D tensor
        masks in the shape [batch_size, n_steps].
    :param leading_symbolic: int
        number of symbolic dependency types leading in type alphabets)
    :return:
    """

    def find_cycle(par):
        added = np.zeros([length], np.bool)
        added[0] = True
        cycle = set()
        findcycle = False
        for i in range(1, length):
            if findcycle:
                break

            if added[i] or not curr_nodes[i]:
                continue

            # init cycle
            tmp_cycle = set()
            tmp_cycle.add(i)
            added[i] = True
            findcycle = True
            l = i

            while par[l] not in tmp_cycle:
                l = par[l]
                if added[l]:
                    findcycle = False
                    break
                added[l] = True
                tmp_cycle.add(l)

            if findcycle:
                lorg = l
                cycle.add(lorg)
                l = par[lorg]
                while l != lorg:
                    cycle.add(l)
                    l = par[l]
                break

        return findcycle, cycle

    def chuLiuEdmonds():
        par = np.zeros([length], dtype=np.int32)
        # create best graph
        par[0] = -1
        for i in range(1, length):
            # only interested at current nodes
            if curr_nodes[i]:
                max_score = score_matrix[0, i]
                par[i] = 0
                for j in range(1, length):
                    if j == i or not curr_nodes[j]:
                        continue

                    new_score = score_matrix[j, i]
                    if new_score > max_score:
                        max_score = new_score
                        par[i] = j

        # find a cycle
        findcycle, cycle = find_cycle(par)
        # no cycles, get all edges and return them.
        if not findcycle:
            final_edges[0] = -1
            for i in range(1, length):
                if not curr_nodes[i]:
                    continue

                pr = oldI[par[i], i]
                ch = oldO[par[i], i]
                final_edges[ch] = pr
            return

        cyc_len = len(cycle)
        cyc_weight = 0.0
        cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
        for id, cyc_node in enumerate(cycle):
            cyc_nodes[id] = cyc_node
            cyc_weight += score_matrix[par[cyc_node], cyc_node]

        rep = cyc_nodes[0]
        for i in range(length):
            if not curr_nodes[i] or i in cycle:
                continue

            max1 = float("-inf")
            wh1 = -1
            max2 = float("-inf")
            wh2 = -1

            for j in cyc_nodes:
                if score_matrix[j, i] > max1:
                    max1 = score_matrix[j, i]
                    wh1 = j

                scr = cyc_weight + score_matrix[i, j] - score_matrix[par[j], j]

                if scr > max2:
                    max2 = scr
                    wh2 = j

            score_matrix[rep, i] = max1
            oldI[rep, i] = oldI[wh1, i]
            oldO[rep, i] = oldO[wh1, i]
            score_matrix[i, rep] = max2
            oldO[i, rep] = oldO[i, wh2]
            oldI[i, rep] = oldI[i, wh2]

        rep_cons = []
        for i in range(cyc_len):
            rep_cons.append(set())
            cyc_node = cyc_nodes[i]
            for cc in reps[cyc_node]:
                rep_cons[i].add(cc)

        for cyc_node in cyc_nodes[1:]:
            curr_nodes[cyc_node] = False
            for cc in reps[cyc_node]:
                reps[rep].add(cc)

        chuLiuEdmonds()

        # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
        found = False
        wh = -1
        for i in range(cyc_len):
            for repc in rep_cons[i]:
                if repc in final_edges:
                    wh = cyc_nodes[i]
                    found = True
                    break
            if found:
                break

        l = par[wh]
        while l != wh:
            ch = oldO[par[l], l]
            pr = oldI[par[l], l]
            final_edges[ch] = pr
            l = par[l]

    if labeled:
        assert energies.ndim == 4, 'dimension of energies is not equal to 4'
    else:
        assert energies.ndim == 3, 'dimension of energies is not equal to 3'
    input_shape = energies.shape
    batch_size = input_shape[0]
    max_length = input_shape[2]

    pars = np.zeros([batch_size, max_length], dtype=np.int32)
    types = np.zeros([batch_size, max_length], dtype=np.int32) if labeled else None
    for i in range(batch_size):
        energy = energies[i]

        # calc the realy length of this instance
        length = lengths[i]
   
        # calc real energy matrix shape = [length, length, num_labels - #symbolic] (remove the label for symbolic types).
        if labeled:
            energy = energy[leading_symbolic:, :length, :length]
            energy = energy - energy.min() + 1e-6
            # get best label for each edge.
            label_id_matrix = energy.argmax(axis=0) + leading_symbolic
            energy = energy.max(axis=0)
        else:
            energy = energy[:length, :length]
            energy = energy - energy.min() + 1e-6
            label_id_matrix = None
        # get original score matrix
        orig_score_matrix = energy
        # initialize score matrix to original score matrix
        score_matrix = np.array(orig_score_matrix, copy=True)

        oldI = np.zeros([length, length], dtype=np.int32)
        oldO = np.zeros([length, length], dtype=np.int32)
        curr_nodes = np.zeros([length], dtype=np.bool)
        reps = []

        for s in range(length):
            orig_score_matrix[s, s] = 0.0
            score_matrix[s, s] = 0.0
            curr_nodes[s] = True
            reps.append(set())
            reps[s].add(s)
            for t in range(s + 1, length):
                oldI[s, t] = s
                oldO[s, t] = t

                oldI[t, s] = t
                oldO[t, s] = s

        final_edges = dict()
        chuLiuEdmonds()
        par = np.zeros([max_length], np.int32)
        if labeled:
            type = np.ones([max_length], np.int32)
            type[0] = 0
        else:
            type = None

        for ch, pr in final_edges.items():
            par[ch] = pr
            if labeled and ch != 0:
                type[ch] = label_id_matrix[pr, ch]

        par[0] = 0
        pars[i] = par
        if labeled:
            types[i] = type

    return pars, types