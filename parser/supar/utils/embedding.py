# -*- coding: utf-8 -*-

import torch
import gzip

class Embedding(object):

    def __init__(self, tokens, vectors, unk=None):
        self.tokens = tokens
        self.vectors = torch.tensor(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.pretrained

    @property
    def dim(self):
        return self.vectors.size(1)

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.tokens.index(self.unk)
        else:
            raise AttributeError

    @classmethod
    def load(cls, path, unk=None):
        with open(path, 'r') as f:
            lines = [line for line in f]
        '''
        with gzip.open(path, 'rt') as f:
            lines = []
            i = 0
            for line in f:
                i += 1
                if i == 1:
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                lines.append(line)
        '''
        splits = [line.split() for line in lines]
        tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
                                for s in splits])

        return cls(tokens, vectors, unk=unk)
