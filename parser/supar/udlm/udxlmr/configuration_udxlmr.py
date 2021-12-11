from transformers import BertConfig

class UDXLMRobertaConfig(BertConfig):
    model_type = "udxlmr"
    def __init__(
        self, 
        pad_token_id=1, 
        bos_token_id=0, 
        eos_token_id=2, 
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_postag=False,
        num_postags=0,
        udtype2id=None,
        id2udtype=None,
        num_ud_labels=1,
        arc_space=128,
        label_space=64,
        projeciton=16,
        mlp_dropout=0.33,
        convert_strategy=2,
        special_label="APP",
        **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(
            pad_token_id=pad_token_id, 
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id, 
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            **kwargs)
        self.use_postag = use_postag
        self.num_postags = num_postags
        self.arc_space = arc_space
        self.label_space = label_space
        self.projeciton = projeciton
        self.mlp_dropout = mlp_dropout
        self.convert_strategy = convert_strategy
        self.special_label = special_label

        self.id2udtype = id2udtype
        self.udtype2id = udtype2id
        if self.id2udtype is not None:
            self.id2udtype = dict((int(key), value) for key, value in self.id2udtype.items())
        else:
            self.id2udtype = {i: f"LABEL_{i}" for i in range(num_ud_labels)}
            self.udtype2id = dict(zip(self.id2udtype.values(), self.id2udtype.keys()))

    @property
    def num_ud_labels(self) -> int:
        return len(self.id2udtype)