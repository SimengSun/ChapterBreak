'''
A module which implements the basic Transformer
'''
import uuid
import threading
import pdb
import sys
import torch
from torch import nn
import numpy as np
from utils import triu
from models.attention import MultiHeadedAttention
from models.embeddings import PositionEmbedding, TokenEmbedding
from transformers import RobertaTokenizer


class TransformerSublayer(nn.Module):
    '''
    Implements a sub layer of the transformer model, which consists of:
    1) A sub layer module
    2) Followed by dropout
    3) Plus a residual connection
    4) With layer normalization
    '''
    def __init__(self, sublayer, sublayer_shape, dropout_p=0.1, init_std=0.02):
        ''' Initialize the transformer sublayer '''
        super(TransformerSublayer, self).__init__()
        self.init_std = init_std
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(sublayer_shape)
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        # ''' Reset parameters using xavier initialiation '''
        # self.norm.reset_parameters()
        nn.init.normal_(self.norm.weight, 1.0, self.init_std)

    def forward(self, inputs, *sublayer_args, **sublayer_kwargs): # pylint:disable=arguments-differ
        ''' The forward pass of the sublayer '''
        out = self.dropout(self.sublayer(*sublayer_args, **sublayer_kwargs))
        return self.norm(inputs + out)


class TransformerFFN(nn.Module):
    ''' Implements the Transformer feed-forward network '''
    def __init__(self, embedding_size, hidden_dim, init_std=0.02):
        super(TransformerFFN, self).__init__()

        self.init_std = init_std
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(embedding_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, embedding_size)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        nn.init.normal_(self.hidden.weight, 0., self.init_std)
        nn.init.normal_(self.output.weight, 0., self.init_std)
        nn.init.constant_(self.hidden.bias, 0.)
        nn.init.constant_(self.output.bias, 0.)

    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' The forward pass of the feed-forward network '''
        return self.output(self.relu(self.hidden(inputs)))


class TransformerLayer(nn.Module):
    ''' Implements a single decoder layer in a transformer decoder stack '''
    def __init__(self, config, num_heads, dim, hidden_dim, causal=True, 
                 dropout_p=0.1):
        ''' Initialize the transformer layer '''
        super(TransformerLayer, self).__init__()

        self.uuid = uuid.uuid4() 

        self.ffn = TransformerSublayer(
            TransformerFFN(dim, hidden_dim),
            dim, dropout_p)

        self.self_attention = TransformerSublayer(
                MultiHeadedAttention(dim, num_heads=num_heads),
                dim, dropout_p) 

        # unidirectional lm
        self.causal = causal

        # enforce learned heads to look at local windows
        self.config = config

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        self.ffn.reset_parameters()
        self.self_attention.reset_parameters()

    def forward(self, inputs, global_mask=None): # pylint:disable=arguments-differ
        ''' The forward pass '''

        state = inputs['state']
        cache = inputs.get('cache')
        decoder_position = state.shape[1] - 1
        L = state.shape[1]
        # each layer might have different config
        residual = state
        kwargs = {}
        kwargs['attention_mask'] = self.mask(state)             # just causal mask
        kwargs['key_mask'] = inputs['padding_mask']

        state = self.self_attention(
            residual,                                           # residual
            state, state, state, **kwargs                       # passed to attention
        )

        state = self.ffn(
            state,                                              # residual
            state                                               # passed to FF layer
        )

        return {'state': state, 'padding_mask': inputs['padding_mask']}

    _masks = threading.local()
    def mask(self, inputs):
        '''
        Get a self-attention mask
        The mask will be of shape [T x T] containing elements from the set {0, -inf}
        Input shape:  (B x T x E)
        Output shape: (T x T)
        '''
        if not self.causal:
            return None

        dim = inputs.shape[1]
        device = inputs.device
        mask_store = TransformerLayer._masks.__dict__
        if device not in mask_store or (device in mask_store and mask_store[device].shape[1] < dim):
            mask = inputs.new_full((dim, dim), float('-inf'))
            mask_store[device] = triu(mask, 1, 1, 1)

        mask = mask_store[device]
        return mask[None, :dim, :dim]

class Transformer(nn.Module):
    ''' The Transformer LM module '''
    def __init__(self, config, encoder=False):
        ''' Initialize the Transformer '''
        super(Transformer, self).__init__()

        self.config = config

        self.encoder = encoder
        if encoder and config.train_encoder_from_scratch: # need to have token embedding
            # to get the embedding table size, load roberta-base tokenizer
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.padding_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            print(f"embedding table size {len(tokenizer)}")
            self.token_embedding = TokenEmbedding(
                                    len(tokenizer),
                                    config.embedding_size,
                                    padding_idx=self.padding_idx
                                    )
        self.position_embedding = PositionEmbedding(config.embedding_size)

        self.dropout = nn.Dropout(config.dropout_p, inplace=True)

        self.layers = self.create_layers(config, encoder=encoder)

        self.reset_named_parameters()

    @classmethod
    def create_layers(self, config, encoder=False, rpe=None):
        ''' Create the transformer decoders '''
        kwargs = {'dropout_p': config.dropout_p, 
                  'causal': not encoder}                    # sublayer kwargs

        args = [config, config.num_heads, config.model_size, config.hidden_dim]

        layers = nn.ModuleList([
            TransformerLayer(*args, **kwargs)
            for layer_i in range(config.num_layers)
        ])

        return layers

    def reset_named_parameters(self):

        for layer in self.layers:
            layer.reset_parameters()

        if hasattr(self, "token_embedding"):
            self.token_embedding.reset_parameters()

    def forward(self, batch): # pylint:disable=arguments-differ
        ''' batch: bsz x l x embed_dim if input is segment vector
            else
                   (bsz * num_chunks) x num_tokens_per_chunk otherwise
        '''

        if self.encoder:
            padding_mask = batch['padding_mask'].bool()
            batch = batch['data']
            _, L = batch.shape

        else:
            padding_mask = batch['padding_mask'][:, :-1].bool()
            batch = batch['data'][:, :-1, :]
            bsz, L, embed_dim = batch.shape

        pos_added_batch = self.embed(batch, token_embedding=getattr(self, "token_embedding", None))
        decoded = {'state': pos_added_batch, 'padding_mask' : padding_mask}
        
        # decoded['state'][batch == self.padding_idx] = 0
        for i, decoder in enumerate(self.layers):
            decoded = decoder(decoded)
        return decoded['state']          # bs x L x hidden_dim


    def embed(self, inputs, token_embedding=None):
        ''' Embed the given inputs '''
        if token_embedding is None: # input is segment vector, no need to encode each token
            return self.dropout(inputs + self.position_embedding(inputs))
        else:
            return self.dropout(token_embedding(inputs) + self.position_embedding(inputs))



