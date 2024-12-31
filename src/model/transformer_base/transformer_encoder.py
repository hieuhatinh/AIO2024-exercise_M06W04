import torch
import torch.nn as nn

from transformer_base.transformer_model import TokenAndPositionEmbedding


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim, bias=True)
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(query + attn_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        out_2 = self.layernorm_2(out_1 + ffn_output)
        return out_2


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super(TransformerEncoder, self).__init__()
        self.embedding = TokenAndPositionEmbedding(
            vocab_size,
            embed_dim,
            max_length,
            device
        )
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                ) for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, x, x)
        return x


# multihead attention
# class MyMultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=1):
#         super(MyMultiHeadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert (
#             self.head_dim * num_heads == self.embed_dim
#         ), "embed_dim must be divisible by num_heads"

#         self.query_linear = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.key_linear = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.value_linear = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)

#     def forward(self, query, key, value):
#         batch_size = query.size(0)

#         # Linear projections
#         query = self.query_linear(query)
#         key = self.key_linear(key)
#         value = self.value_linear(value)
#         print('query shape 1: ', query.shape)

#         # reshape query, key, value to (batch_size, num_heads, seq_len, head_dim)
#         query = query.view(batch_size, self.num_heads, -1, self.head_dim)
#         key = key.view(batch_size, self.num_heads, -1, self.head_dim)
#         value = value.view(batch_size, self.num_heads, -1, self.head_dim)
#         print('query shape 2: ', query.shape)

#         # scaled dot-product attention
#         attn_weights = query.matmul(key.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         normal_weights = nn.functional.softmax(attn_weights, dim=-1)
#         context = normal_weights.matmul(value)
#         print('context shape', context.shape)

#         context = context.view(batch_size, -1, self.embed_dim)
#         output = self.out_linear(context)

#         return output, normal_weights
