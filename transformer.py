import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embedding_dim,
                 number_of_heads):
        super(MultiHeadAttention, self).__init__()

        assert embedding_dim % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dim // number_of_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)

        self.out_projection_net = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, v, k, q, mask=None):
        # q.shape = None, max_len, embed_dim

        batch_size = q.shape[0]
        q = self.query(q).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        # todo malo ovaj deo ispitaj i ovu masku i to

        # scaled dot-product attetnion
        # matmul and scale
        out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        # mask
        if mask is not None:
            out.masked_fill_(mask == torch.tensor(False), float("-inf"))
        # softmax
        out = torch.softmax(out, dim=-1)
        # matmul
        out = torch.matmul(out, v)

        out = out.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        out = self.out_projection_net(out)
        return out


class FeedForward(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dropout_probability,
                 width_mult=4):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim * width_mult)
        self.linear2 = nn.Linear(in_features=embedding_dim * width_mult, out_features=embedding_dim)

        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim,
                 number_of_heads,
                 dropout_probability):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim,
                                                       number_of_heads)
        self.feed_forward = FeedForward(embedding_dim,
                                        dropout_probability)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        out = self.multi_head_attention(x, x, x)
        out = self.norm(out)
        out = x + self.dropout(out)

        out = self.feed_forward(out)
        out = self.norm(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim,
                 number_of_heads,
                 dropout_probability):
        super(DecoderBlock, self).__init__()

        self.masked_multi_head_attention = MultiHeadAttention(embedding_dim,
                                                              number_of_heads)
        self.multi_head_attention = MultiHeadAttention(embedding_dim,
                                                       number_of_heads)
        self.feed_forward = FeedForward(embedding_dim,
                                        dropout_probability)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, encoder_output, x):
        out = self.masked_multi_head_attention(x, x, x)
        out = self.norm(out)
        out = x + self.dropout(out)

        x = out

        out = self.multi_head_attention(encoder_output, encoder_output, x)
        out = self.norm(out)
        out = x + self.dropout(out)

        out = self.feed_forward(out)
        out = self.norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self,
                 n_blocks,
                 embedding_dim,
                 number_of_heads,
                 dropout_probability):
        super(Encoder, self).__init__()
        self.encoder_blocks = [EncoderBlock(embedding_dim,
                                            number_of_heads,
                                            dropout_probability)
                               for _ in range(n_blocks)]

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 n_blocks,
                 embedding_dim,
                 number_of_heads,
                 dropout_probability):
        super(Decoder, self).__init__()
        self.decoder_blocks = [DecoderBlock(embedding_dim,
                                            number_of_heads,
                                            dropout_probability)
                               for _ in range(n_blocks)]

    def forward(self, encoder_output, x):
        for block in self.decoder_blocks:
            x = block(encoder_output, x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout_probability, expected_max_sequence_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim)

        positional_encodings_table = torch.zeros(expected_max_sequence_length, embedding_dim)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)

        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        return self.dropout(embeddings_batch + positional_encodings)


class Transformer(nn.Module):
    def __init__(self,
                 embedding_dim,
                 input_vocab_size,
                 output_vocab_size,
                 max_len_input,
                 max_len_output,
                 number_of_heads,
                 dropout_probability,
                 n_blocks):
        super().__init__()

        self.encoder = Encoder(n_blocks,
                               embedding_dim,
                               number_of_heads,
                               dropout_probability)
        self.decoder = Decoder(n_blocks,
                               embedding_dim,
                               number_of_heads,
                               dropout_probability)

        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(output_vocab_size, embedding_dim)

        # todo create real positional i vidi sta treba da bude embedding_dim
        self.input_positional_encoding = PositionalEncoding(embedding_dim, dropout_probability, max_len_input)
        self.output_positional_encoding = PositionalEncoding(embedding_dim, dropout_probability, max_len_output)

        # todo ovo samo pretpostavljam
        self.output_layer = nn.Linear(embedding_dim, output_vocab_size)

    def forward(self, inputs, outputs):
        # inputs, outputs = batch, len_text
        inputs = self.input_embedding(inputs)
        inputs = self.input_positional_encoding(inputs)

        outputs = self.output_embedding(outputs)
        outputs = self.output_positional_encoding(outputs)

        # inputs, outputs = batch, len_text, embed_dim

        encoder_output = self.encoder(inputs)
        decoder_output = self.decoder(encoder_output, outputs)

        predictions = self.output_layer(decoder_output)

        return predictions


if __name__ == '__main__':
    embedding_dim = 512
    input_vocab_size = 1000
    output_vocab_size = 1000
    max_len_input = 100
    max_len_output = 100
    number_of_heads = 8
    dropout_probability = 0.5
    n_blocks = 6

    model = Transformer(
        embedding_dim,
        input_vocab_size,
        output_vocab_size,
        max_len_input,
        max_len_output,
        number_of_heads,
        dropout_probability,
        n_blocks)

    a = model(torch.ones((128, 25), dtype=int),
              torch.ones((128, 50), dtype=int))
    pass
