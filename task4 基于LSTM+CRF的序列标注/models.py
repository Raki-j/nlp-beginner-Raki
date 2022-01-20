import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchcrf import CRF

class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_index, embedding_size, hidden_size, max_length, vectors=None):
        super(LSTM_CRF, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tag_to_index = tag_to_index
        self.target_size = len(tag_to_index)
        if vectors is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(vectors)
        self.lstm = nn.LSTM(embedding_size, hidden_size // 2, bidirectional=True)
        self.hidden_to_tag = nn.Linear(hidden_size, self.target_size)
        self.crf = CRF(self.target_size, batch_first=True)
        self.max_length = max_length

    def get_mask(self, length_list):
        mask = []
        for length in length_list:
            mask.append([1 for i in range(length)] + [0 for j in range(self.max_length - length)])
        return torch.tensor(mask, dtype=torch.bool)

    def LSTM_Layer(self, sentences, length_list):

        embeds = self.embedding(sentences)

        packed_sentences = pack_padded_sequence(embeds, lengths=length_list, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(packed_sentences)

        result, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.max_length)

        feature = self.hidden_to_tag(result)

        return feature

    def CRF_layer(self, input, targets, length_list):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        return self.crf(input, targets, self.get_mask(length_list))

    def forward(self, sentences, length_list, targets):
        x = self.LSTM_Layer(sentences, length_list)
        x = self.CRF_layer(x, targets, length_list)

        return x

    def predict(self, sentences, length_list):
        out = self.LSTM_Layer(sentences, length_list)
        mask = self.get_mask(length_list)

        return self.crf.decode(out, mask)

