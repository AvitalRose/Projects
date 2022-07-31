# Avital Rose 318413408
import torch
import torch.nn as nn
import math


class SentenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        # forward pass
        self.forward_pass = Layer(args, True)

        # backward pass
        self.backward_pass = Layer(args, False)

        # multi head attention and max pooling
        self.multi_dimensional_attention_and_max_pooling = MultiDimensionalAttentionAndMaxPooling(args=args)

    def forward(self, inputs, inputs_lengths):
        """
        Function to encode sentence. The sentence embedding is passed through two layers, forward layer and backward
        layer. Then the output of both layers are concatenated and passed to max pooling and multi dimension attention,
        and the results of those two stages are concatenated
        :param inputs: tensor
        :param inputs_lengths: list of lengths
        :return:
        """
        batch_size, sequence_length, embedding_dim = inputs.size()
        # get masks for premise or hypothesis
        inputs_mask = self.get_individual_mask(batch_size=inputs.shape[0], lengths=inputs_lengths)

        # pass sentence through both layers, each layer output has shape of (batch_size X seq_len X embedding_dim)
        forward_output = self.forward_pass(inputs, inputs_mask)
        backward_output = self.backward_pass(inputs, inputs_mask)

        # concatenated passes shape should be (batch_size X seq_len X 2 * embedding_dim)
        concatenated_passes = torch.cat([forward_output, backward_output], dim=-1)

        # multi dimensional attention and max pooling
        output = self.multi_dimensional_attention_and_max_pooling(concatenated_passes, inputs_mask)

        return output

    def get_individual_mask(self, batch_size, lengths):
        """
        Every sentence has a different length n, so the mask for every sentence should be different.
        The function gets the sentences and creates matrix mask where each row has 1's until length of sentence and then
        zeros.
        :param batch_size: int, size of batch
        :param lengths: list of ints
        :return:
        """
        max_sequence_length = max(lengths)
        mask = torch.FloatTensor(batch_size, max_sequence_length).to(
            self.device)  # creating new tensor, need to do to(device)
        mask.data.fill_(1)
        for i in range(batch_size):
            mask[i, lengths[i]:] = 0
        mask = mask.unsqueeze_(-1)
        return mask


class Layer(nn.Module):
    def __init__(self, args, forward_pass):
        super().__init__()
        # masked multi head dimension
        self.masked_multi_head_attention = MaskedMultiHeadAttention(args=args, forward_pass=forward_pass)

        # fusion gate and norm
        self.fusion_gate_and_norm = FusionGateAndNorm(args=args)

        # position wise
        self.position_wise_add_and_norm = PositionWiseAddAndNorm(args=args)

    def forward(self, inputs, inputs_mask):
        """
        Each layer consists of a multi head attention, followed by fusion gate (which receives also original input, i.e.
        residual connection), followed by  position wise feed forward networks and residual connection to fusion gate
        output
        :param inputs: tensor
        :param inputs_mask: tensor
        :return:
        """
        # triple the inputs to be the queries, keys and values
        output = self.masked_multi_head_attention(inputs, inputs, inputs, inputs_mask)

        # fusion gate gets output as well as skip connection
        output = self.fusion_gate_and_norm(inputs, output)

        # position wise feed forward, add and norm (with residual connection inside)
        output = self.position_wise_add_and_norm(output)

        return output


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, args, forward_pass):
        super().__init__()
        self.embeds_dim = args.embedding_dim
        self.heads = args.num_heads
        self.heads_dim = self.embeds_dim // self.heads
        self.alpha = args.alpha
        self.scaling_factor = math.pow(self.embeds_dim, 0.5)
        self.device = args.device
        self.dropout_rate = args.dropout
        self.forward_pass = forward_pass

        # the heads dim * heads needs to be equal to the embeds side.
        # for example, embeds = 256, heads = 8, heads_dim = 32. 256 == 32 * 8. heads = 7, would result in 256 != 7 * 36
        assert self.heads_dim * self.heads == self.embeds_dim, "heads dim * heads needs to be equal to the embeds side"

        # # this was used previously in development, same weights for all heads - mistake
        # weights are in the A matrix inside the linear layer, no need for bias
        # self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)

        # with different weights for each head
        self.queries = nn.Linear(self.embeds_dim, self.embeds_dim, bias=False)
        self.keys = nn.Linear(self.embeds_dim, self.embeds_dim, bias=False)
        self.values = nn.Linear(self.embeds_dim, self.embeds_dim, bias=False)

        # initialize weights
        torch.nn.init.xavier_normal_(self.queries.weight)
        torch.nn.init.xavier_normal_(self.keys.weight)
        torch.nn.init.xavier_normal_(self.values.weight)

        self.softmax = nn.Softmax(dim=-1)

        # after concatenating, we pass the output through a linear layer
        self.linear_weights_o = nn.Linear(self.heads_dim * self.heads, self.embeds_dim)  # w_o

        # layer normalization
        # from paper: "Layer normalization was applied to all linear projections of masked multihead attention"
        self.layer_norm1 = nn.LayerNorm(self.heads_dim)
        self.layer_norm2 = nn.LayerNorm(self.embeds_dim)
        # layer dropout
        # from paper: "We applied residual dropout ... with dropout to the output of masked multi-head attention ..."
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, queries, keys, values, mask):
        """
        Masked attention is Masked(Q, K, V) = softmax((Q * K.T ) / sqrt(d_k) + directional_mask + alpha * distance_mask)
        A masked multi head attention is Masked_multi_head_attention(O, K, V) = concat(head_1, ..., head_n) * W_o
        Where head_i = Masked(Q * W_q_i, K * W_k_i * V * W_v_i)
        h is the number of heads
        W_q_i, W_k_i, W_v_i have the shape [embedding_dim X embedding_dim/heads]
        W_o has the shape [embedding_dim X embedding_dim]
        Q, K, V have shape [batch_size X sequence_length X embedding_dim]
        :param queries: tensor
        :param keys: tensor
        :param values: tensor
        :param mask: tensor
        :return:
        """
        # get batch size
        batch_size = queries.shape[0]

        # get length of incoming sequence
        query_len, key_len, value_len = queries.shape[1], keys.shape[1], values.shape[1]

        # # this was used previously in development, same weights for all heads - mistake
        # # split embedding into self.head_dim, changing from batch*len*embedding to batch*len*heads*heads_dim
        # queries = queries.reshape(batch_size, query_len, self.heads, self.heads_dim)
        # keys = keys.reshape(batch_size, key_len, self.heads, self.heads_dim)
        # values = values.reshape(batch_size, value_len, self.heads, self.heads_dim)

        # pass through linear layer
        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        # split embedding into self.head_dim, changing from batch*len*embedding to batch*len*heads*heads_dim
        queries = queries.reshape(batch_size, query_len, self.heads, self.heads_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.heads_dim)
        values = values.reshape(batch_size, value_len, self.heads, self.heads_dim)


        # layer norm
        queries = self.layer_norm1(queries)
        keys = self.layer_norm1(keys)
        values = self.layer_norm1(values)

        # multiply queries and keys
        # queries has shape - batch(n) X query_len(q) X heads(h) X heads_dim(d)
        # keys has shape - batch(n) X key_len(k) X heads(h) X heads_dim(d)
        # energy has shape - batch(n) X heads(h) X query_len(q) X key_length(K)
        # can interpret energy as for each word in key how much to pay attention in query
        # Each row represents the attention logits for a specific element  to all other elements in the sequence
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # get directional mask and distance mask
        mask = mask.repeat(1, self.heads, 1).view(batch_size, self.heads, query_len)
        # mask.repeat(self.heads, 1, 1).view(-1, query_len, 1)
        dir_mask = self.get_directional_mask(batch_size=batch_size, seq_length=query_len, individual_mask=mask)
        dist_mask = self.get_distance_mask(seq_length=query_len, batch_size=batch_size)

        # compute masked softmax
        masked_pre_softmax = energy / self.scaling_factor + dir_mask + self.alpha * dist_mask  # input to masked softmax
        # we want along each row to equal 1, since each row represents probabilities for a word (queries X keys)
        masked_softmax = self.softmax(masked_pre_softmax)

        # compute masked multi-head attention
        # masked softmax has shape batch_size X heads X queries_length X keys_length
        # values has shape batch_size X values_length X heads X heads_dim
        # out should have shape batch_size X query_length X heads X heads_dim
        out = torch.einsum("nhql,nlhd->nlhd", [masked_softmax, values])
        # now we concat heads. i.e. flatten
        out = out.reshape(batch_size, query_len, self.heads * self.heads_dim)
        # fully connected layer out, followed by layer norm and dropout
        out = self.linear_weights_o(out)
        out = self.layer_norm2(out)
        out = self.dropout(out)

        return out

    def get_distance_mask(self, seq_length, batch_size):
        """
        from paper: "The (i, j) component of the distance mask is −|i − j|, representing the distance between (i + 1)th
        word and (j + 1)th word multiplied by −1"
        :param seq_length: int
        :param batch_size: int
        :return:
        """
        distance_mask = torch.FloatTensor(seq_length, seq_length).to(self.device)
        for i in range(seq_length):
            for j in range(seq_length):
                distance_mask[i, j] = -abs(i - j)
        distance_mask = distance_mask.unsqueeze(0)
        distance_mask = distance_mask.repeat(batch_size, self.heads, 1, 1)
        return distance_mask

    def get_directional_mask(self, batch_size, seq_length, individual_mask):
        """
        from paper: "Mdir consists of the forward mask and backward mask... The forward masks prevent words that appear
         after a given word from being considered in the attention process, while backward masks prevent words that
         appear before from consideration by adding −inf to the logits before taking the softmax at the attention phase.
          The diagonal component of Mdir is also set to −inf so that each token does not consider itself to attention.
        :param batch_size: int
        :param seq_length: int
        :param individual_mask: tensor
        :return:
        """
        # build general directional mask
        directional_mask = torch.FloatTensor(seq_length, seq_length).to(self.device)
        directional_mask.data.fill_(1)
        if self.forward_pass:
            directional_mask = torch.tril(directional_mask,
                                          diagonal=-1)  # diagonal=-1, to make main diagonal 0 (article)
            directional_mask[directional_mask == 0.] = float("-1e20")
            directional_mask[directional_mask == 1.] = float("0")
        else:
            directional_mask = torch.triu(directional_mask, diagonal=1)  # diagonal=1, to make main diagonal 0 (article)
            directional_mask[directional_mask == 0.] = float("-1e20")
            directional_mask[directional_mask == 1.] = float("0")
        directional_mask = directional_mask.unsqueeze_(0)
        directional_mask = directional_mask.repeat(batch_size, self.heads, 1, 1)

        # expand specific mask according to length of sentence
        row_mask = individual_mask.view(batch_size, self.heads, seq_length, 1)
        col_mask = individual_mask.view(batch_size, self.heads, 1, seq_length)
        matrix_individual_mask = torch.mul(row_mask, col_mask)
        # shape is batch_size X heads X seq_length X seq_length, seq_length is max length in batch

        # combine both masks, to get diagonal in proper place
        final_mask = matrix_individual_mask * directional_mask
        return final_mask


class FusionGateAndNorm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embeds_dim = args.embedding_dim
        self.device = args.device
        self.dropout_rate = args.dropout

        self.linear_weights_s = nn.Linear(self.embeds_dim, self.embeds_dim, bias=False)  # no need to mention batch_size
        self.linear_weights_h = nn.Linear(self.embeds_dim, self.embeds_dim, bias=False)
        self.b_f = nn.Parameter(torch.FloatTensor(self.embeds_dim))
        torch.nn.init.xavier_normal_(self.linear_weights_s.weight)
        torch.nn.init.xavier_normal_(self.linear_weights_h.weight)
        torch.nn.init.constant_(self.b_f, 0)  # added after getting nans at times
        self.sigmoid = nn.Sigmoid()

        # layer norm
        # from paper: "Layer normalization was applied to all linear projections of .... fusion gate..."
        self.layer_norm = nn.LayerNorm(self.embeds_dim)
        # layer dropout
        # from paper: "We applied residual dropout ... with dropout to Sf + Hf + bf of fusion gate."
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, s, h):
        """
        At the fusion gate, raw word embedding S (batch_size X seq_len X embedding_dim) and the results of the
        masked-multi-head H (batch_size X seq_len X embedding_dim) are used as input.
        First we generate Sf, Hf by projecting S, H using Ws, Wh (batch_size X embedding_dim X embedding_dim).
        Mathematically Sf = S * Ws, Hf = H * Wh
        Then create gate F Gate(S,H) = F o Sf + (1-F) o Hf. (F = sigmoid(Sf + Hf + bf, bf (batch_size X embedding_dim)
        Finally, we obtain the gated sum by using F. It is common in many papers to use raw S and H in gated sum. We,
        however, use the gated sum of Sf and Hf which resulted in a significant increase in accuracy
        :param s: tensor
        :param h: tensor
        :return:
        """
        # generating Sf and Hf, following by layer norm (since linear_weights is a linear projection)
        s_f = self.layer_norm(self.linear_weights_s(s))
        h_f = self.layer_norm(self.linear_weights_h(h))

        # generating F
        f = self.sigmoid(self.dropout(s_f + h_f + self.b_f))
        out = f * s_f + (1 - f) * h_f

        # adding for better outcome- not sure if in paper
        out = self.layer_norm(out)

        return out


class PositionWiseAddAndNorm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embeds_dim = args.embedding_dim
        self.device = args.device
        self.d_ff = args.d_ff
        self.dropout_rate = args.dropout

        self.linear_weights_p1 = nn.Linear(self.embeds_dim, self.d_ff, bias=True)  # the bias b1p is built in, bias=True
        self.linear_weights_p2 = nn.Linear(self.d_ff, self.embeds_dim, bias=True)  # the bias b1p is built in, bias=True
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.embeds_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):
        """
        The position - wise feed forward network employs the same fully connected network to each position of
        sentence, in which the fully connected layer consists of two linear transformations, with the ReLU activation
        in between.
        Mathematically: FNN(x) = max(0,x * Wp1 + bp1) * Wp2 + bp2
        Dimensions- X: (batch_size X seq_len X embedding_dim), Wp1 (batch_size X embedding_dim X d_ff),
        Wp2 (batch_size X d_ff X embedding_dim), bp1 (batch_size X d_ff), bp2 (batch_size X embedding_dim)
        Note that position - wise feed forward network is combined with the residual connection. That is, FFN learns
        the residuals.
        :param inputs: tensor
        :return:
        """
        out = self.linear_weights_p1(inputs)
        out = self.relu(out)
        out = self.linear_weights_p2(out)
        out = self.dropout(out)  # not sure if in paper, trying to improve
        # add residual connection and norm
        out = self.layer_norm(out + inputs)
        return out


class MultiDimensionalAttentionAndMaxPooling(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embeds_dim = args.embedding_dim

        self.linear_weights_m1 = nn.Linear(2 * self.embeds_dim, 2 * self.embeds_dim, bias=True)  # bias is b21
        self.linear_weights_m2 = nn.Linear(2 * self.embeds_dim, 2 * self.embeds_dim, bias=True)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)  # the softmax is on the row dimensions, not the column

        # layer norm
        # from paper: "Layer normalization was applied to all linear projections of .... fusion gate..."
        self.layer_norm = nn.LayerNorm(2 * self.embeds_dim)

    def forward(self, inputs, seq_mask):
        """
        The input of pooling layer is U =[Ufw;Ubw]  [batch_size X seq_len X 2 * embedding_dim] where each directional
        self attention output is Ufw [batch_size X seq_len X embedding_dim], Ubw [batch_size X seq_len X embedding_dim].
        This section has two parts:
        1. Multidimensional attention
               from paper: "We use the multidimensional source2token self-attention"
               For each row ui of U, we compute l(ui) = ELU(ui * Wm1 +bm1) * Wm2 + bm2
               Dimensions:
               ui (batch_size X 1 X 2*embedding_dim), Wm1 & Wm2 (batch_size X 2 * embedding_dim X 2 * embedding_dim),
               bm1 & bm2 (batch_size X 2 * embedding_dim)
               M = softmax (L) * U, where L is a vector of [l(u1), l(u2) ... l(un)]. from paper: "Note that softmax is
               performed on the row dimension of L, not the column dimension".
               Note: Softmax should not include sequence padding
               So Multidimensional(U) = sum(M)
        2. Max pooling
        The output of the multidimensional attention and the output of the max pooling are concatenated to encode the
        input sentence. The result has a 4 * embedding_dim shape. (concatenating 2 * embedding_dim and 2 *embedding_dim)
        :param seq_mask: tensor
        :param inputs: tensor
        :return:
        """
        batch_size, seq_len, _ = inputs.shape

        # Max pooling
        pooling = nn.MaxPool2d((seq_len, 1), stride=1)
        pool_output = pooling(inputs * seq_mask).view(batch_size, -1)

        # Multidimensional attention
        output = self.layer_norm(self.linear_weights_m1(inputs))
        output = self.elu(output)
        output = self.layer_norm(self.linear_weights_m2(output))
        # seq_mask[seq_mask == 0.] = float("-1e20")  # so will get ery small value after softmax
        masked_output = output * seq_mask
        masked_output[masked_output == 0.] = float("-1e20")
        output = self.softmax(masked_output)
        output = torch.sum(torch.mul(inputs, output), dim=1)

        # concatenate
        output = torch.cat([output, pool_output], dim=-1)  # attach the rows
        return output
