# -*-coding:utf-8 -*-

'''
@File       : beam_search.py
@Author     : HW Shen
@Date       : 2020/8/12
@Desc       :
'''

import numpy as np
import tensorflow as tf

BATCH_SIZE = 1  # 每个batch数据量
BEAM_SIZE = 2  # beam_search的尺度
EMB_DIM = 10  # 词向量维度
ENCODER_UNITS = 20  # encoder层的单位数
DECODER_UNITS = 20  # decoder层的单位数
ATTENTION_UNITS = 20  # attention层的单位数
MAX_LEN = 10  # Decoding输出的最大长度
MIN_LEN = 1   # Decoding输出的最小长度
START_TOKEN = 'START'  # Decoder过程的开始标识符
END_TOKEN = 'END'  # Deocder过程的结束标识符
UNK_TOKEN = 'UNK'  # OOV

word2id = {'START': 0, 'END': 1, 'PAD': 2, '我': 3, '你': 4, '洗澡': 5, '吃饭': 6, 'UNK': 7}
id2word = ['START', 'END', 'PAD', '我', '你', '洗澡', '吃饭', 'UNK']


class Encoder(tf.keras.layers.Layer):
    """编码器 : 双层GRU """

    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):

        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units // 2  # 因为是双向的，所以每一层均分encoder_units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(units=self.encoder_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bi_gru = tf.keras.layers.Bidirectional(layer=self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(value=hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bi_gru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)

        return output, state

    def initialize_hidden_state(self):
        # encoder隐层初始化参数
        return tf.zeros((self.batch_size, self.encoder_units * 2))  # shape:(batch_size, encoder_units)


class Attention(tf.keras.layers.Layer):
    """attention 类"""

    def __init__(self, units):
        super(Attention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_output):

        hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)  # 给dec_hidden增加一个时间维度
        score = self.V(tf.nn.tanh(self.W_s(encoder_output) + self.W_h(hidden_with_time_axis)))  # 注意力得分
        attn_dist = tf.nn.softmax(score, axis=1)  # 注意力得分归一化
        context_vector = attn_dist * encoder_output  # 输入词向量的上下文向量
        context_vector = tf.reduce_sum(context_vector, axis=1)  # 上下文向量的求和，降维

        return context_vector, tf.squeeze(attn_dist, -1)


class Decoder(tf.keras.layers.Layer):
    """ 解码层 ： 单层 GRU + Dense"""

    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units  # Decoder是单层GRU
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(units=self.decoder_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(units=vocab_size, activation=tf.keras.activations.softmax)

    def call(self, x, hidden, enc_output, context_vector):
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        out = self.fc(output)
        return x, out, state


class Seq2SeqModel(object):
    """seq2seq 类"""

    def __init__(self, encoder, decoder, attention):

        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

    def call_encoder(self, encoder_input):

        encoder_hidden = self.encoder.initialize_hidden_state()  # 初始化隐层参数矩阵
        encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)  # 编码器的 output 和 hidden_state

        return encoder_output, encoder_hidden

    def decode_one_step(self, latest_tokens, encoder_states, decoder_in_states):
        """
        Decoder层的每一步解码
        Args:
            latest_tokens: 当前时间步, 解码器输入的 token
            encoder_states: 编码器输出的 output
            decoder_in_states: 解码器上一时间步传来的隐层向量
        Returns:
            top_k_ids: [beam, beam * 2], 解码结果中 top 2 * beam 的 token id
            top_k_log_probs: [beam, beam * 2], 解码过程中 top 2 * beam 的 log 概率得分
            new_states: list, 每个 item 是每个 beam 中的隐层向量
        """
        # latest_tokens = np.transpose(np.array([latest_tokens]))
        # decoder_in_states = np.concatenate(arrays=decoder_in_states, axis=0)
        latest_tokens = tf.transpose([latest_tokens])
        decoder_in_states = tf.concat(values=decoder_in_states, axis=0)
        context_vector, attn_dists = self.attention(decoder_in_states, encoder_states)
        _, prediction, decoder_hidden = self.decoder(latest_tokens, decoder_in_states, encoder_states, context_vector)
        # 把上一步预测的结果prediction作为topK的选依据
        top_k_log_probs, top_k_ids = tf.nn.top_k(input=prediction, k=BEAM_SIZE * 2, sorted=True)
        top_k_log_probs = tf.math.log(x=top_k_log_probs)
        new_states = [np.expand_dims(decoder_hidden[i, :], axis=0) for i in range(BEAM_SIZE)]

        return top_k_ids.numpy(), top_k_log_probs, new_states, attn_dists


class BeamHypotheses(object):
    """ BeamSearch的假设 """

    def __init__(self, token_list, log_prob_list, state, attn_dist_list):

        self.token_list = token_list  # list of all the tokens from time 0 to the current time step t
        self.log_prob_list = log_prob_list  # list of the log probabilities of the tokens
        self.state = state  # decoder state after the last token decoding
        self.attn_dist_list = attn_dist_list  # attention dists of all the tokens

    def extend(self, token, log_prob, state, attn_dist):
        """
        Method to extend the current hypothesis by adding the next decoded token
        Args:
            token: the next decoded token
            log_prob: the log prob of the next decoded token
            state: next hidden state
            attn_dist: # the attention dist of the next decoded toke
        Returns: next hypothesis
        """
        return BeamHypotheses(
            token_list=self.token_list + [token],
            log_prob_list=self.log_prob_list + [log_prob],
            state=state,
            attn_dist_list=self.attn_dist_list + [attn_dist]
        )

    @property
    def latest_token(self):
        return self.token_list[-1]

    @property
    def log_prob(self):
        return sum(self.log_prob_list)

    @property
    def avg_log_prob(self):
        return self.log_prob / len(self.token_list)


class Generation(object):
    """ 按照beam_search的方式 生成Decode的相应结果 """

    def __init__(self):
        pass

    @staticmethod
    def sort_hyp(hyp_list):
        # 对所有假设从大到小排序（按照平均概率值）
        return sorted(hyp_list, key=lambda h: h.avg_log_prob, reverse=True)

    def generate_beam_search(self, model, model_input):

        # encoder states bi-gru 的 output; decoder in state 是拼接的隐层向量
        encoder_states, decoder_in_state = model.call_encoder(model_input)
        hyp_list = [
            BeamHypotheses(token_list=[word2id[START_TOKEN]],
                           log_prob_list=[0.0],
                           state=decoder_in_state,
                           attn_dist_list=[],
                           ) for _ in range(BEAM_SIZE)
        ]
        result = []
        step = 0

        while step < MAX_LEN and len(result) < BEAM_SIZE:
            # 容器里每个 hpy 的最后一个 token 的列表
            latest_tokens = [hyp.latest_token for hyp in hyp_list]
            # 处理异常, 搞成 unk 的 id
            latest_tokens = [token if token in range(len(id2word)) else word2id[UNK_TOKEN] for token in latest_tokens]
            # 容器里的 hpy 在解码器得到 state 的列表
            states = [hyp.state for hyp in hyp_list]

            top_k_ids, top_k_log_probs, new_states, attn_dists = model.decode_one_step(latest_tokens=latest_tokens,
                                                                                       encoder_states=encoder_states,
                                                                                       decoder_in_states=states)
            # extend 操作后, 都收集在 all_hyp_list 中
            all_hyp_list = []
            # 初始状态开始 beam search 时, 实际只有一种状态; 但在后面的 step 时, 每次都有 beam 个状态
            num_ori_hyp = 1 if step == 0 else len(hyp_list)

            for i in range(num_ori_hyp):
                hyp = hyp_list[i]
                new_state = new_states[i]
                attn_dist = attn_dists[i]

                for j in range(BEAM_SIZE * 2):
                    new_hpy = hyp.extend(token=top_k_ids[i, j],
                                         log_prob=top_k_log_probs[i, j],
                                         state=new_state,
                                         attn_dist=attn_dist)
                    all_hyp_list.append(new_hpy)

            # 对 hpy 进行排序, 只有满足了 END TOKEN 和 MIN LEN 才会加入result
            hyp_list = []
            for hyp in self.sort_hyp(all_hyp_list):
                # 出现终止符号
                if hyp.latest_token == word2id[END_TOKEN]:
                    if step >= MIN_LEN:
                        result.append(hyp)
                else:
                    hyp_list.append(hyp)
                if len(hyp_list) == BEAM_SIZE or len(result) == BEAM_SIZE:
                    break

            step += 1

        # 如果循环都结束了, 但都仍然没有 END 出现, result 就是一个空 hyp_list, 此时要把此时的 hyp list 进行排序取最大的输出
        if len(result) == 0:
            result = hyp_list
        hyp_sorted = self.sort_hyp(result)

        return ' '.join(id2word[index] for index in hyp_sorted[0].token_list)


if __name__ == '__main__':
    model_obj = Seq2SeqModel(encoder=Encoder(vocab_size=len(word2id),
                                             embedding_dim=EMB_DIM,
                                             encoder_units=ENCODER_UNITS,
                                             batch_size=BATCH_SIZE),
                             decoder=Decoder(vocab_size=len(word2id),
                                             embedding_dim=EMB_DIM,
                                             decoder_units=DECODER_UNITS,
                                             batch_size=BATCH_SIZE),
                             attention=Attention(units=ATTENTION_UNITS))
    generator_obj = Generation()
    res = generator_obj.generate_beam_search(model_obj, np.array([[0, 3, 5, 5, 5]]))
    print(res)