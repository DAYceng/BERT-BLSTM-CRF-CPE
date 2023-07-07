import torch
import torch.nn as nn
from torch.autograd import Variable


from transformers import BertPreTrainedModel, BertModel
# from torchcrf import CRF


from utils import *

START_TAG = "<START>"
STOP_TAG = "<STOP>"


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(smat):
    """
    参数: smat 是 "status matrix", DP状态矩阵; 其中 smat[i][j] 表示 上一帧为i状态且当前帧为j状态的分值
    作用: 针对输入的【二维数组的每一列】, 各元素分别取exp之后求和再取log; 物理意义: 当前帧到达每个状态的分值(综合所有来源)
    例如: smat = [[ 1  3  9]
                 [ 2  9  1]
                 [ 3  4  7]]
         其中 smat[:, 2]= [9,1,7] 表示当前帧到达状态"2"有三种可能的来源, 分别来自上一帧的状态0,1,2
         这三条路径的分值求和按照log_sum_exp法则，展开 log_sum_exp(9,1,7) = log(exp(9) + exp(1) + exp(7)) = 3.964
         所以，综合考虑所有可能的来源路径，【当前帧到达状态"2"的总分值】为 3.964
         前两列类似处理，得到一个行向量作为结果返回 [ [?, ?, 3.964] ]
    注意数值稳定性技巧 e.g. 假设某一列中有个很大的数
    输入的一列 = [1, 999, 4]
    输出     = log(exp(1) + exp(999) + exp(4)) # 【直接计算会遭遇 exp(999) = INF 上溢问题】
            = log(exp(1-999)*exp(999) + exp(999-999)*exp(999) + exp(4-999)*exp(999)) # 每个元素先乘后除 exp(999)
            = log([exp(1-999) + exp(999-999) + exp(4-999)] * exp(999)) # 提取公因式 exp(999)
            = log([exp(1-999) + exp(999-999) + exp(4-999)]) + log(exp(999)) # log乘法拆解成加法
            = log([exp(1-999) + exp(999-999) + exp(4-999)]) + 999 # 此处exp(?)内部都是非正数，不会发生上溢
            = log([exp(smat[0]-vmax) + exp(smat[1]-vmax) + exp(smat[2]-vmax)]) + vmax # 符号化表达
    代码只有两行，但是涉及二维张量的变形有点晦涩，逐步分析如下, 例如:
    smat = [[ 1  3  9]
            [ 2  9  1]
            [ 3  4  7]]
    smat.max(dim=0, keepdim=True) 是指【找到各列的max】，即: vmax = [[ 3  9  9]] 是个行向量
    然后 smat-vmax, 两者形状分别是 (3,3) 和 (1,3), 相减会广播(vmax广播复制为3*3矩阵)，得到:
    smat - vmax（max_score） = [[ -2  -6  0 ]
                   [ -1  0   -8]
                   [ 0   -5  -2]]
    然后.exp()是逐元素求指数
    然后.sum(axis=0, keepdim=True) 是"sum over axis 0"，即【逐列求和】, 得到的是行向量，shape=(1,3)
    然后.log()是逐元素求对数
    最后再加上vmax（max_score）; 两个行向量相加, 结果还是个行向量
    """
    max_score = smat.max(dim=0, keepdim=True).values
    return (smat - max_score).exp().sum(axis=0, keepdim=True).log() + max_score


class BiLSTM_CRF(nn.Module):#nn.ModuleBertPreTrainedModel

    def __init__(
        self,
        vocab_size,
        tag_to_ix,
        embedding_dim, # 100
        hidden_dim,  # 200
        char_lstm_dim=25,
        char_to_ix=None,
        pre_word_embeds=None,
        char_embedding_dim=25,
        use_gpu=False,
        n_cap=None,
        cap_embedding_dim=None,
        use_crf=True,
        char_mode="CNN",


    ):
        # super(BERT_BiLSTM_CRF, config, self).__init__(config)
        super(BiLSTM_CRF, self).__init__()

        self.use_gpu = use_gpu
        # 开启self.device = device报错 AttributeError: can't set attribute
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.embedding_dim = embedding_dim #词嵌入维度
        self.hidden_dim = hidden_dim  #LSTM隐藏层维度

        self.vocab_size = vocab_size  # 词（向量）长度
        self.tag_to_ix = tag_to_ix  # 给每个IOB标签的编号
        '''
            注：n_cap与cap_embedding_dim未启用
        '''
        self.n_cap = n_cap  # 大写特征的编号Capitalization feature num
        self.cap_embedding_dim = cap_embedding_dim  # Capitalization feature dim
        self.use_crf = use_crf # 是否使用crf来优化优化输出结果
        self.tagset_size = len(tag_to_ix) # 获取标签数量
        self.out_channels = char_lstm_dim #
        self.char_mode = char_mode # 选择处理字符特征的模型（CNN和LSTM可选）

        print("char_mode: %s, out_channels: %d, hidden_dim: %d, " % (char_mode, char_lstm_dim, hidden_dim))

        # self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # out_dim = config.hidden_size
        #
        # if self.use_bert:pass


        # 若定义了大小写特征则生成大小写特征嵌入
        if self.n_cap and self.cap_embedding_dim:
            '''
                大小写特征层结构：
                n_cap：获取大小写特征的编号
                cap_embedding_dim：获取定义的大小写特征嵌入维度数值
            '''
            self.cap_embeds = nn.Embedding(self.n_cap, self.cap_embedding_dim)
            torch.nn.init.xavier_uniform_(self.cap_embeds.weight)
            '''
                为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等，
                Xavier作为一种初始化方法，就是用来实现一种均匀分布，以达到上述目的
            '''

        # 若定义了字符嵌入维度则生成字符嵌入
        if char_embedding_dim is not None:
            # 若使用LSTM处理字符嵌入特征，则以下是该层的维数定义
            self.char_lstm_dim = char_lstm_dim
            # 定义字符嵌入的大小
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            torch.nn.init.xavier_uniform_(self.char_embeds.weight)
            if self.char_mode == "LSTM":
                self.char_lstm = nn.LSTM(self.char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
                init_lstm(self.char_lstm)
            if self.char_mode == "CNN":
                self.char_cnn3 = nn.Conv2d(
                    in_channels=1,
                    out_channels=self.out_channels,
                    kernel_size=(3, char_embedding_dim),
                    padding=(2, 0),
                )

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
            # 是否使用预训练词嵌入
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(0.5)
        """
            1、Bert:
                self.lstm = nn.LSTM(embedding_dim + self.out_channels, hidden_dim, bidirectional=True)
                Args:
                    embedding_dim + self.out_channels:  
                    hidden_dim: 
                释义:
            
            """
        # self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # out_dim = config.hidden_size # 768



        """
            2、BiLSTM:
                self.lstm = nn.LSTM(embedding_dim + self.out_channels, hidden_dim, bidirectional=True)
                Args:
                    embedding_dim + self.out_channels:  
                    hidden_dim: 
                释义:
            
            """
        if self.n_cap and self.cap_embedding_dim:
            if self.char_mode == "LSTM":
                self.lstm = nn.LSTM(
                    embedding_dim + char_lstm_dim * 2 + cap_embedding_dim,
                    hidden_dim,
                    bidirectional=True,
                )
            if self.char_mode == "CNN":
                self.lstm = nn.LSTM(
                    embedding_dim + self.out_channels + cap_embedding_dim,
                    hidden_dim,
                    bidirectional=True,
                )
        else:

            if self.char_mode == "LSTM":
            # 特征维数（此项来那个维数）：embedding_dim + char_lstm_dim * 2，150
            # LSTM隐藏层维度：hidden_dim ，200
                self.lstm = nn.LSTM(embedding_dim + char_lstm_dim * 2, hidden_dim, bidirectional=True)
            # 当字符特征以CNN模式处理时，之后的BLSTM层参数如下
            # 特征维数（此项来那个维数）：embedding_dim + self.out_channels，125
            # LSTM隐藏层维度：hidden_dim ，200
            if self.char_mode == "CNN":
                # LSTM(125, 200, bidirectional=True)
                self.lstm = nn.LSTM(embedding_dim + self.out_channels, hidden_dim, bidirectional=True)
        #         # 将BERT的输出层（768维，通过修改配置降低成125维，）作为LSTM的输入
        #         self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=hidden_dim, bidirectional=True)
        # init_lstm(self.lstm)
        # # high way 高速网络，用于解决网络过于复杂的问题
        # self.hw_trans = nn.Linear(self.out_channels, self.out_channels)
        # self.hw_gate = nn.Linear(self.out_channels, self.out_channels)
        # self.h2_h1 = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.tanh = nn.Tanh()
        # init_linear(self.h2_h1)
        # init_linear(self.hw_gate)
        # init_linear(self.hw_trans)

        # 设置全连接层，实现标签分类
        """
        全连接层:
            self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)
            Args:
                hidden_dim:  
                tagset_size: 

            释义:
                
        """
        #Linear(in_features=400, out_features=10, bias=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)
        init_linear(self.hidden2tag)

        """
        CRF层:
            释义:  
        """
        if self.use_crf:
            # torch.randn生成一个tensor，形状由两个输入参数控制，内部元素为满足正态分布的随机数
            # 此处生成一个以标签尺寸为长宽的tensor作为模型训练参数
            # 例如有十个IOB标签，那么这个tensor就是10x10的大小，里面是随机数
            self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
            self.transitions.data[:, tag_to_ix[START_TAG]] = -10000
            self.transitions.data[tag_to_ix[STOP_TAG], :] = -10000

    def _score_sentence(self, feats, tags):
        # 作为crf计算的分子对数
        # Gives the score of a provided tag sequence
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        """
        求路径pair: feats->tags 的分值
        index:      0   1   2   3   4   5   6   7   8
        feats:     F0  F1  F2  F3  F4  F5  F6
        tags:  <s>  'O', 'S-VENDOR', 'B-PRODUCT', 'E-PRODUCT', 'S-PRODUCT', 'I-PRODUCT', 'B-VENDOR', 'E-VENDOR', '<START>', '<STOP>'}  <e>
        """
        r = torch.LongTensor(range(feats.size()[0])).to(self.device) #self.device
        pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]).to(self.device), tags]) #self.device
        pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]]).to(self.device)]) #self.device

        score = torch.sum(self.transitions[pad_start_tags, pad_stop_tags]) + torch.sum(feats[r, tags])
        return score

    def _get_lstm_features(self, sentence, chars, caps, chars2_length, d):
        '''
        # 求出每一个feats对应的隐向量，即求发射矩阵
        :param sentence:
        :param chars:
        :param caps:
        :param chars2_length:
        :param d:
        :return:
        '''

        if self.char_mode == "LSTM":
            # self.char_lstm_hidden = self.init_lstm_hidden(dim=self.char_lstm_dim, bidirection=True, batchsize=chars.size(0))
            chars_embeds = self.char_embeds(chars).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            lstm_out, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros(
                (outputs.size(0), outputs.size(2))))).to(self.device) #self.device
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((
                    outputs[i, index - 1, :self.char_lstm_dim],
                    outputs[i, 0, self.char_lstm_dim:],
                ))
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        if self.char_mode == "CNN":
            chars_embeds = self.char_embeds(chars).unsqueeze(1)
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                    kernel_size=(chars_cnn_out3.size(2),
                                                                 1)).view(chars_cnn_out3.size(0), self.out_channels)

        # t = self.hw_gate(chars_embeds) # high way
        # g = nn.functional.sigmoid(t)
        # h = nn.functional.relu(self.hw_trans(chars_embeds))
        # chars_embeds = g * h + (1 - g) * chars_embeds

        #concatenate层（连接层），使用torch.cat实现
        embeds = self.word_embeds(sentence)
        if self.n_cap and self.cap_embedding_dim:
            cap_embedding = self.cap_embeds(caps)
            embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
        else:
            embeds = torch.cat((embeds, chars_embeds), 1)


        embeds = embeds.unsqueeze(1) # 扩充维度
        embeds = self.dropout(embeds)  # dropout
        lstm_out, _ = self.lstm(embeds)

        # 规定LSTM输出形状(seq_len, batch=1, input_size)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)# 将高维tensor展开为一维（用于linear层输入等）
        lstm_out = self.dropout(lstm_out)  # dropout
        lstm_feats = self.hidden2tag(lstm_out) # 把LSTM输出的隐状态张量去掉batch维，然后降维到tag空间
        return lstm_feats

    def _forward_alg(self, feats):
        # 利用随机的转移矩阵，通过前向传播计算出一个score，作为crf计算的分母对数
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        '''给定每个feats的发射分值; 按照当前的CRF层参数算出所有可能序列的分值和，用作概率归一化分母'''
        alpha = torch.full((1, self.tagset_size), -10000.0, device=self.device) #self.device
        alpha[0][self.tag_to_ix[START_TAG]] = 0.0
        for feat in feats:
            # log_sum_exp()内三者相加会广播: 当前各状态的分值分布(列向量) + 发射分值(行向量) + 转移矩阵(方形矩阵)
            # 相加所得矩阵的物理意义见log_sum_exp()函数的注释; 然后按列求log_sum_exp得到行向量
            alpha = log_sum_exp(alpha.T + feat.unsqueeze(0) + self.transitions)
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        return log_sum_exp(alpha.T + 0 + self.transitions[:, [self.tag_to_ix[STOP_TAG]]]).flatten()[0]

    def viterbi_decode(self, feats):
        # 维特比解码，实际上在预测的时候使用，输出得分与路径值
        backtrace = [] # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是什么状态
        alpha = torch.full((1, self.tagset_size), -10000.0, device=self.device) #self.device
        alpha[0][self.tag_to_ix[START_TAG]] = 0
        for feat in feats:
            # 这里跟 _forward_alg()稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            smat = (alpha.T + feat.unsqueeze(0) + self.transitions)  # (tagset_size, tagset_size)，当前feat每个状态的最优"来源"
            backtrace.append(smat.argmax(0))  # column_max
            alpha = log_sum_exp(smat) # 转移规律跟 _forward_alg()一样; 只不过转移之前拿smat求了一下回溯路径
        # 回溯路径
        smat = alpha.T + 0 + self.transitions[:, [self.tag_to_ix[STOP_TAG]]]
        best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):  # ignore START_TAG，从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return log_sum_exp(smat).item(), best_path[::-1]  # item() return list? 返回最优路径分值 和 最优路径

    def neg_log_likelihood(self, sentence, tags, chars, caps, chars2_length, d):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        # 实际上这个就是损失函数
        '''
        求一对 <sentence, tags> 在当前参数下的负对数似然，作为loss
        :param sentence:
        :param chars:
        :param caps:
        :param chars2_length:
        :param d:
        :return:
        '''
        # 获取每个feats的分数
        feats = self._get_lstm_features(sentence, chars, caps, chars2_length, d)

        if self.use_crf:
            forward_score = self._forward_alg(feats) # 所有路径的分数和
            gold_score = self._score_sentence(feats, tags) # 正确路径的分数
        # -(正确路径的分数 - 所有路径的分数和）;注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores

    # forward是前向传播函数,也是模型做出推断的地方
    # 在所有的子类中都需要重写这个函数
    def forward(self, sentence, chars, caps, chars2_length, d):
        # 获取LSTM层的输出结果，feats，即求出每一帧（每个feats）的发射矩阵
        feats = self._get_lstm_features(sentence, chars, caps, chars2_length, d)
        # viterbi to get tag_seq

        if self.use_crf:
            # 如果使用CRF，则用训练过的CRF层来做维特比解码，得到最优路径及其分数
            score, tag_seq = self.viterbi_decode(feats)
        else:
            # 如果不使用CRF，则用直接选择发射分数大标签作为预测结果
            score, tag_seq = torch.max(feats, 1) # 输出feat这个tensor中每列的最大值
            tag_seq = tag_seq.cpu().tolist()

        return score, tag_seq