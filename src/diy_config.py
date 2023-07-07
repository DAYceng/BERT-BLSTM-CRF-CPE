# -*- coding: utf-8 -*-
# @description:
# @author:
# @time:
# @file: diy_config.py

import datetime
import os
import threading,optparse
import torch
from collections import OrderedDict



class BLSTM_Config(object):
    #使用锁防止交替打印出错
    _instance_lock = threading.Lock()
    _init_flag = False
    def __init__(self):
        if not BLSTM_Config._init_flag:
            BLSTM_Config._init_flag = True
            root_path = str(os.getcwd()).replace("\\", "/")
            if 'source' in root_path.split('/'):
                self.base_path = os.path.abspath(os.path.join(os.path.pardir))
            else:
                self.base_path = os.path.abspath(os.path.join(os.getcwd()))
            self._init_train_config()
    def __new__(cls, *args, **kwargs):
        """
        单例类，判断属性是否存在
        :param args:
        :param kwargs:
        :return:
        """
        if not hasattr(BLSTM_Config, '_instance'):
            with BLSTM_Config._instance_lock:
                if not hasattr(BLSTM_Config, '_instance'):
                    BLSTM_Config._instance = object.__new__(cls)
        return BLSTM_Config._instance

    def _init_train_config(self):

        # 使用optparse模块创建一个OptionParser对象，生成使用和帮助信息
        # 添加新参数词条
        optparser = optparse.OptionParser()
        optparser.add_option("-T", "--train", default="data/eng.train", help="Train set location")
        optparser.add_option("-d", "--dev", default="data/eng.testa", help="Dev set location")
        optparser.add_option("-t", "--test", default="data/eng.testb", help="Test set location")
        optparser.add_option("--test_train", default="data/eng.train50000", help="test train")
        optparser.add_option("--score", default="evaluation/temp/score.txt", help="score file location")
        optparser.add_option("-s", "--tag_scheme", default="iob", help="Tagging scheme (IOB or IOBES)")
        optparser.add_option(
            "-l",
            "--lower",
            default="1",
            type="int",
            help="Lowercase words (this will not affect character inputs)",
        )
        optparser.add_option("-z", "--zeros", default="0", type="int", help="Replace digits with 0")
        optparser.add_option("-q", "--tf_idf", default="0", type="int", help="TF IDF feature")
        optparser.add_option("-c", "--char_dim", default="25", type="int", help="Char embedding dimension")
        optparser.add_option(
            "-C",
            "--char_lstm_dim",
            default="25",
            type="int",
            help="Char LSTM hidden layer size",
        )
        # 获取bert模型位置
        optparser.add_option(
            "--bert_model_path",
            default="/bert-base-cased",
            help="使用bert-base-cased",
        )
        optparser.add_option(
            "-b",
            "--char_bidirect",
            default="1",
            type="int",
            help="Use a bidirectional LSTM for chars",
        )
        optparser.add_option("-w", "--word_dim", default="100", type="int", help="Token embedding dimension")
        optparser.add_option(
            "-W",
            "--word_lstm_dim",
            default="200",
            type="int",
            help="Token LSTM hidden layer size",
        )
        optparser.add_option(
            "-B",
            "--word_bidirect",
            default="1",
            type="int",
            help="Use a bidirectional LSTM for words",
        )
        optparser.add_option(
            "-p",
            "--pre_emb",
            default="data/glove.6B.100d.txt",
            help="Location of pretrained embeddings",
        )
        optparser.add_option("-A", "--all_emb", default="1", type="int", help="Load all embeddings")
        optparser.add_option(
            "-a",
            "--cap_dim",
            default="0",
            type="int",
            help="Capitalization feature dimension (0 to disable)",
        )
        optparser.add_option("-f", "--crf", default="1", type="int", help="Use CRF (0 to disable)")
        optparser.add_option(
            "-D",
            "--dropout",
            default="0.5",
            type="float",
            help="Droupout on the input (0 = no dropout)",
        )
        optparser.add_option("-r", "--reload", default="0", type="int", help="Reload the last saved model")
        optparser.add_option("-g", "--use_gpu", default="1", type="int", help="whether or not to ues gpu")
        optparser.add_option("--loss", default="loss.txt", help="loss file location")
        optparser.add_option("--name", default="cve_ner_models", help="model name")
        optparser.add_option("--char_mode", choices=["CNN", "LSTM"], default="CNN", help="char_CNN or char_LSTM")

        # bert相关参数
        optparser.add_option("--do_lower_case", default="0", type="int", help="是否在分词时做全小写处理")

        opts = optparser.parse_args()[0]

        # 设置已存在的参数的值
        # 新增参数时可不添加parameters，但是一定要先optparser.add_option将参数加入参数字典
        '''
        使用区别：
        添加parameters -> name = parameters["name"]
        仅添加add_option -> name = opts.name
        '''
        parameters = OrderedDict()
        parameters["tag_scheme"] = opts.tag_scheme#设置标记方案，默认为“IOBE",即NER中的实体标签
        parameters["lower"] = opts.lower == 1 #把单词全部小写
        parameters["zeros"] = opts.zeros == 1 #用0代替单词中的数字
        parameters["tf_idf"] = opts.tf_idf == 1
        parameters["char_dim"] = opts.char_dim#设置字符嵌入的维度，默认25维
        parameters["char_lstm_dim"] = opts.char_lstm_dim#设置LSTM隐藏层大小，默认为25,5X5

        parameters["char_bidirect"] = opts.char_bidirect == 1 #对字符使用双向 LSTM
        parameters["word_dim"] = opts.word_dim#标记嵌入的维度，默认为100
        parameters["word_lstm_dim"] = opts.word_lstm_dim#标记LSTM的大小
        parameters["word_bidirect"] = opts.word_bidirect == 1 #对单词使用双向 LSTM
        parameters["pre_emb"] = opts.pre_emb #预训练嵌入文件位置
        parameters["all_emb"] = opts.all_emb == 1 #加载所有的嵌入
        parameters["cap_dim"] = opts.cap_dim #将维度特征大写（0 表示禁用）
        parameters["crf"] = opts.crf == 1 #使用CRF（0 表示禁用）
        parameters["dropout"] = opts.dropout #输入时使用dropout，dropout解释：https://blog.csdn.net/program_developer/article/details/80737724
        parameters["reload"] = opts.reload == 1 #加载上次保存的模型
        parameters["name"] = opts.name #模型名称
        # parameters["char_mode"] = opts.char_mode#选择使用CNN还是LSTM训练char
        parameters["use_gpu"] = opts.use_gpu == 1 and torch.cuda.is_available()
        use_gpu = parameters["use_gpu"]

        self.switch_device = torch.device("cuda" if use_gpu else"cpu") #if use_gpu else"cpu"


        # name = parameters["name"]

        '''
        train_file:训练集
        dev_file:交叉验证集
        test_file:测试集
        test_train_file:评估测试集
        pre_emb_file:预训练嵌入
        '''
        # self.train_file = os.path.join(self.base_path, 'data', 'eng.train') # 'G:\\CVE\\bert-blstm-crf-CPE\\src\\src\\data\\eng.train'
        # self.dev_file = os.path.join(self.base_path, 'data', 'eng.testa')
        # self.test_file = os.path.join(self.base_path, 'data', 'eng.testb')

        self.train_file = opts.train
        self.dev_file = opts.dev
        self.test_file = opts.test
        self.test_train_file = opts.test_train
        self.pre_emb_file = parameters["pre_emb"]
        self.embed_models_file = "models/mapping.pkl"
        self.embed_models_path = "models/"
        self.output_score = "evaluation/temp/score.txt"
        self.output_path = os.path.join(self.base_path, 'output', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

        self.use_gpu = parameters["use_gpu"] #"whether or not to ues gpu"
        self.device = torch.device("cuda" if self.use_gpu else"cpu") #if use_gpu else"cpu"
        self.loss = opts.loss# "loss file location"
        self.name = parameters["name"] # 模型保存名称
        self.model_savename = "models/" + parameters["name"]
        # self.model_savename = "models/cve_ner_models"
        self.tag_scheme = parameters["tag_scheme"] # "Tagging scheme (IOB or IOBES)"
        self.lower = parameters["lower"] # "Lowercase words (this will not affect character inputs)"
        self.zeros = parameters["zeros"] # "Replace digits with 0"


        self.char_mode = "CNN"# 字符特征处理方式，可选CNN或LSTM
        self.tf_idf = parameters["tf_idf"] # 是否加入TF特征,默认不加
        self.do_lower_case = opts.do_lower_case# "是否在分词时做全小写处理"

        # 指定Config, Tokenizer和Model三个核心模型的位置（就是从抱抱脸下载的那几个文件）
        self.bert_model_path = os.path.join(self.base_path, 'bert-base-cased')
        self.word_dim = parameters["word_dim"]#100"Token embedding dimension"
        self.char_dim = parameters["char_dim"]#25
        self.word_lstm_dim = parameters["word_lstm_dim"]#200"Token LSTM hidden layer size"
        self.word_bidirect = parameters["word_bidirect"]# "Use a bidirectional LSTM for words"
        self.char_bidirect = parameters["char_bidirect"]# "Use a bidirectional LSTM for chars"
        self.all_emb = parameters["all_emb"]# "Load all embeddings"
        self.cap_dim = parameters["cap_dim"]# "Capitalization feature dimension (0 to disable)"
        # self.crf = parameters["crf"]# "Use CRF (0 to disable)"
        self.dropout = parameters["dropout"]# "Droupout on the input (0 = no dropout)"
        self.reload = parameters["reload"]#"Reload the last saved model"
        self.parameters_dict = parameters

        self.embedding_dim = 100
        self.hidden_dim = 200
        self.char_lstm_dim = 25
        self.char_to_ix = None
        self.pre_word_embeds = None
        self.char_embedding_dim = 25
        self.n_cap = None
        self.cap_embedding_dim = None
        self.use_crf = parameters["crf"]

        self.need_birnn = True
        self.rnn_dim = 128