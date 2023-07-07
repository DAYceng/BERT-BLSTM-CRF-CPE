import os
import re
import string
import numpy as np
from torch.nn import init
from sklearn.feature_extraction.text import TfidfVectorizer

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def summary_deal(summary_raw,setcase):
    '''
    用于处理原始summary的函数
    :param summary_raw: 从summary列表遍历得到一条summary并传入
    :param setcase:设置输出summary的大小写形式，0为全小写，1为保留原始大小写
    :return: 返回一个分隔好的字符串形式的summary
    '''
    summary_backup = summary_raw  #先将原始summary（即summary_i ）暂存

    sentence = summary_raw.replace('_',' ')
    sentence = summary_raw.replace('.','')
    # print(sentence)
    summary_data = re.findall(r"[\w']+|[\"',!?;*_]", sentence)#将小写后的summary标点分割
    # print(summary_data)

    summary_data = ' '.join(summary_data)
    # print(summary_data)
    # summary_data = summary_data.replace('. NET','.NET')
    # summary_data = summary_data.replace('8 . 1','8.1')
    # summary_data = summary_data.replace('. d','.d')
    # summary_data = summary_data.replace('. sh','.sh')
    # print(summary_data)

    summary_lower = summary_data.lower()  #变为小写
    # print(summary_lower)
    summary_yuan = ' '.join(summary_data)
    # print(summary_yuan)
    if setcase == 0:
        return summary_lower
    elif setcase == 1:
        return summary_data

def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][: v[::-1].index("/")][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(",", "")) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(pretrained, param_value.shape).astype(np.float32))


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    从字典创建映射（item to ID / ID to item）。项目按频率递减排序。
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub("\d", "0", s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    判断是否符合IOB格式
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            return False
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split("-")[0] == "B":
            if i + 1 != len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("B-", "S-"))
        elif tag.split("-")[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("I-", "E-"))
        else:
            raise Exception("Invalid IOB format!")
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split("-")[0] == "B":
            new_tags.append(tag)
        elif tag.split("-")[0] == "I":
            new_tags.append(tag)
        elif tag.split("-")[0] == "S":
            new_tags.append(tag.replace("S-", "B-"))
        elif tag.split("-")[0] == "E":
            new_tags.append(tag.replace("E-", "I-"))
        elif tag.split("-")[0] == "O":
            new_tags.append(tag)
        else:
            raise Exception("Invalid format!")
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data["words"]
    chars = data["chars"]
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters["cap_dim"]:
        caps = data["caps"]
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters["word_dim"]:
        input.append(words)
    if parameters["char_dim"]:
        input.append(char_for)
        if parameters["char_bidirect"]:
            input.append(char_rev)
        input.append(char_pos)
    if parameters["cap_dim"]:
        input.append(caps)
    if add_label:
        input.append(data["tags"])
    return input


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    init.uniform_(input_embedding, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    init.xavier_normal_(input_linear.weight.data)
    init.normal_(input_linear.bias.data)


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for param in input_lstm.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data)

def write_Fdata_list(Fdata_list):
    # for software in cpedatasave_dict:
    #     cpedatasave_dict[software] = sorted(cpedatasave_dict[count])
    with open('Fscore_data_list.py', 'w') as f_write:
        f_write.write('Fdata_list = ' + str(Fdata_list))

def write_lossdata_list(lossdata_list,model):
    # for software in cpedatasave_dict:
    #     cpedatasave_dict[software] = sorted(cpedatasave_dict[count])
    if model == 0:
        with open('loss_data_list.py', 'w') as f_write:
            f_write.write('lossdata_list = ' + str(lossdata_list))
    elif model == 1:
        with open('testset_loss_data_list.py', 'w') as f_write:
            f_write.write('lossdata_list = ' + str(lossdata_list))

#计算tfidf特征矩阵
def tfidf_calculate(train_doc,test_doc):
    tfidf_list = []
    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, stop_words='english',norm='l2') # 实例化tf实例
    tv_fit = tv.fit_transform(train_doc) # 训练，构建词汇表以及词项idf值，并将输入文本列表转成VSM矩阵形式
    sparse_result = tv.transform(train_doc) # 输出TFIDF矩阵中有计算值的项并标明位置和具体的数值
    vocab_name = tv.get_feature_names() # 查看一下构建的词汇表
    vocab_VSMarray = tv_fit.toarray() # 查看输入文本列表的VSM矩阵

    sg = SegWord(load_inner=False)
    test_sg = sg.tokenize_no_space(test_doc[0].lower())

    test_fit = tv.transform(test_doc)
    testvocab_name = tv.get_feature_names()
    testvocab_VSMarray = test_fit.toarray()
    vocabdict = dict(zip(testvocab_name, testvocab_VSMarray[0]))
    # 去除字典中值为0的元素
    for v in list(vocabdict.keys()):   #对字典a中的keys，相当于形成列表list
        if vocabdict[v] == 0:
            del vocabdict[v]

    for word in test_sg:
        for k in list(vocabdict.keys()):
            if k == word:
                tfidf_list.append(vocabdict[k])
                flag = 1
                break
            else:
                flag = 0
    if flag == 0:
        tfidf_list.append(0) #若为停用词，直接将其tfidf值设为0

    tmpsort_list = list(tfidf_list) #将tfidf列表的元素暂存，防止排序操作影响原列表顺序
    tfidf_feature = list(tfidf_list)
    tmpsort_list.sort(reverse=True) #对tfidf值进行降序排序
    # print(tmpsort_list)

    #将tfidf值由大到小排名的前五个值分别标记为1~5,并还原到原来的值的位置
    for i,v1 in enumerate(tmpsort_list):
        for j,v2 in enumerate(tfidf_feature):
            if v1 == v2 and i <= 4:
                tfidf_feature[j] = i+1
                break

    # 将其余的值标记为0
    for i,v in enumerate(tfidf_feature):
        if isinstance(v,float):
            tfidf_feature[i] = 0
        else:pass

    return tfidf_list, tfidf_feature, test_sg

# 用于分词
class SegWord(object):

    def __init__(self, load_inner=True):
        self._dict_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./innerDict.ini")
        self._word_dict = {}
        if load_inner:
            self.load_dict_file()

    def add_word(self, word):
        """
        添加自定义的分词词语到分词词典中
        :param word: 对应的词语
        :return:
        """
        if not word:
            raise Exception("word is null")
        first_char = word[0]
        self._word_dict.setdefault(first_char, [])
        if word not in self._word_dict[first_char]:
            self._word_dict[first_char].append(word)
            self._sort_word_dict()

    def _sort_word_dict(self):
        """
        对对应的字符所包含的字典进行排序
        :return:
        """
        for first_char, words in self._word_dict.items():
            self._word_dict[first_char] = sorted(words, key=lambda x: len(x), reverse=True)

    def load_dict_file(self, dict_file_path: str = ''):
        """
        加载字典
        :param dict_file_path: 字典文件地址
        :return:
        """
        if not dict_file_path:
            load_dict_path = self._dict_file_path
        else:
            if not os.path.exists(dict_file_path):
                raise Exception("can't find this file %s" % dict_file_path)
            else:
                load_dict_path = dict_file_path
        with open(load_dict_path, 'r', encoding='utf8') as reader:
            words = [word for word in reader.read().replace("\n\n", "\n").split('\n')]
        self.batch_add_words(words)

    def batch_add_words(self, words: list):
        """
        批量增加数据
        :param words: word的list
        :return:
        """
        for word in words:
            first_char = word[0]
            self._word_dict.setdefault(first_char, [])
            if word not in self._word_dict[first_char]:
                self._word_dict[first_char].append(word)
        self._sort_word_dict()

    def _match_word(self, first_char, i, sentence):
        """
        匹配
        :param first_char: 最新需要处理的开头字符
        :param i: 开头字符对应的索引
        :param sentence: 原始语句
        :return:
        """
        if first_char not in self._word_dict:
            if first_char in string.ascii_letters:
                return self._match_ascii(i, sentence)
            return first_char
        words = self._word_dict[first_char]
        for word in words:
            if sentence[i:i + len(word)] == word:
                return word
        if first_char in string.ascii_letters:
            return self._match_ascii(i, sentence)
        return first_char

    @staticmethod
    def _match_ascii(i, sentence):
        _result = ''
        for i in range(i, len(sentence)):
            if sentence[i] not in string.ascii_letters:
                break
            _result += sentence[i]
        return _result

    def tokenize(self, sentence):
        """
        分词
        :param sentence: 待分词语句
        :return:
        """
        tokens = []
        if not sentence:
            return tokens
        i = 0
        while i < len(sentence):
            first_char = sentence[i]
            matched_word = self._match_word(first_char, i, sentence)
            tokens.append(matched_word)
            i += len(matched_word)
        return tokens

    def tokenize_no_space(self, sentence):
        """
        返回无空格的分词
        :param sentence:
        :return:
        """
        _seg_word_result = self.tokenize(sentence)
        return [word for word in _seg_word_result if word not in string.whitespace]
