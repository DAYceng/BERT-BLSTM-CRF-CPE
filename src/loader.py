import os
import re
from utils import create_mapping, zero_digits
from utils import iob2, iob_iobes
import model
from collections import Counter
from utils import SegWord
from sklearn.feature_extraction.text import TfidfVectorizer

tf_flag = 0

def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    从/src中读取eng.train、eng.testa、eng.tastb
    存入sentences字典
    形式就是一个嵌套列表，如：
    [...,['surprising', 'JJ', 'I-ADJP', 'O'],...]
    本质上就是一篇完整的语料，把其中的词单独存为一个列表(带有标签的)然后再构成一个列表保存
    """
    print('Loading sentences from %s...' % path)
    print(os.path.dirname(os.path.abspath(__file__)))#打印路径
    sentences = []
    sentence = []
    # summaries = []
    # word_list = []

    for line in open(path, 'r', encoding='utf-8'):
        #print(line)
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        #用0代替所有数字

        if not line and len(sentence) > 0:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence) #遇到截止标记，将之前获取的一条句子加到sentences
                # if model==1:
                #     summary = ' '.join(word_list) #还原整句的summary
                #     summaries.append(summary)
            sentence = []
            # word_list = []
            # print(sentences)
        else:
            word = line.split()
            # print(word)
            assert len(word) >= 2
            sentence.append(word)
            # if model == 1:
            #     word_list.append(word[0])
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence) #遇到截止标记，将之前获取的一条句子加到sentences
            # if model == 1:
            #     summary = ' '.join(word_list)
            #     summaries.append(summary)
    # if model == 0:
    #     return sentences
    # elif model == 1:
    #     return summaries
    return sentences
    # print(sentences)
    # return sentences,summaries




def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]#遍历获取sentences中一个列表元素的标签
        #print(tags)
        # Check that tags are given in the IOB format判断是否符合IOB格式
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    创建一个字典和单词映射，按照词频排序
    """
    words = [x[0].lower() if lower else x[0] for s in sentences for x in s]
    dico = dict(Counter(words))#统计词频
    # print(dico)
    
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=3}
    # print(dico)
    word_to_id, id_to_word = create_mapping(dico)
    # print(word_to_id) #一个字典{...,'victories': 7442, 'video': 7443, 'videos': 7444,...}
    # print(id_to_word) #也是一个字典{...，7313: 'superman', 7314: 'surface', 7315: 'surging',...}
    print("Found %i unique words (%i in total)" % (
        len(dico), len(words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    sentence形式：一个列表储存一个完整的句子，句子中每个单词又单独与其标签构成一个列表
    [['The', 'DT', 'I-NP', 'O'], ['case', 'NN', 'I-NP', 'O'],...]
    """
    chars = ''.join([w[0] for s in sentences for w in s])
    dico = dict(Counter(chars))
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000

    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [word[-1] for s in sentences for word in s]
    dico = dict(Counter(tags)) #Counter是一种类似字典结构的对象
    # print(dico)
    dico[model.START_TAG] = -1
    dico[model.STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    标注大小写特征
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

#计算tfidf特征矩阵
def tfidf_calculate(tv, sg, test_doc):
    tfidf_list = []
    # tv = TfidfVectorizer(use_idf=True, smooth_idf=True, stop_words='english',norm='l2') # 实例化tf实例
    # tv_fit = tv.fit_transform(train_doc) # 训练，构建词汇表以及词项idf值，并将输入文本列表转成VSM矩阵形式
    # # sparse_result = tv.transform(train_doc) # 输出TFIDF矩阵中有计算值的项并标明位置和具体的数值
    # # vocab_name = tv.get_feature_names() # 查看一下构建的词汇表
    # # vocab_VSMarray = tv_fit.toarray() # 查看输入文本列表的VSM矩阵
    #
    # sg = SegWord(load_inner=False)
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

    return tfidf_feature


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }

def prepare_summaries(sentences):
    tmp_summarylist = []
    summaries = []
    for sentence in sentences:
        for wl in sentence:
            tmp_summarylist.append(wl[0])
        single_summary = ' '.join(tmp_summarylist)
        tmp_summarylist = []
        summaries.append(single_summary)

    train_doc = summaries
    return train_doc

def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=True, tf_idf=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    准备数据集
    说明：
    sentence形式：一个列表储存一个完整的句子，句子中每个单词又单独与其标签构成一个列表
    [['The', 'DT', 'I-NP', 'O'], ['case', 'NN', 'I-NP', 'O'], ['is', 'VBZ', 'I-VP', 'O'], ['closed', 'JJ', 'I-ADJP', 'O'], [',', ',', 'O', 'O'], ['"', '"', 'O', 'O'], ['a', 'DT', 'I-NP', 'O'], ['spokesman', 'NN', 'I-NP', 'O'], ['said', 'VBD', 'I-VP', 'O'], ['.', '.', 'O', 'O']]

    str_words：['The','case','is','closed','a','spokesman','said']
    """
    def f(x): return x.lower() if lower else x#定义函数fx，判断参数x是否为小写字符
    data = []
    # for summary in summaries:
    #     tfidf_list, tfidf_feature, test_sg = tfidf_calculate(summaries,summary)

    # tmp_summarylist = []
    # summaries = []
    #
    # for sentence in sentences:
    #     for wl in sentence:
    #         tmp_summarylist.append(wl[0])
    #     single_summary = ' '.join(tmp_summarylist)
    #     tmp_summarylist = []
    #     summaries.append(single_summary)
    #
    # train_doc = summaries
    if tf_idf:
        train_doc = prepare_summaries(sentences)
        tv = TfidfVectorizer(use_idf=True, smooth_idf=True, stop_words='english',norm='l2') # 实例化tf实例
        tv_fit = tv.fit_transform(train_doc) # 训练，构建词汇表以及词项idf值，并将输入文本列表转成VSM矩阵形式
        sg = SegWord(load_inner=False)
        global tf_flag
        tf_flag = 1


    for sentence in sentences:
        # print(sentence)
        str_words = [w[0] for w in sentence]#将句子提取出来
        # print(str_words)
        test_doc = ' '.join(str_words)

        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>'] for w in str_words]
        # print(words)
        # Skip characters that are not in the training set
        chars = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words] #遍历str_words获取单个词语，然后使用cap_feature识别大小写特征
        tags = [tag_to_id[w[-1]] for w in sentence]

        # tfidf_list = []
        if tf_flag == 1:
            tfidf_feature = tfidf_calculate(tv, sg, [test_doc])
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'caps': caps,
                'tags': tags,
                'tfidf': tfidf_feature,
            })
        else:
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'caps': caps,
                'tags': tags,
            })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in open(ext_emb_path, 'r', encoding='utf-8')
    ])
    # print(pretrained)

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    '''
    我们要么添加预训练文件中的每个单词，
    要么只添加“单词”列表中给出的单词，
    我们可以为其分配预训练嵌入
    '''
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    # print(dictionary)
    return dictionary, word_to_id, id_to_word


def pad_seq(seq, max_length, PAD_token=0):
    # add pads
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq















