# coding=utf-8
import itertools
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import sys
import visdom
import torch
import pickle
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import src.evaluation.conlleval as evaluate
# import evaluation.conlleval as evaluate

from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from tensorboardX import SummaryWriter
# 查看保存的训练数据：tensorboard --logdir "./eval"
import loader
from utils import *
from loader import *
from model import BiLSTM_CRF
from src.diy_config import BLSTM_Config
# from diy_config import BLSTM_Config


class CVE_NER(object):

    def __init__(self):
        # 初始化系统配置、数据预处理
        self.BLSTM_config = BLSTM_Config()
    #评估函数
    def evaluating(self, model, datas, best_F, draw, writer, epoch, drawingname=''):
        # self.draw = draw
        # FB1 on pharse level
        prediction = []
        eval_list = []
        save = False
        new_F = 0.0
        confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
        for data in datas:
            #datas为一个字典，下面载入其中各键对应的值作为训练数据
            ground_truth_id = data['tags']
            words = data['str_words']
            chars2 = data['chars']
            caps = data['caps']

            if self.BLSTM_config.char_mode == 'LSTM':
                chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
                d = {}
                for i, ci in enumerate(chars2):
                    for j, cj in enumerate(chars2_sorted):
                        if ci == cj and not j in d and not i in d.values():
                            d[j] = i
                            continue
                chars2_length = [len(c) for c in chars2_sorted]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
                for i, c in enumerate(chars2_sorted):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            if self.BLSTM_config.char_mode == 'CNN':
                d = {}
                chars2_length = [len(c) for c in chars2]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                for i, c in enumerate(chars2):
                    chars2_mask[i, :chars2_length[i]] = c

                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            dwords = Variable(torch.LongTensor(data['words']))
            dcaps = Variable(torch.LongTensor(caps))
            if self.BLSTM_config.use_gpu:
                val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
            else:
                val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
            predicted_id = out

            for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
                line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
                eval_list_line = ''.join([line, "\n"])
                prediction.append(line)
                eval_list.append(eval_list_line)
                confusion_matrix[true_id, pred_id] += 1
            prediction.append('')
            eval_list.append("\n")

        predf = eval_temp + '/pred.' + self.BLSTM_config.name
        scoref = eval_temp + '/score.' + self.BLSTM_config.name
        #训练后进行临时的测试并计算当前F值
        with open(predf, 'w', encoding='utf8') as f:
            f.write('\n'.join(prediction))
        os.system('%s < %s > %s' % (eval_script, predf, scoref)) #开启一个子shell执行评估脚本conlleval
        # with open(scoref, 'w', encoding='utf8') as f2:
        #     line = f2.readline()
        #     print(line)

        # # 使用pl版conlleval来评估结果
        # eval_lines = [l.rstrip() for l in open(scoref, 'r', encoding='utf8')]
        # # print(eval_lines)
        #
        # for i, line in enumerate(eval_lines):
        #     print('eval_lines:',eval_lines)
        #     print('line:',line)
        #     if i == 1:
        #         new_F = float(line.strip().split()[-1])
        #         if draw == 1:
        #             writer.add_scalar(drawingname, new_F, epoch)
        #         if new_F > best_F:
        #             best_F = new_F
        #             save = True
        #             print('the best F is ', new_F)

        # 使用py版conlleval来评估结果
        counts = evaluate.evaluate(eval_list)
        # counts = evaluate(eval_list)
        evaluate.report(counts)
        # report(counts)
        # 返回一个tup：('Metrics', 'tp fp fn prec rec fscore')
        overall, by_type = evaluate.metrics(counts)
        # overall, by_type = metrics(counts)
        # 获取F值、准确率、召回率
        f1_score = overall.fscore
        # prec_score = overall.prec
        # rec_score = overall.rec
        new_F = f1_score
        if draw == 1:
            writer.add_scalar(drawingname, overall.fscore, epoch)
        if new_F > best_F and new_F > 90:
            best_F = new_F
            save = True
            print('the best F is ', new_F)


        # # 用于输出标签评估信息
        # print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        #     "ID", "NE", "Total",
        #     *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
        # ))# 打印输出的标题（标签）
        # # {":补位字符<补位方向>补位数量"}
        # # exam
        # # print("{:0<5}".format(8))
        # # >>80000
        # for i in range(confusion_matrix.size(0)):
        #     print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        #         str(i), id_to_tag[i], str(confusion_matrix[i].sum().item()),
        #         *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
        #           ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        #     ))
        # # print()
        # # confusion_matrix.size(0) = 5
        # # id_to_tag：{0: 'O', 1: 'B-PRODUCT', 2: 'I-PRODUCT', 3: 'B-VENDOR', 4: 'I-VENDOR', 5: '<START>', 6: '<STOP>'}
        # # id_to_tag[i] = 'O'
        # # str(confusion_matrix[i].sum().item()) = 43323
        # # confusion_matrix.size(0) = 5

        return best_F, new_F, save

    def train(self):
      # 训练参数设置
        learning_rate = 0.015
        # optimizer是优化器，SGD（随机梯度下降）是最基础的优化方法，把整套数据不断放入网络中训练
        # momentum（动量加速），一种更新W参数的方法
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        losses = [] #
        loss = 0.0
        best_dev_F = -1.0
        best_test_F = -1.0
        best_train_F = -1.0
        all_F = [[0, 0, 0]]
        plot_every = 100
        eval_every = 200
        count = 0
        vis = visdom.Visdom()
        sys.stdout.flush()
        savelosslist = []
        save_F = []

        # SummaryWriter构造函数
        writer = SummaryWriter(logdir=os.path.join(CVE_NER.BLSTM_config.output_path, "eval"), comment="ner")

        model.train(True) #开启训练
        # 训练10000个epoch，使用tqdm对循环过程进行可视化（进度条）
        # np.random.permutation（）对序列进行随机排序
        for epoch in range(1, 100):
            for iter, index in enumerate(tqdm(np.random.permutation(len(train_data)))):
                data = train_data[index]
                model.zero_grad() # 把模型中参数的梯度设为0
                count += 1
                sentence_in = data["words"]
                sentence_in = Variable(torch.LongTensor(sentence_in)) # 将tensor形式的sentence_in，即句子映射，Variable化，方便之后进行运算
                tags = data["tags"]
                chars = data["chars"]

                # char lstm
                if self.BLSTM_config.char_mode == "LSTM":
                    chars_sorted = sorted(chars, key=lambda p: len(p), reverse=True)
                    d = {}
                    for i, ci in enumerate(chars):
                        for j, cj in enumerate(chars_sorted):
                            if ci == cj and not j in d and not i in d.values():
                                d[j] = i
                                continue
                    chars_length = [len(c) for c in chars_sorted]
                    char_maxl = max(chars_length)
                    chars_mask = np.zeros((len(chars_sorted), char_maxl), dtype="int")
                    for i, c in enumerate(chars_sorted):
                        chars_mask[i, :chars_length[i]] = c
                    chars_mask = Variable(torch.LongTensor(chars_mask))

                # char cnn
                if self.BLSTM_config.char_mode == "CNN":
                    d = {}
                    chars_length = [len(c) for c in chars] # chars有长有短，统计各个char的长度
                    char_maxl = max(chars_length) # 找出最长的char做为padding的依据
                    chars_mask = np.zeros((len(chars_length), char_maxl), dtype="int") # 以最长的char为长度创建一个用0填充的数组
                    # 将chars对应数据添加到数组中，相当于对char做了padding
                    for i, c in enumerate(chars):
                        chars_mask[i, :chars_length[i]] = c
                    chars_mask = Variable(torch.LongTensor(chars_mask))

                targets = torch.LongTensor(tags)
                caps = Variable(torch.LongTensor(data["caps"]))
                if self.BLSTM_config.use_gpu:
                    neg_log_likelihood = model.neg_log_likelihood(
                        #求loss
                        sentence_in.cuda(),
                        targets.cuda(),
                        chars_mask.cuda(),
                        caps.cuda(),
                        chars_length,
                        d,
                    )
                else:
                    neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars_mask, caps, chars_length, d)
                loss += neg_log_likelihood.data.item() / len(data["words"])
                neg_log_likelihood.backward() #反向传播相关
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) #梯度剪裁，防止梯度爆炸
                optimizer.step() #用于更新参数

               # 在count为plot_every的整数倍时触发绘图模块
                if count % plot_every == 0:
                    loss /= plot_every
                    # 保存训练loss
                    writer.add_scalar("Train/loss", loss, count)
                    # print(loss)
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)

                    # tmptup = (i,losses)
                    # savelosslist.append(tmptup)
                    # # loss_dict[i] = losses
                    # write_lossdata_list(savelosslist,0)

                    text = "<p>" + "</p><p>".join([str(l) for l in losses[-9:]]) + "</p>"
                    losswin = "loss_" + self.BLSTM_config.name
                    textwin = "loss_text_" + self.BLSTM_config.name
                    vis.line(
                        np.array(losses),
                        X=np.array([plot_every * i for i in range(len(losses))]),
                        win=losswin,
                        opts={
                            "title": losswin,
                            "legend": ["loss"]
                        },
                    )
                    vis.text(text, win=textwin, opts={"title": textwin}) #visdom可视化相关
                    loss = 0.0
                #负责模型保存，count为训练次数计数值
                #保存条件：1、count为200的整数倍 and 2、count>4000 或 count为400的整数倍且count<400
                if (count % (eval_every) == 0 and count > (eval_every * 20) or count % (eval_every * 4) == 0 and count <
                    (eval_every * 20)):

                    model.train(False)
                    '''
                    调用evaluating方法计算当前的F值
                    其输入为模型文件、上次的最佳F值（用作比较）和模型待使用（评估）的数据集
                    计算完成，返回新的F值和当前的最佳F值以及是否保存模型的save标志
                    若新F值为当前最佳，则save == ture，保存模型
                    保存完毕会调用当前模型再次计算F值作为当前最佳F值供下次计算使用
                    '''
                    # train_drawingname = "Eval/new_dev_F"
                    best_train_F, new_train_F, _ = self.evaluating(model, test_train_data, best_train_F, 0, writer, epoch)

                    # dev_drawingname = "Eval/new_dev_F"
                    best_dev_F, new_dev_F, save = self.evaluating(model, dev_data, best_dev_F, 0, writer, epoch)

                    # print('trainF:', best_train_F)
                    print('NEWtrainF:', new_train_F)

                    if save:
                        # torch.save(model, model_name) #保存模型
                        torch.save(model.state_dict(), CVE_NER.BLSTM_config.model_savename, _use_new_zipfile_serialization=True) #保存模型
                    # test_drawingname = "Eval/new_test_F"
                    best_test_F, new_test_F, _ = self.evaluating(model, test_data, best_test_F, 0, writer, epoch)

                    sys.stdout.flush()

                    all_F.append([new_train_F, new_dev_F, new_test_F])

                    # # 为使用visdom本地绘图保存数据
                    # if new_train_F != 0 :
                    #     save_F.append(all_F)
                    # write_Fdata_list(save_F)

                    Fwin = "F-score of {train, dev, test}_" + self.BLSTM_config.name
                    vis.line(
                        np.array(all_F),
                        win=Fwin,
                        X=np.array([eval_every * i for i in range(len(all_F))]),
                        opts={
                            "title": Fwin,
                            "legend": ["train", "dev", "test"]
                        },
                    )

                    model.train(True)

                if count % len(train_data) == 0:
                    adjust_learning_rate(optimizer, lr=learning_rate / (1 + 0.05 * count / len(train_data)))

            test_drawingname = "Eval/new_test_F"
            best_test_F, new_test_F, _ = self.evaluating(model, test_data, best_test_F, 1, writer, epoch, test_drawingname)
            # 将F值保存值一个list中用于本地matplotlib绘图
            save_F.append(new_test_F)
            write_Fdata_list(save_F)
        plt.plot(losses)
        plt.show()
        print('训练次数：',count)


if __name__ == "__main__":

    CVE_NER = CVE_NER()
    assert os.path.isfile(CVE_NER.BLSTM_config.train_file)
    assert os.path.isfile(CVE_NER.BLSTM_config.dev_file)
    assert os.path.isfile(CVE_NER.BLSTM_config.test_file)

    assert CVE_NER.BLSTM_config.char_dim > 0 or CVE_NER.BLSTM_config.word_dim > 0
    assert 0.0 <= CVE_NER.BLSTM_config.dropout < 1.0
    assert CVE_NER.BLSTM_config.tag_scheme in ["iob", "iobes"]
    assert not CVE_NER.BLSTM_config.all_emb or CVE_NER.BLSTM_config.pre_emb_file
    assert not CVE_NER.BLSTM_config.pre_emb_file or CVE_NER.BLSTM_config.word_dim > 0
    assert not CVE_NER.BLSTM_config.pre_emb_file or os.path.isfile(CVE_NER.BLSTM_config.pre_emb_file)

    #确认conlleval脚本的存在，该脚本用于对CRF测试结果进行评价（计算F1值）
    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)
    if not os.path.exists(CVE_NER.BLSTM_config.embed_models_path):
        os.makedirs(CVE_NER.BLSTM_config.embed_models_path)

    device = CVE_NER.BLSTM_config.switch_device
    lower = CVE_NER.BLSTM_config.lower
    zeros = CVE_NER.BLSTM_config.zeros
    tag_scheme = CVE_NER.BLSTM_config.tag_scheme
    tf_idf = CVE_NER.BLSTM_config.tf_idf
    #载入从/src中读取eng.train、eng.testa、eng.tastb存入sentences字典
    #载入sentences作为train_sentences，即训练sentences集

    #载入训练集
    train_sentences = loader.load_sentences(CVE_NER.BLSTM_config.train_file, lower, zeros)

    # #载入tfidf的语料库
    # train_doc = loader.load_sentences(opts.train, lower, zeros)

    #载入交叉验证集(开发集)
    dev_sentences = loader.load_sentences(CVE_NER.BLSTM_config.dev_file, lower, zeros)
    #载入测试集
    test_sentences = loader.load_sentences(CVE_NER.BLSTM_config.test_file, lower, zeros)

    #测试sentences集
    test_train_sentences = loader.load_sentences(CVE_NER.BLSTM_config.test_train_file, lower, zeros)

    #检查上述sentences集中数据标注方案是否符合IOB规范（可去除）
    update_tag_scheme(train_sentences, tag_scheme)
    update_tag_scheme(dev_sentences, tag_scheme)
    update_tag_scheme(test_sentences, tag_scheme)
    update_tag_scheme(test_train_sentences, tag_scheme)

    dico_words_train = word_mapping(train_sentences, lower)[0]
    # print(dico_words_train)

    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        CVE_NER.BLSTM_config.pre_emb_file,
        list(itertools.chain.from_iterable([[w[0] for w in s] for s in dev_sentences +
                                            test_sentences])) if not CVE_NER.BLSTM_config.all_emb else None,
    )
    # print(dico_words) #一个字典{...，'super-middleweight': 0, 'putland': 0, 'homeopathic': 0,...}
    # print(word_to_id) #一个字典{...，'supertram': 349163, 'supertramp': 349164, 'supertruck': 349165,...}
    # print(id_to_word) #一个字典{...,'unpenalized', 373156: 'unperformed', 373157: 'unpermitted',...}
    #
    '''
上述字典均是载入glove6B后生成的，与载入前生成的不同
包括打印结果中也发现word_to_id和id_to_word中含有大量在仅载入eng.train时没有的词
说明此时程序已经执行映射，将训练语料中没有的词由glove6B添加进来
   '''

    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    # 标签个数
    num_labels = len(dico_tags)
    # print(len(tag_to_id))
    # print(dico_tags)
    #原版数据{'S-ORG': 3836, 'O': 169578, 'S-MISC': 2580, 'B-PER': 4284, 'E-PER': 4284, 'S-LOC': 6099, 'B-ORG': 2485, 'E-ORG': 2485, 'I-PER': 244, 'S-PER': 2316, 'B-MISC': 858, 'I-MISC': 297, 'E-MISC': 858, 'I-ORG': 1219, 'B-LOC': 1041, 'E-LOC': 1041, 'I-LOC': 116, '<START>': -1, '<STOP>': -2}
    #{'O': 692995, 'S-GPE': 2, '<START>': -1, '<STOP>': -2}


    train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, lower, tf_idf)
    # print(train_data)
    #{'str_words': ['Leading', 'scores', 'after'], 'words': [258, 565, 42], 'chars': [[47, 2, 3, 11, 6, 5, 18], [9, 13, 7, 8, 2, 9], [3, 16, 4, 2, 8]], 'caps': [2, 0, 0], 'tags': [0, 0, 0]}

    dev_data = prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, lower, tf_idf)
    test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower, tf_idf)
    test_train_data = prepare_dataset(test_train_sentences, word_to_id, char_to_id, tag_to_id, lower, tf_idf)
    print("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    all_word_embeds = {}
    # opts.pre_emb
    for i, line in enumerate(open(CVE_NER.BLSTM_config.pre_emb_file, "r", encoding="utf-8")):
        #打开glove6B
        s = line.strip().split()
        # print(s) #['show', '0.10735', '-0.13863', '0.057066',...]
        if len(s) == CVE_NER.BLSTM_config.word_dim + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), CVE_NER.BLSTM_config.word_dim))

    for w in word_to_id:
        if w in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

    print("Loaded %i pretrained embeddings." % len(all_word_embeds))

    with open(CVE_NER.BLSTM_config.embed_models_file, "wb") as f:
        mappings = {
            "word_to_id": word_to_id,
            "tag_to_id": tag_to_id,
            "char_to_id": char_to_id,
            "parameters": CVE_NER.BLSTM_config.parameters_dict, #  OrderedDict[str, bool] = OrderedDict(
            "word_embeds": word_embeds,
        }
        pickle.dump(mappings, f) #加载glove6B映射数据

    # print(word_to_id) #一个字典{...,'victories': 7442, 'video': 7443,
    print("word_to_id: ", len(word_to_id))

    model = BiLSTM_CRF(
        vocab_size=len(word_to_id),
        tag_to_ix=tag_to_id,
        char_to_ix=char_to_id,
        pre_word_embeds=word_embeds,
        # BertConfig的参数与原模型参数有冲突
        embedding_dim=CVE_NER.BLSTM_config.word_dim,
        hidden_dim=CVE_NER.BLSTM_config.word_lstm_dim,
        use_gpu=CVE_NER.BLSTM_config.use_gpu,
        use_crf=CVE_NER.BLSTM_config.use_crf,
        char_mode=CVE_NER.BLSTM_config.char_mode,
    )

    if CVE_NER.BLSTM_config.reload:
        # model_name = CVE_NER.BLSTM_config.embed_models_path + CVE_NER.BLSTM_config.name  # get_name(parameters)
        model = torch.load(CVE_NER.BLSTM_config.model_savename)

    model.to(device)
    CVE_NER.train()

