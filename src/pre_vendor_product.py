# coding=utf-8
import optparse
import os

from src.loader import load_sentences
from src.utils import eval_script, eval_temp

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import time
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from torch.autograd import Variable

from loader import *
from utils import *

from model import BiLSTM_CRF

# python -m visdom.server 训练之前先启动可视化模块

optparser = optparse.OptionParser()
optparser.add_option(
    "-t", "--test", default="data/pre.txt",#data/eng.testb
    help="Test set location"
)
# optparser.add_option(
#     "-pre", "--pred_test", default="evaluation/temp/pred.test",
#     help="Forecast result"
# )
optparser.add_option(
    '--score', default='evaluation/temp/score.txt',
    help='score file location'
)
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    '--loss', default='loss.txt',
    help='loss file location'
)
optparser.add_option(
    '--model_path', default='models/test',
    help='model path'
)
optparser.add_option(
    '--map_path', default='models/mapping.pkl',
    help='model path'
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)

opts = optparser.parse_args()[0]

mapping_file = opts.map_path

with open(mapping_file, 'rb') as f:
    mappings = pickle.load(f) #加载模型
#mapping的作用是什么？

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

use_gpu =  opts.use_gpu == 1 and torch.cuda.is_available()


# assert os.path.isfile(opts.test)
assert parameters['tag_scheme'] in ['iob', 'iobes']

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# test_sentences = load_sentences(opts.test, lower, zeros) #per.txt
# # update_tag_scheme(test_sentences, tag_scheme)
# test_data = prepare_dataset(
#     test_sentences, word_to_id, char_to_id, tag_to_id, lower
# )
#
# model = torch.load(opts.model_path)
# model_name = opts.model_path.split('/')[-1].split('.')[0]
#
# if use_gpu:
#     model.cuda()
# model.eval()

# model = torch.load(opts.model_path)
m_state_dict = torch.load('mymodule.pt')
model = BiLSTM_CRF()
model.load_state_dict(m_state_dict)

model_name = opts.model_path.split('/')[-1].split('.')[0]

if use_gpu:
    model.cuda()
model.eval()

# #测试用
# summary = 'Microsoft PowerPoint 2000 in Office 2000 SP3 has an interaction with Internet Explorer that allows remote attackers to obtain sensitive information via a PowerPoint presentation that attempts to access objects in the Temporary Internet Files Folder (TIFF).'
summary = 'The kernel in Microsoft Windows XP SP2, Windows Server 2003 SP2, Windows Vista SP2, Windows Server 2008 SP2, R2, and R2 SP1, and Windows 7 Gold and SP1 does not properly load structured exception handling tables, which allows context-dependent attackers to bypass the SafeSEH security feature by leveraging a Visual C++ .NET 2003 application, aka "Windows Kernel SafeSEH Bypass Vulnerability."'
# summary = 'The kernel'
def inputdata(summary):
    '''
    预处理函数，搭配predict_vendor_product()使用
    输入：字符串形式的summary
    输出：用于预测的数据datas
    '''
    summary_token = summary_deal(summary,1)
    prelist = summary_token.split(" ")
    prelist.append('.')
    padinglist = ['O']*len(prelist)
    tmp_array = np.vstack((prelist,padinglist))
    lable_array = tmp_array.T

    np.savetxt(opts.test,lable_array,fmt="%s") #将转制后的矩阵存为txt
    # print(lable_array)

    test_sentences = load_sentences(opts.test, lower, zeros) #per.txt
    update_tag_scheme(test_sentences, tag_scheme)
    test_data = prepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id, lower
    )

    return test_data
# inputdata(summary)


def predict_vendor_product(model, test_data):
    '''
    用于预测VENDOR和PRODUCT的函数
    :param model: 输入需要使用的模型
    :param datas: 用于预测的输入数据（预处理后的summary）
    :return:
    v_list--vendor列表
    p_list--product列表
    '''
    #用于存放预测结果
    prediction = []
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in test_data:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        if parameters['char_mode'] == 'LSTM':
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

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            '''
            Mask 是相对于 PAD 而产生的技术，具备告诉模型一个向量有多长的功效。
            Mask 矩阵有如下特点：Mask 矩阵是与 PAD 之后的矩阵具有相同的 shape。
            mask 矩阵只有 1 和 0两个值，
            如果值为 1 表示 PAD 矩阵中该位置的值有意义，
            值为 0 则表示对应 PAD 矩阵中该位置的值无意义。
            '''
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c #chars2_mask是由特征矩阵构成的Tensor
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        #输入文本中的大小写特征
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(),chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        #预测结果的输出（以id的形式输出）
        predicted_id = out
        #将文本单词与预测结果对应
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    #规定预测结果保存位置
    predf = eval_temp + '/pred.' + model_name
    scoref = eval_temp + '/score.' + model_name

    #保存预测结果以及预测得分
    with open('evaluation/temp/pred.test', 'w',encoding='utf8') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    with open(scoref, 'r',encoding='utf8') as f:
        for l in f.readlines():
            print(l.strip())

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum().item()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))

#读取预测结果，输出vendor和product
def get_vendor_product():
    pre_res = np.loadtxt('evaluation/temp/pred.test', dtype=bytes).astype(str) #读取预测结果

    v_list = []
    p_list = []

    #用于暂存多的vendor/prpduct以便之后合并
    tmp_vlist = []
    tmp_plist = []

    for i in pre_res:

        # print(i[2])

        s = i[2]
        print(s)
        #判定标签类型，以便识别多词vendor/prpduct
        jv = s.find('B-VENDOR')
        jv2 = s.find('I-VENDOR')
        jv3 = s.find('E-VENDOR')
        jvs = s.find('S-VENDOR')

        jp = s.find('B-PRODUCT')
        jp2 = s.find('I-PRODUCT')
        jp3 = s.find('E-PRODUCT')
        jps = s.find('S-PRODUCT')#S-PRODUCT为独立PRODUCT
        if jv != -1 :
            # print('VENDOR:',i[0])
            v_list.append(i[0])
            # return vp_dict
            tmp_vlist.append(i[0])
            tmp1 = i[0]
            v_list.append(tmp1)

        elif jv2 != -1 :
            # tmp2 = i[0]
            tmp_vlist.append(i[0])
            del v_list[0]
            tmp2 = str(tmp_vlist[0]+i[0])
            v_list.append(tmp2)

        elif jv3 != -1 :
            tmp3 = i[0]
            tmp_vlist.append(i[0])
            del v_list[0]
            tmp3 = str(tmp_vlist[0]+' '+tmp_vlist[1])
            v_list.append(tmp3)

        if jvs != -1 :
            # print('VENDOR:',i[0])
            v_list.append(i[0])


        elif jp != -1:
            tmp_plist.append(i[0])
            tmp1 = i[0]
            p_list.append(tmp1)
        elif jp2 != -1 :
            # tmp2 = i[0]
            tmp_plist.append(i[0])
            del p_list[0]
            tmp2 = str(tmp_plist[0]+i[0])
            p_list.append(tmp2)

            # p_list.append(tmp2)
        elif jp3 != -1 :
            tmp3 = i[0]
            tmp_plist.append(i[0])
            del p_list[0]
            tmp3 = str(tmp_plist[0]+' '+tmp_plist[1])
            p_list.append(tmp3)
        if jps != -1 :
            # print('VENDOR:',i[0])
            p_list.append(i[0])

    return v_list,p_list



if __name__ == '__main__':
    test_data=inputdata(summary)
    predict_vendor_product(model,test_data)
    vendor,product = get_vendor_product()

    # print(vendor)
    # print(product)

