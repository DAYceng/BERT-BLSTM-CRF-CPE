#
# import optparse
# import pre_vendor_product
# import numpy as np
# import torch
#
# from src.utils import summary_deal
#
# optparser = optparse.OptionParser()
# optparser.add_option(
#     "-t", "--test", default="data/pre.txt",
#     help="Test set location"
# )
# optparser.add_option(
#     "-t", "--pred_test", default="evaluation/temp/pred.test",
#     help="Test set location"
# )
# optparser.add_option(
#     '--score', default='evaluation/temp/score.txt',
#     help='score file location'
# )
# optparser.add_option(
#     "-g", '--use_gpu', default='1',
#     type='int', help='whether or not to ues gpu'
# )
# optparser.add_option(
#     '--loss', default='loss.txt',
#     help='loss file location'
# )
# optparser.add_option(
#     '--model_path', default='models/test',
#     help='model path'
# )
# optparser.add_option(
#     '--map_path', default='models/mapping.pkl',
#     help='model path'
# )
# optparser.add_option(
#     '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
#     help='char_CNN or char_LSTM'
# )
#
# opts = optparser.parse_args()[0]
#
# model = torch.load(opts.model_path)
# model_name = opts.model_path.split('/')[-1].split('.')[0]
#
#
#
# summary = 'Buffer overflow in the Windows Redirector function in Microsoft Windows XP allows local users to execute arbitrary code via a long parameter.'
# summary_token = summary_deal(summary,1)
# prelist = summary_token.split(" ")
# prelist.append('.')
# padinglist = ['O']*len(prelist)
# tmp_array = np.vstack((prelist,padinglist))
# lable_array = tmp_array.T
#
#
# np.savetxt(opts.test,lable_array,fmt="%s") #将转制后的矩阵存为txt
# print(lable_array)
# # eval(model, test_data)
#
#
# def get_vendor_product(cve_summary):
#     vendor = ''
#     product = ''
#
#     # 使用 NER-BiLSTM-CRF 训练的模型参数，针对cve_summary进行预测
#
#
#
#     # 返回预测结果
#     return vendor, product