import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# python -m visdom.server
# 训练之前先启动可视化模块
import sys
import visdom
from loss_data_list import lossdata_list
from Fscore_data_list import Fdata_list
import numpy as np


vis = visdom.Visdom()

def drawloss(losslist,interval):
    '''
    绘制loss曲线
    '''
    text = "<p>" + "</p><p>".join([str(l) for l in losslist[-9:]]) + "</p>"
    losswin = "loss_" + "cve_ner_models"
    textwin = "loss_text_" + "cve_ner_models"
    vis.line(
        np.array(losslist),
        X=np.array([interval * i for i in range(len(losslist))]),
        win=losswin,
        opts={
            "title": losswin,
            "legend": ["loss"]
        },
    )
    vis.text(text, win=textwin, opts={"title": textwin}) #visdom可视化相关

def drawFscore(Fscore,interval):
    """
    绘制模型在训练集、交叉验证集、测试集上的F1值，F1值由脚本conlleval计算得出
    """
    Fwin = "F-score of {train, dev, test}_" + "cve_ner_models"
    vis.line(
        np.array(Fscore),
        win=Fwin,
        X=np.array([interval * i for i in range(len(Fscore))]),
        opts={
            "title": Fwin,
            "legend": ["train", "dev", "test"]
        },
    )


if __name__ == "__main__":
    plot_every = 100 #绘图时的横坐标，间隔100绘制一个loss点
    eval_every = 200
    # testlist = [0.3772641985990339, 0.3772641985990339, 0.2636281775267217, 0.16108094010858703, 0.21519573784577634, 0.14043518340568378, 0.17065463636566314, 0.16769745772940273, 0.13736119451811973]
    for tup in lossdata_list:
        # print(tup[1])
        testlist = tup[1]
        C = tup[0]
        drawloss(testlist, plot_every)
    for Flist in Fdata_list:
        drawFscore(Flist, eval_every)
    losstup = lossdata_list[-1] #将最后一次更新的绘图展示出来
    plt.plot(losstup[1])
    plt.show()