# NER-BiLSTM-CRF-PyTorch
这是一个用于复现论文：Automated CPE Labeling of CVE Summaries with Machine Learning 的工程。

## 快速开始

1、配置环境：

```python
pip install -r requirements.txt
```

2、运行get_pretrained.sh下载glove词嵌入预训练模型并解压

3、运行run.sh脚本



ps：在linux下训练的步骤

```shell
nohup python3 -m visdom.server> web.log 2>&1 & #后台运行visdom可视化工具（必须）

nohup python -u train.py > 1test.log 2>&1 &#后台运行训练程序，log存至1test.log
```



## 工程结构

/data---存放训练用的数据

/src---存放主要代码

train.py---训练相关的代码，运行它可以对模型进行训练与测试，训练结果的记录使用tensorboard保存至output文件夹下

```shell
启动tensorboard（切换至output文件夹下）
tensorboard --logdir "./eval" 
```

diy_config.py---用于设置模型和训练相关的各种参数

eval.py---评估模型的脚本

loader.py---加载数据的脚本

utils.py---存放一些可供调用的工具函数

## References
- https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

- https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch

  

