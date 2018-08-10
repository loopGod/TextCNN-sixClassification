# TextCNN-sixClassification
一个政府公文的文本分类比赛，TextCNN的CV可以达到99%准确率

分成了6类'agriculture', 'commerce', 'culture', 'education', 'others', 'transportation'在data/cnews_loader.py里面有定义

mytraining.txt和myVal.txt是训练集和验证集，本来是很大的上传不了就保留了几篇文章看看格式即可，把这个txt改成自己的txt，格式一致，再修改data/cnews_loader.py里面的分类类别和model.py里面的分类数就可以训练自己的数据了

run_cnn训练，默认是带gpu的，predict.py的main里面内容很乱不对，但是predict(message)函数是对的，message是文章，调用predict(message)可以直接得到文章分类。


                  ————————LoopGod刘刚

