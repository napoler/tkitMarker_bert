# tkitMarker_bert
tkitmarker_bert

# 安装
```
pip install tkitMarker_bert
```



# 测试

```
from tkitMarker_bert import Marker

text="柯基犬是一个小短腿"
word="柯基犬"
#加载预测描述
pred=Marker(model_path="./model")
model,tokenizer=pred.load_model()
pall=pred.pre(word,text,model,tokenizer)
print(word,pall)
>>>柯基犬 ['是一个小短腿']

```
# 训练数据标记
https://github.com/napoler/tmark_Description


## 模型
一个小模型，标记数据不是很多
https://www.kaggle.com/terrychanorg/tmarkdescriptionmodel

# 更多返回

[By Terry Chan](https://www.terrychan.org/)




