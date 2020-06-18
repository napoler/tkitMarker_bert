

import numpy as np
import torch
from transformers import AutoModelForTokenClassification,AutoTokenizer
import os
import re


import tkitFile

from config import *
from tqdm import tqdm
import time
import tkitText
from tkitMarker_bert import Marker



tt=tkitText.Text()
text="约克夏梗的头部经常用丝带装饰，并拥有梗类调皮的性格。约克夏梗拥有“上流 贵妇人香闺”般的魅力。"
word="约克夏梗"
#加载预测描述
pred=Marker(model_path="./model")
model,tokenizer=pred.load_model()
pall=pred.pre(word,text)
print(word,pall)

exit()

# 加载ner模型
ner_pred=Marker(model_path="./model/ner")
ner_pred_model,ner_pred_tokenizer=ner_pred.load_model()

while  True:
    print("\n"*4)
    print("输入文字中的实体和文字,提取关于实体的描述信息")
    keyword=input("输入搜索关键词:")
    for it in search_content(keyword):
        print("##"*10+it.title)
        text=it.title+"\n"+it.content
        words=ner_pred.pre_ner(text,ner_pred_model,ner_pred_tokenizer)
        words=list(set(words))
        for word in words:
            pall=pred.pre(word,text,model,tokenizer)
            print(word,pall)

# # i=0
# # for it in DB.content_pet.find({}):
# #     i=i+1
# #     if i==1000:
# #         break

# #     sents=tt.sentence_segmentation_v1(it['content'])
# #     print("##"*10+it['title'])
# #     text=it['title']+"\n"+it['content']
# #     words=ner_pred.pre_ner(text,ner_pred_model,ner_pred_tokenizer)
# #     # print(words)

# #     # result=TNer.pre(sents) 
# #     # # print(result)
# #     # words=[]
# #     # for it_re in result:
# #     #     for w in it_re[1]:
# #     #         # print("ner_w",ner_w)
# #     #         # for w in ner_w:
# #     #         print("w",w)
# #     #         if w['type']=="实体":
# #     #             words.append(w['words'])
# #     words=list(set(words))
# #     for word in words:
# #         pall=pred.pre(word,text,model,tokenizer)
# #         print(word,pall)
