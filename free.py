import torch
from tkitMarker_bert import Marker
from memory_profiler import profile
import gc
@profile
def my_func():
    text="柯基犬是一个小短腿"
    word="柯基犬"
    #加载预测描述
    pred=Marker(model_path="./model")
    # model,tokenizer=pred.load_model()
    pred.load_model()
    pall=pred.pre(word,text)
    print(word,pall)
    # model.cpu()
    # pred.release()
    del pred
    torch.cuda.empty_cache()
    gc.collect()
    mem_alloc = torch.cuda.memory_allocated()
    mem_cache = torch.cuda.memory_cached()
    print("mem_alloc",mem_alloc)
    print("mem_cache",mem_cache)

if __name__ == '__main__':
    my_func()
# 运行
# python -m memory_profiler free.py