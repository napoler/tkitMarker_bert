# import sqlite3
import os
import pymongo
# from albert_pytorch import classify
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl import Q
# from  tkitMarker import  *

#这里定义mongo数据
client = pymongo.MongoClient("localhost", 27017)
DB = client.gpt2Write
print(DB.name)
# DB.my_collection
# Collection(Database(MongoClient('localhost', 27017), u'test'), u'my_collection')
# print(DB.my_collection.insert_one({"x": 10}).inserted_id)

def search_content(keyword):
    client = Elasticsearch()
    q = Q("multi_match", query=keyword, fields=['title', 'body'])
    # s = s.query(q)

    # def search()
    s = Search(using=client)
    # s = Search(using=client, index="pet-index").query("match", content="金毛")
    s = Search(using=client, index="pet-index").query(q)
    response = s.execute()
    return response

