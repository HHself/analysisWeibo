import datasets as ds
import math
import jieba
import jieba.posseg as pseg
from sklearn import cluster,metrics
import BTMmodel as bm
import scipy.spatial.distance
import numpy 
import json
import pandas as pd


def readfile(path):
    data=[]
    for line in file(path):
        data.append(line)
    return data

def writefile(data, path):
    filee=open(path,'w')
    if isinstance(data,dict):
        for i in data.keys():
            filee.write(str(i) + "\t"+str(data[i]) + "\n")
    elif isinstance(data,list):
        if isinstance(data[0], list):
            for line in data:
                #filee.write(" ".join([s.encode("utf-8") for s in line])+"\n")
                filee.write(" ".join(line) + "\n")
        else:
            for i in data:
                filee.write(str(i[0])+" "+str(i[1]) + "\n")
    elif isinstance(data, str):
        filee.write(data)
    else:
        print "not list or dict"
    filee.close()

def getactivitydata():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    data_source = []  
    num = 0
    for line in file(filepath2 + "content.txt"):
        num +=1
        if num%100==0: print num
        try:
            d = pd.read_csv(filepath2 + line.replace("\n",""))
            d_source = pd.DataFrame(d['msginfo'])
            filter_mathod = lambda row: row['msginfo'].startswith("#美女车模#") or row['msginfo'].startswith("#游戏美女#") or row['msginfo'].startswith("#三国来了#")or row['msginfo'].startswith("#伦敦奥运#")or row['msginfo'].startswith("#中国好声音#")or row['msginfo'].startswith("#IT新闻#")or row['msginfo'].startswith("#汽车知道#")or row['msginfo'].startswith("#美食#")or row['msginfo'].startswith("#台风海葵#")or row['msginfo'].startswith("#爱情公寓#")
            d_source = d_source[d_source.apply(filter_mathod, axis = 1)]
        except:
            print line
        data_source += list(d_source['msginfo'])
    
    writefile(data_source, "activity_10_data.txt")

def cutallcontent(sent):    
    stopwords = [line.replace("\n","").replace("\r","").decode("utf-8") for line in file("stopwords.txt")]
    cut_content = list(jieba.cut(sent.decode("utf-8"), cut_all = False))
    cut_content = [word.encode("utf-8") for word in cut_content if len(word)>1 and word not in stopwords]
    return cut_content


def getcutcontent():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    data_source = {}  
    num = 0
    for line in file("./output4/content.txt"):
        num +=1
        if num%10==0: print num
        try:
            d = pd.read_csv("./output4/" + line.replace("\n",""))
            d_source = pd.DataFrame(d['msginfo'])
            d_source["cut_content"] =  d['msginfo'].map(cutallcontent)
        except:
            print line
    d_source.drop(["msginfo"])
    d.source.to_csv("./result/cut_content.csv", encoding="utf-8", index = False)

def kmeanscluster(self, pd_z, k):
    k_means = cluster.KMeans(k)
    k_means.fit_transform(pd_z)
    return k_means.labels_

def calcF1(labels_true,labels_alg):
    # jc = lambda x: x(x-1)/2
    # TP_FP = 0
    # for l in labels_alg:
    #     TP_FP += jc(len(l))
    return metrics.f1_score(labels_true, labels_alg, average='weighted')


if __name__ == "__main__":
    # getcutcontent()
    getactivitydata()

