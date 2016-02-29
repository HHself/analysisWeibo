import jieba
import pandas as pd

def cutallcontent(sent):    
    stopwords = [line.replace("\n","").replace("\r","") for line in file("stopwords.txt")]
    cut_content = [list(jieba.cut(line[1], cut_all = False)) for line in pid_context]
    cut_content = [[word.encode("utf-8") for word in doc if len(word)>2 and word not in stopwords] for doc in cut_content]
    return cut_content

def getcutcontent():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    data_source = {}  
    num = 0
    for line in file("./output4/content.txt"):
        num +=1
        if num%100==0: print num
        try:
            d = pd.read_csv(filepath4 + line.replace("\n",""))
            d_source = pd.DataFrame(d['msginfo'])
            d_source["cut_content"] =  d['msginfo'].map(cutallcontent)
        except:
            print line
    d_source.drop(["msginfo"])
    d.source.to_csv("./result/cut_content.csv", encoding="utf-8", index = False)

if __name__ == "__main__":
    getcutcontent()

