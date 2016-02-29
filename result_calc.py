import jieba
import pandas as pd

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
        if num%100==0: print num
        try:
            d = pd.read_csv("./output4/" + line.replace("\n",""))
            d_source = pd.DataFrame(d['msginfo'])
            d_source["cut_content"] =  d['msginfo'].map(cutallcontent)
        except:
            print line
    d_source.drop(["msginfo"])
    d.source.to_csv("./result/cut_content.csv", encoding="utf-8", index = False)

if __name__ == "__main__":
    getcutcontent()

