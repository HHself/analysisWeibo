#coding:utf-8

import time
import pandas as pd
import numpy as np
import time
import re
import random
# import jieba
# import jieba.posseg as pseg
from snownlp import SnowNLP

filepath = "/home/weibo/"
output = "./output/"
filepath2 = "./output2/"
output2 = "./result/"
filepath3 = "./output3/"
filepath4 = "./output4/"
#stopwords = [s.replace("\n", "").decode("utf-8") for s in file("stopwords.txt")]

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
                filee.write(" ".join([str(s) for s in line]) + "\n")
        else:
            for i in data:
                filee.write(str(i) + "\n")
    elif isinstance(data, str):
        filee.write(data)
    else:
        print "not list or dict"
    filee.close()

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    #  value为传入的值为时间戳(整形)，如：1332888820
    value = time.localtime(value)
    ## 经过localtime转换后变成
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=0)
    # 最后再经过strftime函数转换为正常日期格式。
    dt = time.strftime(format, value)
    return str(dt)[:16]

#发现所有消息中的时间戳
def findtimeinfo():
    times = {}
    num = 0
    csvcontent = ["消息ID", "用户ID", "用户名", "屏幕名", "用户头像", "转发消息ID", "消息内容", "消息URL", "来源", "图片URL", "音频URL", "视频URL", "转发数", "评论数", "发布时间", "@用户"]
    for line in file(filepath + "content.txt"):
        #print times
        num += 1
        if num%500 == 0: print num
        try:
            d = pd.read_table(filepath + line.replace("\n",""))
            for t in d["发布时间"].values:
                ti = timestamp_datetime(t)[:7]
                #times.append(ti)
                times.setdefault(ti, 0)
                times[ti] += 1

            #times = list(set(times))
        except:
            print "illegal file..."
            continue

    return sorted(times.iteritems(), key = lambda x:x[1], reverse = True)

#取出2012年的微博消息
def find2012msg():
    maxline = 150000
    num = 0
    numm = 0
    # precol = ["用户ID", "用户名", "屏幕名", "消息内容", "来源", "转发数", "评论数", "发布时间", "@用户"]
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    csvcontent = ["消息ID", "用户ID", "用户名", "屏幕名", "用户头像", "转发消息ID", "消息内容", "消息URL", "来源", "图片URL", "音频URL", "视频URL", "转发数", "评论数", "发布时间", "@用户"]

    newFrame = pd.DataFrame(columns = precol)
    for line in file(filepath + "content.txt"):
        num += 1
        if num%500 == 0: print num
        try:
            d = pd.read_table(filepath + line.replace("\n",""))
            d = d.drop(list(set(csvcontent) - set(["用户ID", "用户名", "屏幕名", "消息内容", "来源", "转发数", "评论数", "发布时间", "@用户"])), axis = 1)
            d.columns = precol

            d1 = d[d.releasetime >= 1325347200]
            d2 = d1[d1.releasetime <= 1356839999]
            d2["releasetime"] = d2["releasetime"].apply(timestamp_datetime)
            #print d2['releasetime']
            d2 = d2.drop_duplicates()

            d2["msglen"] = d2.msginfo.apply(lambda x:len(str(x).decode("utf-8")))
            d3 = d2[d2.msglen > 10]
            d4 = d3.drop("msglen", axis = 1)
            newFrame = pd.concat([newFrame, d4])

            if len(newFrame)  > maxline:
                #print newFrame
                newFrame.iloc[:maxline, :].to_csv("./output2/2012weibodata_num_" + str(numm) +".csv", encoding="utf-8", index = False)
                numm +=1
                newFrame = newFrame.iloc[maxline:, :]
            #if numm >2: break
        except:
            print "illegal file: ",line

#basic statistics for 2012 msg
def tongji_time():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    time_24 = {} #24 hour
    time_12 = {} #12 month
    num = 0
    for line in file(filepath2 + "content.txt"):
        print line
        num +=1
        if num%100==0: print num
        try:
            d = pd.read_csv(filepath2 + line.replace("\n",""))
            
            d_time_24 = pd.DataFrame(d['releasetime'].apply(lambda x :x[11:13]))
            d_time_24["countt"] = d["commentNum"]
            d_time_24 = d_time_24.groupby('releasetime').count()
            for nu in d_time_24.index:
                    time_24.setdefault(nu, 0)
                    time_24[nu] += int(d_time_24["countt"][nu])
    
            d_time_12 = pd.DataFrame(d['releasetime'].apply(lambda x :x[5:7]))
            d_time_12["countt"] = d["commentNum"]
            d_time_12 = d_time_12.groupby('releasetime').count()
            for nu in d_time_12.index:
                    time_12.setdefault(nu, 0)
                    time_12[nu] += int(d_time_12["countt"][nu])
        except:
        	print line
	writefile(time_24, output2 + "time_24_hour.txt")
	writefile(time_12, output2 + "time_12_month.txt")


def tongji_source():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    data_source = {} #24 hour 
    num = 0
    for line in file(filepath2 + "content.txt"):
        print line
        num +=1
        if num%100==0: print num
        try:
            d = pd.read_csv(filepath2 + line.replace("\n",""))
            d_source = pd.DataFrame(d['source'])
            d_source["countt"] = d["commentNum"]
            d_source = d_source.groupby('source').count()
            for nu in d_source.index:
                    data_source.setdefault(nu, 0)
                    data_source[nu] += int(d_source["countt"][nu])
        except:
        	print line
    temp_source= sorted(data_source.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    writefile({i[0]:i[1] for i in temp_source}, output2 + "data_source.txt")

def tongji_userfre():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    data_user = {} #
    num = 0
    for line in file(filepath2 + "content.txt"):
        #print line
        num +=1
        if num%100==0: print num
        try:
            d = pd.read_csv(filepath2 + line.replace("\n",""))
            d_user = pd.DataFrame(d['screenname'])
            d_user["countt"] = d["commentNum"]
            d_user = d_user.groupby('screenname').count()
            for nu in d_user.index:
                    data_user.setdefault(nu, 0)
                    data_user[nu] += int(d_user["countt"][nu])
        except:
        	print line
    temp_userfre = sorted(data_user.iteritems(), key = lambda x:x[1], reverse = True)
    writefile({i[0]:i[1] for i in temp_userfre}, output2 + "data_userfre.txt")

# def cutwords_stars(sentence): 
# 	stars = [s.decode("utf-8") for s in file("stars.txt")]
# 	stopwords = [s.replace("\n", "").decode("utf-8") for s in file("stopwords.txt")]
# 	cutcontent = [w for w in list(jieba.cut(sentence, cut_all = False)) if w not in stopwords and w in stars]
# 	return cutcontent
def cutwords_stars(sentence):
    stars = [s.replace("\n", "") for s in file("stars.txt")]
    cutcontent = [w for w in stars if w in sentence]
    return cutcontent

def cutwords_keywords(sentence):
    s = SnowNLP(sentence.decode("utf-8"))
    keyw = s.keywords(1)
    if len(keyw) == 0 or keyw[0] in stopwords or len(keyw[0]) ==1 or len(keyw[0]) >8: keyw ='None'
    else: keyw = keyw[0]
    return keyw

def cutwords_sentiment(sentence): 
    s = SnowNLP(sentence.decode("utf-8"))
    senti = s.sentiments
    st = 0
    if senti > 0.6:
        st = 1
    elif senti >= 0.4 and senti<=0.6:
        st = 0
    else:
        st = -1
    return st

def cutwords_area(sentence): 
    district = ["北京", "天津", "重庆", "上海", "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾"]
    #words = pseg.cut(sentence) #words
    #district_word = [w.word for w in words if str(w.flag) == 'ns']
    district_word = [w for w in district if w in sentence]
    
    return district_word

def cutwords_phone(sentence): 
   phones = ["三星", "小米", "苹果", "华为", "诺基亚", "联想", "索尼", "魅族", "酷派", "金立"]
   cutcontent = [w for w in phones if w in sentence]
   return cutcontent

def cutwords_internet(sentence): 
   internet = ["百度", "阿里", "腾讯", "京东", "网易", "搜狐", "新浪", "携程", "优酷"]
   cutcontent = [w for w in internet if w in sentence]
   return cutcontent

def cutwords_social(sentence): 
   social = ["微博", "知乎", "贴吧", "微信", "QQ", "糗事百科", "人人", "豆瓣", "世纪佳缘"]
   cutcontent = [w for w in social if w in sentence]
   return cutcontent

def cutwords_sentiword(sentence): 
   sentiword = ["爱", "恨", "开心", "伤心", "哈哈", "唉", "么么哒", "萌萌哒"]
   cutcontent = [w for w in sentiword if w in sentence]
   return cutcontent

def cutwords_stock(sentence):
    stock = ["股市", "股票", "炒股", "涨", "跌", "涨停", "跌停", "被套了"]
    cutcontent = [w for w in stock if w in sentence]
    return cutcontent

def tongji_msg():
    # stopwords = [s.replace("\n", "").decode("utf-8") for s in file("stopwords.txt")]
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    num = 0
    data_stars = {}
    data_keywords = {}
    data_sentiment = {+1:0, 0:0, -1:0}
    data_area = {}
    data_phone = {}
    data_internet = {}
    data_social = {}
    data_sentiword = {}
    data_stock = {}

    # filelist = list(file(filepath3 + "content.txt"))
    # randlist = random.sample(filelist, 30)
    randlist = list(file(filepath4 + "content.txt"))
    for line in randlist:
        print num, line 
        # num +=1
        # if num%100==0: print num
        if num>2: break
        #try:
        d = pd.read_csv(filepath4 + line.replace("\n",""))
        d = d.drop(["userID", "username", "screenname", "source", "forwardNum", "commentNum", "etuser"], axis = 1)

        # d["cut_stars"] = d['msginfo'].map(cutwords_stars)
        # print [d['msginfo'][ind], d['msginfo'][ind]]
        
        # d["cut_sentiment"] = d['msginfo'].map(cutwords_sentiment)
        # d = d.groupby("cut_sentiment").count()
        filter_mathod = lambda row: "2012-07" in row['releasetime'] or "2012-08" in row['releasetime'] or "2012-09" in row['releasetime']
        d = d[d.apply(filter_mathod, axis = 1)]
        d["cut_keywords"] = d['msginfo'].map(cutwords_keywords)
        d = d.groupby("cut_keywords").count()
        # d["cut_area"] = d['msginfo'].map(cutwords_area)
        # d["cut_phone"] = d['msginfo'].map(cutwords_phone)
        # d["cut_internet"] = d['msginfo'].map(cutwords_internet)
        # d["cut_social"] = d['msginfo'].map(cutwords_social)
        # d["cut_sentiword"] = d['msginfo'].map(cutwords_sentiword)
        # d["cut_stock"] = d['msginfo'].map(cutwords_stock)

        for ind in d.index:
            # sta = d["cut_stars"][ind]
            # ar = d["cut_area"][ind]
            # data_sentiment[ind] += int(d["msginfo"][ind])
            # keyw = d["cut_keywords"][ind]
            if ind == "None": continue
            data_keywords.setdefault(ind, 0)
            data_keywords[ind] += int(d["msginfo"][ind])
            # ph = d["cut_phone"][ind]
            # inte = d["cut_internet"][ind]
            # so = d["cut_social"][ind]
            # sentiw = d["cut_sentiword"][ind]
            # sto = d["cut_stock"][ind]

            # for s in sta:
        	    # data_stars.setdefault(s, 0)
        	    # data_stars[s] += 1
            # for a in ar:
            #     data_area.setdefault(a, 0)
            #     data_area[a] += 1 
            
            # for se in senti:
                # data_sentiment.setdefault(se, 0)
                # data_sentiment[se] += 1 
            # for k in keyw:
            #     if k in stopwords:continue        
            #     data_keywords.setdefault(k, 0)
            #     data_keywords[k] += 1
            # for p in ph:
            #     data_phone.setdefault(p, 0)
            #     data_phone[p] += 1
            #     data_internet.setdefault(i, 0)
            # for i in inte:
            #     data_internet[i] += 1
            # for s in so:
            #     data_social.setdefault(s, 0)
            #     data_social[s] += 1
            # for sen in sentiw:
            #     data_sentiword.setdefault(sen, 0)
            #     data_sentiword[sen] += 1 
            # for s in sto:
            #     data_stock.setdefault(s, 0)
            #     data_stock[s] += 1
     
        # except:
        # 	print line

    # data_stars  = sorted(data_stars.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    # data_area = sorted(data_area.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    # data_sentiment  = sorted(data_sentiment.iteritems(), key = lambda x:x[1], reverse = True)
    data_keywords = sorted(data_keywords.iteritems(), key = lambda x:x[1], reverse = True)[:10000]
    # data_phone  = sorted(data_phone.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    # data_internet = sorted(data_internet.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    # data_social  = sorted(data_social.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    # data_sentiword = sorted(data_sentiword.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    # data_stock = sorted(data_stock.iteritems(), key = lambda x:x[1], reverse = True)
    
    # writefile({i[0]:i[1] for i in data_stars}, output2 + "data_stars.txt")
    # writefile({i[0]:i[1] for i in data_area}, output2 + "data_area.txt")
    # writefile({i[0]:i[1] for i in data_sentiment}, output2 + "data_sentiment.txt")
    writefile({i[0]:i[1] for i in data_keywords}, output2 + "data_keywords_789_2.txt")
    # writefile({i[0]:i[1] for i in data_phone}, output2 + "data_phone.txt")
    # writefile({i[0]:i[1] for i in data_internet}, output2 + "data_internet.txt")
    # writefile({i[0]:i[1] for i in data_social}, output2 + "data_social.txt")
    # writefile({i[0]:i[1] for i in data_sentiword}, output2 + "data_sentiword.txt")
    # writefile({i[0]:i[1] for i in data_stock}, output2 + "data_stock.txt")

def fiteret(sentence):
    targ = re.sub(r'#.*#|@.*,|@.* |http://.*', '', sentence)
    return targ

def filtertopuser():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    topuser = [line.split("\t")[0] for line in file("./result/data_userfre_del.txt")]
    maxline = 150000
    num = 0
    numm = 0
    
    newFrame = pd.DataFrame(columns = precol)
    for line in file(filepath3 + "content.txt"):
        num += 1
        if num%100 == 0: print num
        # try:
        d = pd.read_csv(filepath3 + line.replace("\n",""))
        filter_mathod = lambda row: r'//@' not in row['msginfo'] and not row['msginfo'].startswith('【') and not row['msginfo'].startswith('#') and len(row['msginfo']) > 30 and "此微博已被删除" not in row['msginfo'] and "分享图片" not in row['msginfo']
        d = d[d.apply(filter_mathod, axis = 1)]
        d = d.drop_duplicates(["msginfo"])
        
        # d = d[d.apply(lambda row: not row['msginfo'].startswith('【'), axis = 1)]
        newFrame = pd.concat([newFrame, d])
        if len(newFrame) > maxline:
            newFrame.iloc[:maxline, :].to_csv("./output3/2012weibodata_num_" + str(numm) +".csv", encoding="utf-8", index = False)
            numm +=1
            newFrame = newFrame.iloc[maxline:, :]
        # if numm >3: break
        # except:
        #     print "illegal file: ",line
def gethalfyear():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    # topuser = [line.split("\t")[0] for line in file("./result/data_userfre_del.txt")]
    maxline = 150000
    num = 0
    numm = 0
    
    newFrame = pd.DataFrame(columns = precol)
    for line in file(filepath3 + "content.txt"):
        num += 1
        if num%100 == 0: print num
        try:
            d = pd.read_csv(filepath3 + line.replace("\n",""))
        
            filter_mathod = lambda row: "2012-07" in row['releasetime'] or "2012-08" in row['releasetime'] or "2012-09" in row['releasetime'] or "2012-10" in row['releasetime'] or "2012-11" in row['releasetime'] or "2012-12" in row['releasetime']
            d = d[d.apply(filter_mathod, axis = 1)]
            
            newFrame = pd.concat([newFrame, d])
            if len(newFrame) > maxline:
                newFrame.iloc[:maxline, :].to_csv("./output4/2012weibodata_num_" + str(numm) +".csv", encoding="utf-8", index = False)
                numm +=1
                newFrame = newFrame.iloc[maxline:, :]
            # if numm >2: break
        except:
            print "illegal file: ",line
#------------------------------------second------------------------------
def filteractivity(sent):
    res = "No Activity"
    m = re.search(r'#.*#',sent)
    if m:
        res = m.group(0)
    return res
def getactivity():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    data_source = {}  
    num = 0
    for line in file(filepath2 + "content.txt"):
        print line
        num +=1
        if num%100==0: print num
        # try:
        d = pd.read_csv(filepath2 + line.replace("\n",""))
        d_source = pd.DataFrame(d['msginfo'])
        d_source["activity"] = d_source['msginfo'].map(filteractivity)
        d_source["activity_num"] = d["commentNum"]
        d_source= d_source.groupby('activity').count()


        for nu in d_source.index:
            # print nu,d_source['activity_num'][nu]
            data_source.setdefault(nu, 0)
            data_source[nu] += int(d_source["activity_num"][nu])
        # for index,row in d_source.iterrows():
        #     print "come index"
        #     print row['activity'],row['num']
        #     data_source.setdefault(row['activity'], 0)
        #     data_source[row['activity']] += int(row['num'])
        # except:
        #     print line
        break 
    temp_source= sorted(data_source.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    print temp_source
    writefile({i[0]:i[1] for i in temp_source},  "activity.txt")

if __name__ =="__main__":
    #find2012msg()
    #tongji_time()
    #tongji_source()
    # tongji_userfre()
    # tongji_msg()
    # filtertopuser()
    # gethalfyear()
    getactivity()