#coding:utf-8

import time
import pandas as pd
import numpy as np
import time
import jieba
import jieba.posseg as pseg
#from snownlp import SnowNLP

filepath = "/home/weibo/"
output = "./output/"
filepath2 = "./output2/"
output2 = "./result/"

def readfile(path):
    data=[]
    for line in file(path):
        data.append(line)
    return data

def writefile(data, path):
    filee=open(path,'w')
    if isinstance(data,dict):
        for i in data:
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
    temp_userfre = sorted(data_user.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    writefile({i[0]:i[1] for i in temp_userfre}, output2 + "data_userfre.txt")

# def cutwords_stars(sentence): 
# 	stars = [s.decode("utf-8") for s in file("stars.txt")]
# 	stopwords = [s.replace("\n", "").decode("utf-8") for s in file("stopwords.txt")]
# 	cutcontent = [w for w in list(jieba.cut(sentence, cut_all = False)) if w not in stopwords and w in stars]
# 	return cutcontent
def cutwords_stars(sentence): 
  stars = [s for s in file("stars.txt")]
  cutcontent = [w for w in stars if w in sentence]
  return cutcontent

def cutwords_keywords(sentence):
	s = SnowNLP(sentence.decode("utf-8"))
	return s.cut_keywords(3)

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
    district = [u"北京", u"天津", u"重庆", u"上海", u"河北", u"山西", u"辽宁", u"吉林", u"黑龙江", u"江苏", u"浙江", u"安徽", u"福建", u"江西", u"山东", u"河南", u"湖北", u"湖南", u"广东", u"海南", u"四川", u"贵州", u"云南", u"陕西", u"甘肃", u"青海", u"台湾"]
    words = pseg.cut(sentence)
    district_word = [w.word for w in words if str(w.flag) == 'ns']
    return district_word

def cutwords_phone(sentence): 
   phones = [u"三星", u"小米", u"苹果", u"华为", u"诺基亚", u"联想", u"索尼", u"魅族", u"酷派", u"金立"]
   cutcontent = [w for w in stars if w in sentence]
   return cutcontent

def cutwords_internet(sentence): 
   internet = [u"百度", u"阿里", u"腾讯", u"京东", u"网易", u"搜狐", u"新浪", u"携程", u"优酷"]
   cutcontent = [w for w in stars if w in sentence]
   return cutcontent

def cutwords_social(sentence): 
   social = [u"微博", u"知乎", u"贴吧", u"微信", u"QQ", u"糗事百科", u"人人", u"豆瓣", u"世纪佳缘"]
   cutcontent = [w for w in stars if w in sentence]
   return cutcontent

def cutwords_sentiword(sentence): 
   sentiword = [u"爱", u"恨", u"开心", u"伤心", u"哈哈", u"唉", u"么么哒", u"萌萌哒"]
   cutcontent = [w for w in stars if w in sentence]
   return cutcontent


def tongji_msg():
    precol = ["userID", "username", "screenname", "msginfo", "source", "forwardNum", "commentNum", "releasetime", "etuser"]
    num = 0
    data_stars = {}
    data_keywords = {}
    data_sentiment = {}
    data_area = {}
    data_phone = {}
    data_internet = {}
    data_social = {}
    data_sentiword = {}
    for line in file(filepath2 + "content.txt"):
        #print line
        num +=1
        if num%100==0: print num
        if num>3: break
        #try:
        d = pd.read_csv(filepath2 + line.replace("\n",""))
        d = d.drop(["userID", "username", "screenname", "source", "forwardNum", "commentNum", "releasetime"], axis = 1)
        d["cut_stars"] = d['msginfo'].apply(cutwords_stars)
        d["cut_keywords"] = d['msginfo'].apply(cutwords_keywords)
        d["cut_sentiment"] = d['msginfo'].apply(cutwords_sentiment)
        d["cut_area"] = d['msginfo'].apply(cutwords_area)
        d["cut_phone"] = d['msginfo'].apply(cutwords_phone)
        d["cut_internet"] = d['msginfo'].apply(cutwords_internet)
        d["cut_social"] = d['msginfo'].apply(cutwords_social)
        d["cut_sentiword"] = d['msginfo'].apply(cutwords_sentiword)
        for ind in d.index:
            sta = d["cut_stars"][ind]
            ar = d["cut_area"][ind]
            senti = d["cut_sentiment"][ind]
            keyw = d["cut_keywords"][ind]
            ph = d["cut_phone"][ind]
            inte = d["cut_internet"][ind]
            so = d["cut_social"][ind]
            sentiw = d["cut_sentiword"][ind]

            for s in sta:
        	    data_stars.setdefault(s, 0)
        	    data_stars[s] += 1
            for a in ar:
                data_area.setdefault(a, 0)
                data_area[a] += 1 
            for se in senti:
                data_sentiment.setdefault(se, 0)
                data_sentiment[se] += 1 
            for k in keyw:                
                data_keywords.setdefault(k, 0)
                data_keywords[keyw] += 1
            for p in ph:
                data_phone.setdefault(p, 0)
                data_phone[p] += 1
            for i in inte:
                data_internet.setdefault(i, 0)
                data_internet[i] += 1
            for s in so:
                data_social.setdefault(s, 0)
                data_social[s] += 1
            for sen in sentiw:
                data_sentiword.setdefault(sen, 0)
                data_sentiword[sen] += 1 
     
        #except:
        #	print line

    data_stars  = sorted(data_stars.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    data_area = sorted(data_area.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    data_sentiment  = sorted(data_sentiment.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    data_keywords = sorted(data_keywords.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    data_phone  = sorted(data_phone.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    data_internet = sorted(data_internet.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    data_social  = sorted(data_social.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    data_sentiword = sorted(data_sentiword.iteritems(), key = lambda x:x[1], reverse = True)[:50]
    
    writefile({i[0]:i[1] for i in data_stars}, output2 + "data_stars.txt")
    writefile({i[0]:i[1] for i in data_area}, output2 + "data_area.txt")
    writefile({i[0]:i[1] for i in data_sentiment}, output2 + "data_sentiment.txt")
    writefile({i[0]:i[1] for i in data_keywords}, output2 + "data_keywords.txt")
    writefile({i[0]:i[1] for i in data_phone}, output2 + "data_phone.txt")
    writefile({i[0]:i[1] for i in data_internet}, output2 + "data_internet.txt")
    writefile({i[0]:i[1] for i in data_social}, output2 + "data_social.txt")
    writefile({i[0]:i[1] for i in data_sentiword}, output2 + "data_sentiword.txt")




if __name__ =="__main__":
    #find2012msg()
    #tongji_time()
    #tongji_source()
    #tongji_userfre()
    tongji_msg()


#抱歉，此微博已被作者删除 "分享图片"








'''
        for t in d.index:
            distime = int(d["发布时间"][t])
            if distime < 1325347200 or dis > 1356839999: continue
                newFrame["用户ID"][numm] =
                newFrame["用户名"][numm] = 
                newFrame["屏幕名"][numm] = 
                newFrame["消息内容"][numm] =
                newFrame["来源"][numm] = 
                newFrame["转发数"][numm] =
                newFrame["评论数"][numm] =
                newFrame["发布时间"] =
                newFrame["@用户"] =
        d.rename(columns = {"用户ID":"userID"}, inplace = True)
            d.rename(columns = {"用户名":"username"}, inplace = True)
            d.rename(columns = {"屏幕名":"screenname"}, inplace = True)
            d.rename(columns = {"消息内容":"msginfo"}, inplace = True)
            d.rename(columns = {"来源":"source"}, inplace = True)
            d.rename(columns = {"转发数":"forwardNum"}, inplace = True)
            d.rename(columns = {"评论数":"commentNum"}, inplace = True)
            d.rename(columns = {"发布时间":"releasetime"}, inplace = True)
            d.rename(columns = {"@用户":"etuser"}, inplace = True) 
'''