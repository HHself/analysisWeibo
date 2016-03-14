import lstm_rnn as lr
import pybrain
import random
import numpy as np
import copy
import re
import math

N = 128
M = 32 # the dim of wordvec
C = 64 # the number of cell
NW = 10 #the number of negitive sample
miu = 0.99  #momentum parameter
gama = 10 #scaling
era = 1.0 #learning rate
PN = 15 # the number of parameters

sigmoid = lambda x : 1.0/(1 + np.exp(-x))
tanh = lambda x : np.tanh(x)



#zhuanzhi 1*m ===>m*1
def transps1(vec):
    vec.shape = (vec.shape[0], 1)
    return vec

def transps2(vec):
    vec.shape = (1, vec.shape[0])
    return vec

def fmtoutput(y_ss, y_pp, y_nn):
    #make len(y_s) = len(y_p) = len(y_n) = maxlen
    maxlen = max(len(y_ss), len(y_pp), len(y_nn))
    while len(y_ss) < maxlen or len(y_pp) < maxlen or len(y_nn) < maxlen:
        if len(y_ss) < maxlen: y_ss.append([0]*N)
        if len(y_pp) < maxlen: y_pp.append([0]*N)
        if len(y_nn) < maxlen: y_nn.append([0]*N)
    return y_ss, y_pp, y_nn, maxlen
def jianyan(y_p, y_n):
    if len(y_p) != len(y_n):
            print " positive not equal neg..."
            return
    for t in range(1,C)   
        for i in range(len(y_p)):
        
            print y_p[i][t].shape, y_n[i][t].shape


def calgradient(param, y_s, y_p, y_n, data):
    #data[0]:source, data[1]:positive, data[2]:negative
    gra = [np.array(p) for p in [[[0 for j in range(M)] for i in range(N)]  for k in range(11)] + [[random.random() for p in range(N)] for q in range(4)]]
    lasts_p = [np.array(p) for p in [[0 for i in range(M)]  for k in range(6)] + [[0 for i in range(N)]  for k in range(10)] + [[[0 for j in range(N)] for i in range(N)]  for k in range(6)]]
    lasts_q = copy.deepcopy(lasts_p)

    # y_s, y_p, y_n, maxlen = fmtoutput(y_s, y_p, y_n)
    for t in range(1, C):
        if len(set(y_s[-1][t])) + len(set(y_s[-1][t])) + len(set(y_s[-1][t])) == 3 and sum(y_s[-1][t]) + sum(y_s[-1][t])+ sum(y_s[-1][t]) == 0:
            break
        gra_p, lasts_p = calgraR(param, y_s, y_p, lasts_p, data[0:2], t)
        jianyan(y_p, y_n)

        gra_q, lasts_q = calgraR(param, y_s, y_n, lasts_q, data[0:3:2], t)
        for k in range(PN):
            gra[k] += gra_p[k] - gra_q[k]

    return gra


def calgraR(param, yq, yd, lasts, data, tt):
    gra = [[] for i in range(15)]
    s = yq[-1][len(data[0])-1]
    t = yd[-1][len(data[1])-1]
    a = np.dot(s, t)
    b = 1.0 / np.linalg.norm(s)
    c = 1.0 / np.linalg.norm(t)
    vq = b * c * t - a * b**3 * c * s
    vd = b * c * s - a * b * c**3 * t

    # ---------------------------for output gate------------------------
    sigmarqt1 = yq[4][tt] * (1 - yq[4][tt]) * tanh(yq[3][tt]) * vq  #n*1
    sigmardt1 = yd[4][tt] * (1 - yd[4][tt]) * tanh(yd[3][tt]) * vd    

    gra_wr1 = np.dot(transps1(sigmarqt1), transps2(yq[-1][tt-1])) + np.dot(transps1(sigmardt1), transps2(yd[-1][tt-1]))
    gra_w1 =  np.dot(transps1(sigmarqt1), transps2(data[0][tt])) + np.dot(transps1(sigmardt1), transps2(data[1][tt]))
    gra_wp1 =  np.dot(transps1(sigmarqt1), transps1(yq[3][tt]).T) + np.dot(transps1(sigmardt1), transps1(yq[3][tt]).T)
    gra_b1 = sigmarqt1 + sigmardt1
    gra[4].append(gra_wr1)
    gra[0].append(gra_w1)
    gra[8].append(gra_wp1)
    gra[11].append(gra_b1)

    #  ---------------------------for input gate------------------------
    sigmart3 = lambda y_c, y_o, v : (1 - tanh(y_c)) * (1 + tanh(y_c)) * y_o * v
    bit = lambda ygt, it : ygt[tt] * it[tt] * (1-it[tt])
    syvq_ig = sigmart3(yq[3][tt].T, yq[4][tt], vq)
    syvd_ig = transps2(sigmart3(yd[3][tt], yd[4][tt], vd))
    grarall = lambda g_q, g_d : np.dot(syvq_ig, g_q) + np.dot(syvd_ig, g_d)

    gracwr3 = lambda ft, gracwr3_last, ygt, it, yt: np.dot(transps1(ft[tt]), transps2(gracwr3_last)) + np.dot(transps1(bit(ygt, it)), yt[tt-1]) 
    gracwr3_q = gracwr3(yq[2], lasts[9], yq[0], yq[1], yq[5])
    gracwr3_d = gracwr3(yd[2], lasts[8], yd[0], yd[1], yd[5])
    grwr3_last_q = copy.deepcopy(gracwr3_q)
    grwr3_last_d = copy.deepcopy(gracwr3_d)
    gra_wr3 = grarall(gracwr3_q, gracwr3_d)

    gracw3 = lambda ft, gracw3_last, ygt, it, i: np.dot(transps1(ft[tt]), transps2(gracw3_last)) + np.dot(transps1(bit(ygt, it)), transps2(data[i][tt-1])) 
    gracw3_q = gracw3(yq[2], lasts[3], yq[0], yq[1], 0)
    gracw3_d = gracw3(yd[2], lasts[2], yd[0], yd[1], 1)
    grw3_last_q = copy.deepcopy(gracw3_q)
    grw3_last_d = copy.deepcopy(gracw3_d)
    gra_w3 = grarall(gracw3_q, gracw3_d)

    gracwp3 = lambda ft, gracwp3_last, ygt, it, ct: np.dot(transps1(ft[tt]), transps2(gracwp3_last)) + np.dot(transps1(bit(ygt, it)), transps2(ct[tt-1])) 
    gracwp3_q = gracwp3(yq[2], lasts[15], yq[0], yq[1], yq[3])
    gracwp3_d = gracwp3(yd[2], lasts[14], yd[0], yd[1], yd[3])
    grwp3_last_q = copy.deepcopy(gracwp3_q)
    grwp3_last_d = copy.deepcopy(gracwp3_d)
    gra_wp3 = grarall(gracwp3_q, gracwp3_d)

    gracb3 = lambda ft, gracb3_last, ygt, it: np.dot(transps1(ft[tt]).T, gracb3_last) + bit(ygt, it)
    gracb3_q = gracb3(yq[2], lasts[19], yq[0], yq[1])
    gracb3_d = gracb3(yd[2], lasts[18], yd[0], yd[1])
    grb3_last_q = copy.deepcopy(gracb3_q)
    grb3_last_d = copy.deepcopy(gracb3_d)
    gra_b3 = grarall(gracb3_q.T, gracb3_d.T)

    gra[6].append(gra_wr3)
    gra[2].append(gra_w3)
    gra[10].append(gra_wp3)
    gra[13].append(gra_b3)
    #  ---------------------------for forget gate------------------------

    sigmart2 = lambda ct, ot, v : (1 - tanh(ct)) * (1 + tanh(ct) * ot * v)
    bft = lambda ct, ft : ct[tt-1].T * ft[tt] * (1-ft[tt])
    syvq_fg = sigmart2(yq[3][tt].T, yq[4][tt], vq)
    syvd_fg = transps2(sigmart2(yd[3][tt], yd[4][tt], vd))
    grarall_fg = lambda g_q, g_d : np.dot(syvq_fg, g_q) + np.dot(syvd_fg, g_d)


    gracwr2 = lambda ft, gracwr2_last, ct, yt: np.dot(ft[tt], transps2(gracwr2_last)) + np.dot(bft(ct, ft), yt[tt-1]) 
    gracwr2_q = gracwr2(yq[2], lasts[7], yq[3], yq[5])
    gracwr2_d = gracwr2(yd[2], lasts[6], yd[3], yd[5])
    grwr2_last_q = copy.deepcopy(gracwr2_q)
    grwr2_last_d = copy.deepcopy(gracwr2_d)
    gra_wr2 = grarall_fg(gracwr2_q, gracwr2_d) 

    
    gracw2 = lambda ft, gracw2_last, ct, i: np.dot(ft[tt], transps2(gracw2_last)) + np.dot(bft(ct, ft), transps2(data[i][tt-1]))
    gracw2_q = gracw2(yq[2], lasts[1], yq[3], 0)
    gracw2_d = gracw2(yd[2], lasts[0], yd[3], 1)
    grw2_last_q = copy.deepcopy(gracw2_q)
    grw2_last_d = copy.deepcopy(gracw2_d)
    gra_w2 = grarall_fg(gracw2_q, gracw2_d) 

    gracwp2 = lambda ft, gracwp2_last, ct: np.dot(ft[tt], transps2(gracwp2_last)) + np.dot(bft(ct, ft), ct[tt-1]) 
    gracwp2_q = gracwp2(yq[2], lasts[13], yq[3])
    gracwp2_d = gracwp2(yd[2], lasts[12], yd[3])
    grwp2_last_q = copy.deepcopy(gracwp2_q)
    grwp2_last_d = copy.deepcopy(gracwp2_d)
    gra_wp2 = grarall_fg(gracwp2_q, gracwp2_d) 
 
    gracb2 = lambda ft, gracb2_last, ct: np.dot(ft[tt].T, gracb2_last) + bft(ct, ft) 
    gracb2_q = gracb2(yq[2], lasts[17], yq[3])
    gracb2_d = gracb2(yd[2], lasts[16], yd[3])
    grb2_last_q = copy.deepcopy(gracb2_q)
    grb2_last_d = copy.deepcopy(gracb2_d)
    gra_b2 = grarall_fg(gracb2_q, gracb2_d)

    gra[5].append(gra_wr2)
    gra[1].append(gra_w2)
    gra[9].append(gra_wp2)
    gra[12].append(gra_b2)
    #  ---------------------------for input without gate------------------------
    sigmart4 = lambda ct, ot, v : (1 - tanh(ct)) * (1 + tanh(ct)) * ot * v
    bgt = lambda it, ygt : it[tt] * (1 - ygt[tt]) * (1 + ygt[tt])
    syvq_iwg = sigmart4(yq[3][tt].T, yq[4][tt], vq)
    syvd_iwg = transps2(sigmart4(yd[3][tt], yd[4][tt], vd))
    grarall_iwg = lambda g_q, g_d : np.dot(syvq_iwg, g_q) + np.dot(syvd_iwg, g_d)

    gracwr4 = lambda ft, gracwr4_last, it, ygt, yt: np.dot(ft[tt], transps2(gracwr4_last)) + np.dot(transps1(bgt(it, ygt)), yt[tt-1]) 
    gracwr4_q = gracwr4(yq[2], lasts[11], yq[1], yq[0], yq[5])
    gracwr4_d = gracwr4(yd[2], lasts[10], yd[1], yd[0], yd[5])
    grwr4_last_q = copy.deepcopy(gracwr4_q)
    grwr4_last_d = copy.deepcopy(gracwr4_d)
    gra_wr4 = grarall_iwg(gracwr4_q, gracwr4_d) 

    gracw4 = lambda ft, gracw4_last, it, ygt, i: np.dot(transps1(ft[tt]), transps2(gracw4_last)) + np.dot(transps1(bgt(it, ygt)), transps2(data[i][tt-1]))
    gracw4_q = gracw4(yq[2], lasts[5], yq[1], yq[0], 0)
    gracw4_d = gracw4(yd[2], lasts[4], yd[1], yd[0], 1)
    grw4_last_q = copy.deepcopy(gracw4_q)
    grw4_last_d = copy.deepcopy(gracw4_d)
    gra_w4 = grarall_iwg(gracw4_q, gracw4_d)

    gracb4 = lambda ft, gracb4_last, it, ygt: np.dot(ft[tt], gracb4_last) + bgt(it, ygt) 
    gracb4_q = gracb4(yq[2][tt].T, lasts[21], yq[1], yq[0])
    gracb4_d = gracb4(yd[2][tt].T, lasts[20], yd[1], yd[0])
    grb4_last_q = copy.deepcopy(gracb4_q)
    grb4_last_d = copy.deepcopy(gracb4_d)
    gra_b4 = grarall_iwg(gracb4_q.T, gracb4_d.T)

    gra[7].append(gra_wr4)
    gra[3].append(gra_w4)
    gra[14].append(gra_b4)
    
    # lasts = [gracw2_last, gracw3_last, gracw4_last, gracwr2_last, gracwr3_last, gracwr4_last, gracb2_last, gracb3_last, gracb4_last]
    lasts = [grw2_last_d, grw2_last_q, grw3_last_d, grw3_last_q, grw4_last_d, grw4_last_q, grwr2_last_d, grwr2_last_q, grwr3_last_d, grwr3_last_q, grwr4_last_d, grwr4_last_q, grwp2_last_d, grwp2_last_q, grwp3_last_d, grwp3_last_q, grb2_last_d, grb2_last_q, grb3_last_d, grb3_last_q, grb4_last_d, grb4_last_q]
    return  gra, lasts

def getlastoutput(param, textvec):
    lr_s = lr.LSTM_RNN(param, textvec)
    y_output = lr_s.lstmrun()
    return y_output

def BPTTtrain(worddict, parameters):
    weibo = [line for line in file("weibo_train.txt")]
    param_last = [np.array(q) for q in [[[0 for j in range(M)] for i in range(N)] for k in range(4)] + [[[0 for j in range(N)] for i in range(N)] for k in range(7)] + [[0 for p in range(N)] for q in range(4)]]
    param = []

    while 1:
        gradient = [np.array(q) for q in [[[0 for j in range(M)] for i in range(N)] for k in range(4)] + [[[0 for j in range(N)] for i in range(N)] for k in range(7)] + [[0 for p in range(N)] for q in range(4)]]
        for r in range(len(weibo)):
            if getacti(weibo[r]) == "None" : continue
            f = 0
            while True:
                data = gettraindata(worddict, r, weibo) #data[0]: source, data[1]:posotive, data[2:]:negatives
                for t in data:
                    if len(t) != 0: f += 1
                if f == len(data): break 
                f = 0

            for k in range(PN):
                param.append(parameters[k] + miu * (parameters[k] - param_last[k]))

            # print "HHHHHHHHH", len(data), len(data[0]), len(data[1]),
            y_s = getlastoutput(param, data[0])
            y_p = getlastoutput(param, data[1])
            cos_y_sp = cossim(y_s[-1][len(data[0])-1], y_p[-1][len(data[1])-1])
            # print "#####", len(data[0]), len(data[1]), y_s[-1][len(data[0])-1], '\n' ,y_p[-1][len(data[1])-1], '\n' ,cos_y_sp,'\n' ,type(cos_y_sp)
            # if len(data[1]) == 0: print data[:5]

            esum = 0
            cosy_spns = []
            for j in range(NW):
                y_n = getlastoutput(param, data[2+j])
                cos_y_sn = cossim(y_s[-1][len(data[0])-1], y_n[-1][len(data[2+j])-1])
                # print "%%%%%%%%%%%", y_n[-1][len(data[2+j])-1], '\n' ,cos_y_sn, '\n' ,type(cos_y_sn)
                # print "##", len(data[2+j])#, y_n[-1]
                cosy_spn = cos_y_sp - cos_y_sn

                # print y_s[-1][len(data[0])-1], y_p[-1][len(data[1])-1], y_n[-1][len(data[2+j])-1], cos_y_sp, cos_y_sn

                esum += np.exp(-1 * gama * cosy_spn)
                cosy_spns.append(cosy_spn)
            for j in range(NW):
                alpharj = (-1 * gama * np.exp(-1 * gama * cosy_spns[j])) / (1 + esum)
                g = calgradient(param, y_s, y_p, y_n, [data[0], data[1], data[2+j]])
                for k in range(PN):
                    gradient[k] += g[k]
        for k in range(PN):
            parameters[k] = parameters[k] + miu * (parameters[k]- param_last[k]) - era * gradient[k]
        param_last = copy.deepcopy(parameters) #parameters.copy()
    

    pa = ["w1", "w2", "w3", "w4", "wr1", "wr2", "wr3", "wr4", "wp1", "wp2", "wp3", "b1", "b2", "b3", "b4"]
    for i in range(15):
        parameters[i].tofile("./param/" + pa[i] + ".txt")


def gettraindata(worddict, i, weibo):
    tdata = []
    s = weibo[i] 
    f = 0
    posi, neg = rdmnegative(weibo, s) 
    # print s, posi
    # for ii in neg:
    #     print ii
    tdata.append(text2vec(worddict, s))
    tdata.append(text2vec(worddict, posi))
    for j in neg:
        tdata.append(text2vec(worddict, j))
    return tdata

def genworddict(worddict):
    dictword2vec = {}
    dictdata = [line for line in file(worddict)]
    for line in dictdata:
        da = line.split("\t")
        word = da[0].decode("utf-8")  
        exec("v = " + da[1])
        dictword2vec[word] = v

    return dictword2vec
def getacti(s):
    # get #...#
    m = re.match("#.*#", s)
    if m:
        info = m.group(0)
    else:
        info = "None"
    return info

def rdmnegative(alldoc, s):
    acti = getacti(s)
    positive = ""
    negative = []

    for i in range(len(alldoc)):
        if positive != "" and len(negative) >= NW: break
        cur = random.choice(alldoc)
        c = getacti(cur)
        cur = cur.replace("c", "")
        if c == acti: positive = cur
        elif c != acti and cur not in negative and len(negative) < NW: 
            negative.append(cur)
    return positive, negative

def text2vec(worddict, te):
    tevec=[] 
    aw = re.sub(ur"#.*#", "", te.decode("utf-8"))
    aw = re.findall(ur"[\u4E00-\u9FA5]{1}", aw) 
    for w in aw:
        if worddict.has_key(w):
            tevec.append(worddict[w])
    return np.array(tevec)

def cossim(ls1, ls2):
    #print "*****", ls1, ls2, ls1.shape, ls2.shape
    if ls1.shape != ls2.shape:
        print "error ,list not equal"
        return
    return np.dot(ls1, ls2)/(np.linalg.norm(ls1) * np.linalg.norm(ls2))

# def cossim(ls1, ls2):
#         if len(ls1) != len(ls2):
#             print "error ,list not equal"
#             return
#         m1 = 0
#         m2 = 0
#         sum = 0
#         for i in xrange(len(ls1)):
#             m1 += math.pow(ls1[i], 2)
#             m2 += math.pow(ls2[i], 2)
#             sum +=ls1[i] * ls2[i]
#         return sum/(math.sqrt(m1) *math.sqrt(m2))

if __name__ == '__main__':
    worddict = genworddict("wordhashdict.txt")
    parameters = [np.array(q) for q in [[[random.random() for j in range(M)] for i in range(N)] for k in range(4)] + [[[random.random() for j in range(N)] for i in range(N)] for k in range(7)] + [[random.random() for p in range(N)] for q in range(4)]]
    BPTTtrain(worddict, parameters)
    