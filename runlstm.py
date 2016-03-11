import lstm_rnn as lr
import random
import numpy as np
import copy

N = 128
M = 32 # the dim of wordvec
C = 64 # the number of cell
NW = 10 #the number of negitive sample
parameters = np.array([[[random.random() for j in range(M)] for i in range(N)]  for k in range(11)] +　[[random.random() for p in range(N)] for q in range(4)])
miu = 0.99  #momentum parameter
gama = 10 #scaling
era = 1.0 #learning rate

sigmoid = lambda x : 1.0/(1 + np.exp(-x))
tanh = lambda x : np.tanh(x)



def fmtoutput(y_ss, y_pp, y_nn):
    #make len(y_s) = len(y_p) = len(y_n) = maxlen
    maxlen = max(len(y_ss), len(y_pp), len(y_nn))
    while len(y_ss) < maxlen or len(y_pp) < maxlen or len(y_nn) < maxlen:
        if len(y_ss) < maxlen: y_ss.append([0]*N)
        if len(y_pp) < maxlen: y_pp.append([0]*N)
        if len(y_nn) < maxlen: y_nn.append([0]*N)
    return y_ss, y_pp, y_nn, maxlen

def calgradient(param, y_s, y_p, y_n, data):
    #data[0]:source, data[1]:positive, data[2]:negative
    gra = np.array([[[0 for j in range(M)] for i in range(N)]  for k in range(11)] +　[[random.random() for p in range(N)] for q in range(4)])
    lasts_p = np.array([[[0 for j in range(M)] for i in range(N)]  for k in range(8)] +　[[random.random() for p in range(N)] for q in range(3)])
    lasts_q = lasts_p.copy()
    # y_s, y_p, y_n, maxlen = fmtoutput(y_s, y_p, y_n)
    for t in range(1, C):
        if len(set(y_s[-1])) + len(set(y_s[-1])) + len(set(y_s[-1])) == 3 and sum(y_s[-1]) + sum(y_s[-1])+ sum(y_s[-1]) == 0:
            break
        gra_p, lasts_p = calgraR(param, y_s, y_p, lasts_p, data[0:2], t)
        gra_q, lasts_q = calgraR(param, y_s, y_n, lasts_q, data[0:3:2], t)
        gra += gra_p - gra_q

    return gra
def calgraR(param, yq, yd, data, tt):
    gra = [[] for i in range(15)]
    s = yq[-1] 
    t = yd[-1]
    a = np.dot(s, t)
    b = 1.0 / np.linalg.norm(s)
    c = 1.0 / np.linalg.norm(t)
    vq = b * c * t - a * b**3 * c * s
    vd = b * c * s - a * b * c**3 * t

    # ---------------------------for output gate------------------------
    sigmarqt1 = yq[4][tt] * (1 - yq[4][tt]) * tanh(yq[3][tt]) * vq
    sigmardt1 = yd[4][tt] * (1 - yd[4][tt]) * tanh(yd[3][tt]) * vd    
    gra_wr1 = np.dot(sigmarqt1, s[tt-1]) + np.dot(sigmardt1, t[tt-1])
    gra_w1 =  np.dot(sigmarqt1, data[0][tt]) + np.dot(sigmardt1, data[1][tt])
    gra_wp1 =  np.dot(sigmarqt1, yq[3][tt]) + np.dot(sigmardt1, yq[3][tt])
    gra_b1 = sigmarqt1 + sigmardt1
    gra[4].append(gra_wr1)
    gra[0].append(gra_w1)
    gra[8].append(gra_wp1)
    gra[11].append(gra_b1)

    #  ---------------------------for input gate------------------------
    sigmart3 = lambda y,v : (1 - tanh(y[3][tt])) * (1 + tanh(y[3][tt])) * y[4][tt] * v
    bit = lambda ygt, it : ygt[tt] * it[tt] * (1-it[tt])
    syvq_ig = sigmart3(yq, vq)
    syvd_ig = sigmart3(yd, vd)
    grarall = lambda g_q, g_d : np.dot(syvq_ig, g_q) + np.dot(syvd_ig, g_d)

    gracwr3 = lambda ft, gracwr3_last, ygt, it, yt: np.dot(ft[tt], gracwr3_last[tt]) + np.dot(bit(ygt, it), yt[tt-1]) 
    gracwr3_q = gracwr3(yq[2], gr_last_q, yq[0], yq[1], yq[5])
    gracwr3_d = gracwr3(yd[2], gr_last_d, yd[0], yd[1], yd[5])
    gr_last_q = gracwr3_q.copy()
    gr_last_d = gracwr3_d.copy()
    gra_wr3 = grarall(gracwr3_q, gracwr3_d)

    gracw3 = lambda ft, gracw3_last, ygt, it, i: np.dot(ft[tt], gracw3_last[tt]) + np.dot(bit(ygt, it), data[i][tt-1]) 
    gracw3_q = gracw3(yq[2], grw3_last_q, yq[0], yq[1], 0)
    gracw3_d = gracw3(yd[2], grw3_last_d, yd[0], yd[1], 1)
    grw3_last_q = gracw3_q.copy()
    grw3_last_d = gracw3_d.copy()
    gra_w3 = grarall(gracw3_q, gracw3_d)

    gracwp3 = lambda ft, gracwp3_last, ygt, it, ct: np.dot(ft[tt], gracwp3_last[tt]) + np.dot(bit(ygt, it), ct[tt-1]) 
    gracwp3_q = gracwp3(yq[2], grwp3_last_q, yq[0], yq[1], yq[3])
    gracwp3_d = gracwp3(yd[2], grwp3_last_d, yd[0], yd[1], yd[3])
    grwp3_last_q = gracwp3_q.copy()
    grwp3_last_d = gracwp3_d.copy()
    gra_wp3 = grarall(gracwp3_q, gracwp3_d)

    gracb3 = lambda ft, gracb3_last, ygt, it: np.dot(ft[tt], gracb3_last[tt]) + bit(ygt, it)
    gracb3_q = gracb3(yq[2], grb3_last_q, yq[0], yq[1], yq[3])
    gracb3_d = gracb3(yd[2], grb3_last_d, yd[0], yd[1], yd[3])
    grb3_last_q = gracb3_q.copy()
    grb3_last_d = gracb3_d.copy()
    gra_b3 = grarall(gracb3_q, gracb3_d)

    gra[6].append(gra_wr3)
    gra[2].append(gra_w3)
    gra[10].append(gra_wp3)
    gra[13].append(gra_b3)
    #  ---------------------------for forget gate------------------------

    sigmart2 = lambda ct, ot, v : (1 - tanh(ct[tt])) * (1 + tanh(ct[tt])) * ot[tt] * v
    bft = lambda ct, ft : ct[tt-1] * ft[tt] * (1-ft[tt])
    syvq_fg = sigmart2(yq[3], yq[4], vq)
    syvd_fg = sigmart2(yd[3], yd[4], vd)
    grarall_fg = lambda g_q, g_d : np.dot(syvq_fg, g_q) + np.dot(syvd_fg, g_d)


    gracwr2 = lambda ft, gracwr2_last, ct, ft, yt: np.dot(ft[tt], gracwr2_last[tt]) + np.dot(bft(ct, ft), yt[tt-1]) 
    gracwr2_q = gracwr2(yq[2], grwr2_last_q, yq[3], yq[2], yq[5])
    gracwr2_d = gracwr2(yd[2], grwr2_last_d, yd[3], yd[2], yd[5])
    grwr2_last_q = gracwr2_q.copy()
    grwr2_last_d = gracwr2_d.copy()
    gra_wr2 = grarall_fg(gracwr2_q, gracwr2_d) 

    
    gracw2 = lambda ft, gracw2_last, ct, ft, i: np.dot(ft[tt], gracw2_last[tt]) + np.dot(bft(ct, ft), data[i][tt-1]) 
    gracw2_q = gracw2(yq[2], grw2_last_q, yq[3], yq[2], yq[5], 0)
    gracw2_d = gracw2(yd[2], grw2_last_d, yd[3], yd[2], yd[5], 1)
    grw2_last_q = gracw2_q.copy()
    grw2_last_d = gracw2_d.copy()
    gra_w2 = grarall_fg(gracw2_q, gracw2_d) 

    gracwp2 = lambda ft, gracwp2_last, ct, ft: np.dot(ft[tt], gracwp2_last[tt]) + np.dot(bft(ct, ft), ct[tt-1]) 
    gracwp2_q = gracwp2(yq[2], grwp2_last_q, yq[3], yq[2])
    gracwp2_d = gracwp2(yd[2], grwp2_last_d, yd[3], yd[2])
    grwp2_last_q = gracwp2_q.copy()
    grwp2_last_d = gracwp2_d.copy()
    gra_wp2 = grarall_fg(gracwp2_q, gracwp2_d) 
 
    gracb2 = lambda ft, gracb2_last, ct, ft: np.dot(ft[tt], gracb2_last[tt]) + bft(ct, ft) 
    gracb2_q = gracb2(yq[2], grb2_last_q, yq[3], yq[2])
    gracb2_d = gracb2(yd[2], grb2_last_d, yd[3], yd[2])
    grb2_last_q = gracb2_q.copy()
    grb2_last_d = gracb2_d.copy()
    gra_b2 = grarall_fg(gracb2_q, gracb2_d)

    gra[5].append(gra_wr2)
    gra[1].append(gra_w2)
    gra[9].append(gra_wp2)
    gra[12].append(gra_b2)
    #  ---------------------------for input without gate------------------------
    sigmart4 = lambda ct, ot, v : (1 - tanh(ct[tt])) * (1 + tanh(ct[tt])) * ot[tt] * v
    bgt = lambda it, ygt : it[tt] * (1 - ygt[tt]) * (1 + ygt[tt])
    syvq_iwg = sigmart4(yq[3], yq[4], vq)
    syvd_iwg = sigmart4(yd[3], yd[4], vd)
    grarall_iwg = lambda g_q, g_d : np.dot(syvq_iwg, g_q) + np.dot(syvd_iwg, g_d)

    gracwr4 = lambda ft, gracwr4_last, it, ygt, yt: np.dot(ft[tt], gracwr4_last[tt]) + np.dot(bgt(it, ygt), yt[tt-1]) 
    gracwr4_q = gracwr4(yq[2], grwr4_last_q, yq[1], yq[0], yq[5])
    gracwr4_d = gracwr4(yd[2], grwr4_last_d, yd[1], yd[0], yd[5])
    grwr4_last_q = gracwr4_q.copy()
    grwr4_last_d = gracwr4_d.copy()
    gra_wr4 = grarall_iwg(gracwr4_q, gracwr4_d) 

    gracw4 = lambda ft, gracw4_last, it, ygt, i: np.dot(ft[tt], gracw4_last[tt]) + np.dot(bgt(it, ygt), data[i][tt-1]) 
    gracw4_q = gracw4(yq[2], grw4_last_q, yq[1], yq[0], yq[5], 0)
    gracw4_d = gracw4(yd[2], grw4_last_d, yd[1], yd[0], yd[5], 1)
    grw4_last_q = gracw4_q.copy()
    grw4_last_d = gracw4_d.copy()
    gra_w4 = grarall_iwg(gracw4_q, gracw4_d)

    gracb4 = lambda ft, gracb4_last, it, ygt: np.dot(ft[tt], gracb4_last[tt]) + bgt(it, ygt) 
    gracb4_q = gracb4(yq[2], grb4_last_q, yq[1], yq[0])
    gracb4_d = gracb4(yd[2], grb4_last_d, yd[1], yd[0])
    grb4_last_q = gracb4_q.copy()
    grb4_last_d = gracb4_d.copy()
    gra_b4 = grarall_iwg(gracb4_q, gracb4_d)

    gra[7].append(gra_wr4)
    gra[3].append(gra_w4)
    gra[14].append(gra_b4)
    
    lasts = [gracw2_last, gracw3_last, gracw4_last, gracwr2_last, gracwr3_last, gracwr4_last, gracb2_last, gracb3_last, gracb4_last]
    return  gra, lasts

def getlastoutput(param, textvec, t, f = "last"):
    lr_s = lr.LSTM_RNN(param, textvec)
    y_output = lr_s.lstmrun(ti = t, flag = f)
    return y_output

def BPTTtrain():
    weibo = [line for line in file("weibo_train.txt")]
    param_last = np.array([[0]*len(pa) for pa in param_init])

    while 1:
        gradient = np.array([[[0 for j in range(M)] for i in range(N)]  for k in range(11)] +　[[random.random() for p in range(N)] for q in range(4)])
        for r in range(len(weibo)):
            if getacti(weibo[r]) == "None" : continue
            data = gettraindata(r, weibo) #data[0]: source, data[1]:posotive, data[2:]:negatives
            param = parameters + miu * (parameters - param_last)
            y_s = getlastoutput(param, data[0], len(data[0]), f = "all")
            y_p = getlastoutput(param, data[1], len(data[1]), f = "all")
            cos_y_sp = cossim(y_s[-1][-1], y_p[-1][-1])
            esum = 0
            for j in range(NW):
                y_n = getlastoutput(param, data[2+j], len(data[2+j]), f = "all")
                cos_y_sn = cossim(y_s[-1][-1], y_n[-1][-1])
                esum += np.exp(-1 * gama * (cos_y_sp - cos_y_sn))
            for j in range(NW):
                alpharj = (-1 * gama * np.exp(-1 * gama *)) / (1 + esum)
                gradient += alpharj * calgradient(param, y_s, y_p, y_n, [data[0], data[1], data[2+j]])
            
        delta_k = miu * (parameters- param_last) - era * gradient
        parameters = parameters + delta_k
        param_last = copy.deepcopy(parameters) #parameters.copy()
    

    pa = ["w1", "w2", "w3", "w4", "wr1", "wr2", "wr3", "wr4", "wp1", "wp2", "wp3", "b1", "b2", "b3", "b4"]
    for i in range(15):
        if i < 11:
            np.array([[k for k in j]for j in parameters[i]]).tofile("./param/" + pa[i] + ".txt")
        else:
            np.array([j for j in parameters[i]]).tofile("./param/" + pa[i] + ".txt")


def gettraindata(i, weibo):
    worddict = genworddict("wordhashdict.txt")
    tdata = []
    s = weibo[i]
    posi, neg = rdmnegative(alldoc, s)
    tdata.append(text2vec(worddict, s))
    tdata.append(text2vec(worddict, posi))
    for i in neg:
        tdata.append(text2vec(worddict, t))
    return tdata

def genworddict(worddict):
    dictword2vec = {}
    dictdata = [line for line in file(worddict)]
    for line in dictdata:
        da = line.split("\t")
        exec(str(dictword2vec[da[0].decode("utf-8")]) + "=" + da[1])
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
        if positive != "" and len(negative) == CW : break
        cur = rd.choice(alldoc)
        c = getacti(cur)
        cur = cur.replace("c", "")
        if c == acti and positive == "":
            positive = cur
        elif len(negative) < CW:
            negative.append(cur)
    return positive, negative

def text2vec(worddict, te):
    tevec=[]
    allwords =worddict.keys()
    aw = re.findall(u"[\u4E00-\u9FA5]{1}", te)
    for w in aw:
        if w not in allwords: continue
        tevec.append(worddict[w])
    return tevec

def cossim(ls1, ls2):
    if ls1.shape != ls2.shape:
        print "error ,list not equal"
        return
    return np.dot(ls1, ls2)/(np.linalg.norm(ls1) * np.linalg.norm(ls2))

if __name__ == '__main__':
    BPTTtrain()
    