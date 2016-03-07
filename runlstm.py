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

def gettraindata():

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
    gra = 0
    # y_s, y_p, y_n, maxlen = fmtoutput(y_s, y_p, y_n)
    for t in range(1, C):
        if len(set(y_s[-1])) + len(set(y_s[-1])) + len(set(y_s[-1])) == 3 and sum(y_s[-1]) + sum(y_s[-1])+ sum(y_s[-1]) == 0:
            break
        gra += calgraR(param, y_s, y_p, data[0:2], t) - calgraR(param, y_s, y_n, data[0:3:2], t)
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

    # for output gate
    sigmarqt1 = yq[4][tt] * (1 - yq[4][tt]) * tanh(yq[3][tt]) * vq
    sigmardt1 = yd[4][tt] * (1 - yd[4][tt]) * tanh(yd[3][tt]) * vd    
    gra_wr1 = np.dot(sigmarqt1, s[tt-1]) + np.dot(sigmardt1, t[tt-1])
    gra_w1 =  np.dot(sigmarqt1, data[0][tt]) + np.dot(sigmardt1, data[1][tt])
    gra_wp1 =  np.dot(sigmarqt1, yq[3][tt]) + np.dot(sigmardt1, yq[3][tt])
    gra_b1 = sigmarqt1 + sigmardt1
    

    #for input gate
    sigmarqt3 = (1 - tanh(yq[3][tt])) * (1 + tanh(yq[3][tt])) * yq[4][tt] * vq
    sigmardt3 = (1 - tanh(yd[3][tt])) * (1 + tanh(yd[3][tt])) * yd[4][tt] * vd

    gra_wr3 
    gra_w3 = 
    gra_wp3 = 
    gra_b3 =

    #for forget gate

    gra_wr2 = 
    gra_w2 = 
    gra_wp2 = 
    gra_b2 = 

    #for input without gate

    gra_wr4 = 
    gra_w4 =
    gra_b4 = 



    return 
def calgraY():




def getlastoutput(param, textvec, t, f = "last"):
    lr_s = lr.LSTM_RNN(param, textvec)
    y_output = lr_s.lstmrun(ti = t, flag = f)
    return y_output

def BPTTtrain(): 
    param_last = np.array([[0]*len(pa) for pa in param_init])

    while 1:
        gradient = 0
        for r in range(num_weibo):
            data = gettraindata(i) #data[0]: source, data[1]:posotive, data[2:]:negatives
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

    


def genworddict(self, worddict):
        dictword2vec = {}
        dictdata = readfiles(worddict)
        for line in dictdata:
            da = line.split("\t")
            dictword2vec[da[0].decode("utf-8")] = da[1]
        return dictword2vec
    
def rdmnegative(self, alldoc, te, po, num):
    rdmdata = []
    sp = [te, po]
    cur = te
    for i in range(num):
        while cur in sp:
            cur = rd.choice(alldoc)
        sp.append(cur)
        rdmdata.append(cur)
    return rdmdata

def text2vec(self, worddict, te):
    tevec=[]
    allwords =worddict.keys()
    aw = re.findall(u"[\u4E00-\u9FA5]{1}", te)
    for w in aw:
        if w not in allwords: continue
        tevec.append(worddict[w])
    return tevec

def cossim(self, ls1, ls2):
    if ls1.shape != ls2.shape:
        print "error ,list not equal"
        return
    return np.dot(ls1, ls2)/(np.linalg.norm(ls1) * np.linalg.norm(ls2))