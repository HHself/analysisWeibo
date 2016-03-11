import numpy as np
import scipy as sp
import math 


N = 128 
C = 64 # the number of cell
sigmoid = lambda x : 1.0/(1 + np.exp(-x))
tanh = lambda x : np.tanh(x)
cos = lambda x,y : np.cos(x, y)

class LSTM_RNN():
	 """Long short-term memory.

    The input consists of 4 parts:
    - input gate
    - forget gate
    - cell input
    - output gate

    """
	def __init__(self, param, textvec, text="", k=3):
        if len(textvec) > C:
	    	print "text length >= max len"
	    	return
	    self.W1 = param[0] # param: W1, W2, W3, W4, Wr1, Wr2, Wr3, Wr4, Wp1, Wp2, Wp3, b1, b2, b3, b4
	    self.W2 = param[1]
	    self.W3 = param[2]
	    self.W4 = param[3]
	    self.Wr1 = param[4]
	    self.Wr2 = param[5]
	    self.Wr3 = param[6]
	    self.Wr4 = param[7]
	    self.Wp1 = param[8]
	    self.Wp2 = param[9]
	    self.Wp3 = param[10] # dim: n*32
	    self.b1 = param[11]
	    self.b2 = param[12]
	    self.b3 = param[13]
	    self.b4 = param[14] # dim: n*1
	    self.textvec = textvec
	    self.text = text # it's unicode without non-chinese
	    self.textlen = len(textvec)
	    self.k = k # the number of keyword
    
    def lstmrun(self):
    	'''
    	    flag means output all vectors or last vector
    	'''
    	if textlen != self.textvec.shape[0]:
    		print "data error, length not equal!"
    		return
    	y_before = 0
    	c_before = 0
    	output = [[[0]*N for j in range(C)] for i in range(6)]


    	for num in range(self.textlen):
    		curvec = self.textvec[num]
    		ygt = tanh(np.dot(self.W4, curvec) + np.dot(self.Wr4, y_before) + self.b4)
    		it  = sigmoid(np.dot(self.W3, curvec) + np.dot(self.Wr3, y_before) + np.dot(self.Wp3, c_before) + self.b3)
    		ft  = sigmoid(np.dot(self.W2, curvec) + np.dot(self.Wr2, y_before) + np.dot(self.Wp2, c_before) + self.b2) 
    		ct  = ft * c_before + it * ygt
    		ot  = sigmoid(np.dot(self.W1, curvec) + np.dot(self.Wr1, y_before) +np.dot(self.Wp1, ct) + self.b1)
    		yt  = ot * tanh(ct)
    		
    		y_before = yt
    		c_before = ct
    		output[0][num] = ygt
    		output[1][num] = it
    	    output[2][num] = ft
    	    output[3][num] = ct
    	    output[4][num] = ot
    	    output[5][num] = yt
    	# output = np.array(output)
    	return output


    def getkeywordind(self):
    	y_output = self.lstmrun(flag = "all")
    	dis = [cossim(y_output[num], y_output[num+1]) for num in range(textlen-1)]
    	dis_ind_val = dict([(ind, val) for ind, val in enumerate(dis, start = 1)])
    	dis_ind = sorted(dis_ind_val)

    	f = 0
    	while f < textlen:
    		for j in range(f, textlen):

    def cossim(self, ls1, ls2):
        if ls1.shape != ls2.shape:
            print "error ,list not equal"
            return
        m1 = 0
        m2 = 0
        sum = 0
        for i in xrange(len(ls1)):
            m1 += math.pow(ls1[i], 2)
            m2 += math.pow(ls2[i], 2)
            sum +=ls1[i] * ls2[i]
        return sum/(math.sqrt(m1) *math.sqrt(m2))




    
    
    
