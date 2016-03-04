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