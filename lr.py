from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def lrmodel(trainX, trainY, predictX, trueY):
    model = LogisticRegression()
    model.fit(trainX, trainY)
    predicted = model.predict(trainX)

    fli = metrics.f1_score(trueY, predicted, average='micro')
    fla = metrics.f1_score(trueY, predicted, average='macro')

    return f1i, fla

def getnodetopic(voca, path):
    vecdata = [[float(num) for num in line.split()] for line in file(path)]
    vec = [[0 for j in range(len(vecdata))] for i in range(len(vecdata[0]))]
    vec2 = [[] for i in range(len(vecdata[0]))]

    for i in range(len(vecdata)):
        for j in range(len(vecdata[0])):
            vec[j][i] = vecdata[i][j]

    for k in range(len(vec)):
        vec2[voca[k]] = vec[k]
    return vec2

def getvoca(path):
    data = {}
    for line in readfiles(path):
        da = line.replace("\n", "").split("\t")
        data[int(da[0])] = int(da[1])
    return data

def readfiles(path):
    data=[]
    for line in file(path):
        data.append(line)
    return data


#return data like {node : label}
def getpercomdic(path):
    data = {}
    for line in readfiles(path):
        # print line
        te = line.split("\t")
        data[int(te[0])] = int(te[1].replace("\n",""))
    return data

def runlr():
    voca = getvoca("voca.txt")
    node_topic = getnodetopic(voca, "pw_z.txt")
    node_label = getpercomdic("10312community.dat")
    label = node_label.values()

    print lrmodel(node_topic, label, node_topic, label)





