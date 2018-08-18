#!/usr/bin/env python
#-*- coding:utf-8 -*-
import csv
import os
import pandas as pd
import numpy as np
from sklearn import svm
import time
import re
import copy

def Guesskey(key,row,k):
    if key == 'Embarked':
        # print row
        if row[2] == 1 or row[9] == 80:
            return 'C'
    else:
        return k

def readData(fileName):
    result = {}
    ss = 0
    pc = 0
    fm = 0
    with open(fileName,'rb') as f:
        rows = csv.reader(f)
        for row in rows:
            if result.has_key('attr_list'):
                ss = 0
                pc = 0
                fm = 0
                for i in range(len(result['attr_list'])-1):

                    key = result['attr_list'][i]
                    if not result.has_key(key):
                        result[key] = []
                    if key == "SibSp":
                        ss = row[i]
                    if key == "Parch":
                        pc = row[i]
                    if key == "Name":
                        SName = re.split(', |. ',row[i])
                        if 'Master' in row[i]:
                            row[i] = "Master"
                        if 'Col' in row[i]:
                            row[i] = "Col"
                        if 'Rev' in row[i]:
                            row[i] = "Rev"
                        if 'Mme' in row[i] or 'Mlle' in row[i]:
                            row[i] = 'Mlle'
                        if 'Capt' in row[i] or 'Don' in row[i] or 'Major' in row[i] or 'Sir' in row[i] or 'Mr' in row[i]:
                            row[i] = 'Mr'
                        if 'Dona' in row[i] or 'Lady' in row[i] or 'the Countess' in row[i] or 'Jonkheer' in row[i] or 'Miss' in row[i]:
                            row[i] = 'Miss'
                        else:
                            if len(SName) == 3:
                                row[i] = SName[1]
                            if len(SName) == 2:
                                row[i] = SName[0]
                        # result[key].append(row[i])
                        # continue
                    if key == 'Cabin' and row[i] != "":
                        # print 'Cabin',row[i], row[i][0]
                        row[i] = row[i][0]
                    row[i] = Guesskey(key,row,row[i])
                    result[key].append(row[i])
                    # print row[i]
                    # print key, result[key]

                key = "FamilySize"
                if not result.has_key(key):
                    result[key] = []
                result[key].append(int(ss) + int(pc))
                # print int(ss) + int(pc)
            else:
                result['attr_list'] = row + ["FamilySize"]

    return result

def writeData(fileName, data):
    csvFile = open(fileName, 'w')
    writer = csv.writer(csvFile)
    n = len(data)
    for i in range(n):
        writer.writerow(data[i])
    csvFile.close()

def convertData(dataList,key):
    hashTable = {}
    count = 0.0

    for i in range(len(dataList)):
        if key == "Ticket":
            if hashTable.has_key(dataList[i]):
                    temp = copy.copy(dataList[i])
                    for j in range(len(dataList)):
                        if dataList[j] == temp:
                            dataList[i] = 2 
            else:
                hashTable[dataList[i]] = count
                dataList[i] = 1
                count += 1
            continue
            
        if not hashTable.has_key(dataList[i]):
            hashTable[dataList[i]] = count
            count += 1
        dataList[i] = hashTable[dataList[i]]

def convertValueData2(dataList):
    sumValue = 0.0
    count = 0
    for i in range(len(dataList)):
        if dataList[i] == "":
            dataList[i] = 0
            continue
        sumValue += float(dataList[i])
        count += 1
        dataList[i] = float(dataList[i])
    avg = sumValue / count
    med = np.median(dataList)
    # print avg, med

    for i in range(len(dataList)):
        if dataList[i] == 0:
            dataList[i] = float(med)

def convertValueData(dataList):
    sumValue = 0.0
    count = 0
    for i in range(len(dataList)):
        if dataList[i] == "":
            continue
        sumValue += float(dataList[i])
        count += 1
        dataList[i] = float(dataList[i])
    avg = sumValue / count
    for i in range(len(dataList)):
        if dataList[i] == "":
            dataList[i] = avg

def dataPredeal(data):
    # print data
    convertValueData(data["Age"])
    convertData(data["Fare"],"Fare")
    convertData(data["Name"],"Name")
    convertData(data["Pclass"],"Pclass")
    convertData(data["Sex"],"Sex")
    # convertData(data["SibSp"],"SibSp")
    # convertData(data["Parch"],"Parch")
    convertData(data["Cabin"],"Cabin")
    convertData(data["Embarked"],"Embarked")
    convertData(data["Ticket"],"Ticket")
    convertData(data["FamilySize"],"FamilySize")
  
def getX(data): 
    x = []
    ignores = {"PassengerId":1, "Survived":1, "SibSp":1, "Parch":1}      
    for i in range(len(data["PassengerId"])):
        x.append([])
        for j in range(len(data["attr_list"])):
            key = data["attr_list"][j]
            if not ignores.has_key(key):
                # print len(data["attr_list"]), j, key, data[key][i]
                x[i].append(data[key][i])
        # print ""
    return x

def getLabel(data):
    label = []
    for i in range(len(data["PassengerId"])):
        label.append(int(data["Survived"][i]))
    return label

def calResult(x,label, input_x):
    svmcal = svm.SVC(kernel='linear', C=1).fit(x, label)
    # svmcal = svm.SVC(kernel='poly', coef0=0.01).fit(x, label)
    # svmcal = svm.SVC(kernel='rbf', coef0=100).fit(x, label)
    return svmcal.predict(input_x)

def run():
    dataRoot = 'input/'
    data = readData(dataRoot + 'train.csv')
    test_data = readData(dataRoot + 'test.csv')
    dataPredeal(data)
    dataPredeal(test_data)
    x = getX(data)
    label = getLabel(data)
    input_x = getX(test_data)
    t1 = time.time()
    x_result = calResult(x, label,input_x)
    print "time {:.3f} sec".format(time.time() - t1)
    res = [[test_data["PassengerId"][i], x_result[i]] for i in range(len(x_result))]
    res.insert(0, ["PassengerId", "Survived"])
    writeData('result.csv', res)


    crossdata = readData(dataRoot + 'train.csv')
    dataPredeal(crossdata)
    cx = getX(crossdata)
    clabel = getLabel(crossdata)
    print "time {:.3f} sec".format(time.time() - t1)
    print "len: ", len(cx)
    sample = 400
    ctest = cx[sample:]
    tlabel = clabel[sample:]
    crossresult = calResult(cx[:sample], clabel[:sample], ctest)
    
    err = 0.0
    for i in range(891 - sample):
        if crossresult[i] == tlabel[i]:
            err += 1
    print sample
    print "error rate: {:.3f}".format(err / (891 - sample))


def main():
    filename = "input/train.csv"

    result = readData(filename)
    # get titanic & test csv files as a DataFrame
    train = pd.read_csv("input/train.csv")
    test  = pd.read_csv("input/test.csv")

    full = train.append( test , ignore_index = True )
    titanic = full[ :891 ]

    del train , test

    print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)
    titanic.head()
    titanic.describe()

if __name__ == '__main__':
    run()