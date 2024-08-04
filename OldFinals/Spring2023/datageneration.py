import hashlib as hs
import numpy as np
import random 
import csv
import scipy
import pandas as pd
from datetime import date, timedelta
import sys

def generateData(name,path):

    print("Writing Random Values for {0} into {1}".format(name,path))
    nameHash = hs.sha1(name.encode('utf-8')).hexdigest()
    print("Hashed Name: ",nameHash)
    seed = int(nameHash[0:8],16)
    random.seed(seed)
    np.random.seed(seed)

    ##problem 1
    a =random.randint(0, 99)
    b = random.randint(0, a)
    a = a/100
    b = b/100

    corr = [[1, a, b], [a, 1, a], [b,a,1]]

    sd = abs(np.random.normal(size=3))
    cov = np.diag(sd) @ corr @ np.diag(sd)

    rnd = np.random.multivariate_normal([.0,.0,.0],cov,size=20)
    start_prices = np.random.normal(100,10,size=3)
    prices = []
    for i in [0,1,2]:
        prices.append([start_prices[i]*(1+rnd.T[i,j]/100) for j in range(0,20)])

    prices = pd.DataFrame(np.matrix(prices).T, columns = ["Price1", "Price2", "Price3"])

    m11 = random.randint(0, 19)
    m12 = random.randint(0, 19)
    prices.Price1[max(m11,m12)] = None
    prices.Price1[min(m11,m12)] = None

    m2 = random.randint(0, 19)
    prices.Price2[max(m11,m2)] = None
    prices.Price2[min(m11,m2)] = None

    m3 = random.randint(0, 19)
    prices.Price3[max(m2,m3)] = None
    prices.Price3[min(m2,m3)] = None

    file_path = path+'/problem1.csv'

    dts = [d for d in (date.today()-timedelta(days=n) for n in range(0,20))]
    dts.reverse()
    prices["Date"] = dts

    prices.to_csv(file_path,index=False)

    print('Problem1 written successfully.') 

    ##problem 2
    kv = {}
    kv['Underlying'] = np.random.normal(100,10,1)
    kv['Strike'] = kv['Underlying'] + np.random.normal(0,10,1)
    kv['IV'] = [round(np.random.normal(0.2,0.02,1)[0],2)]
    kv['TTM'] = [int(np.random.normal(150,10,1))]
    kv['RF'] = [0.045]
    kv['DivRate'] = [abs(np.random.normal(0.05,0.005,1)[0]) ]
    kvout = pd.DataFrame(kv)
    file_path = path+'/problem2.csv'
    kvout.to_csv(file_path,index=False)
        
    print('Problem2 written successfully.') 
    ###problem 3
    a = random.randint(0, 99)/100
    corr = [[1, a, a], [a, 1, a], [a,a,1]]

    sd = abs(np.random.normal(.2,.02,size=3))

    cov = np.diag(sd) @ corr @ np.diag(sd)
    cov = pd.DataFrame(cov,columns= ["Asset1", "Asset2", "Asset3"])

    file_path = path+'/problem3_cov.csv'
    cov.to_csv(file_path,index=False)

    sr = abs(np.random.normal(0.5,0.1,1)[0])

    kv = {} 
    kv['RF']=[0.045]
    RF = 0.045
    kv['Expected_Value_1'] =  [sr * sd[0] + RF]
    kv['Expected_Value_2'] = [sr * sd[1] + RF]
    kv['Expected_Value_3'] = [sr * sd[2] + RF]

    file_path = path+'/problem3_ER.csv'

    pd.DataFrame(kv).to_csv(file_path,index=False)

    print('Problem3 written successfully.') 

    ###problem 4
    w = abs(np.random.normal(0.5,0.2,3))
    w = w/sum(w)
    file_path = path+'/problem4_startWeight.csv'
    pd.DataFrame(np.matrix(w),columns= ["weight1", "weight2", "weight3"]).to_csv(file_path,index=False)

    r = np.random.normal(0,0.2/np.sqrt(12),(20,3))
    r = pd.DataFrame(r,columns=['Asset1','Asset2','Asset3'])
    dts = [d for d in (date.today()-timedelta(days=n) for n in range(0,20))]
    dts.reverse()
    r["Date"] = dts

    file_path = path+'/problem4_returns.csv'
    r.to_csv(file_path,index=False)

    print('Problem4 written successfully.') 

    ###problem 5 
    df = max(5,min(12,np.random.normal(8.0,3.0,1)[0]))
    a = random.randint(0, 99) / 100
    corr = [[1, a, a, a], [a, 1, a, a], [a, a, 1, a], [a, a, a, 1]]
    s = abs(np.random.normal(.02,.002,size=4))
    cov = np.diag(s) @ corr @ np.diag(s)
    rnd = scipy.stats.multivariate_t([0,0,0,0],cov,df=df).rvs(size=61)
    start_prices = np.random.normal(100,10,size=4)
    prices = []
    for i in range(4):
        prices.append([start_prices[i]*(1+rnd.T[i,j]/100) for j in range(0,61)])

    prices = pd.DataFrame(np.matrix(prices).T, columns = ["Price1", "Price2", "Price3","Price4"])
    dts = [d for d in (date.today()-timedelta(days=n) for n in range(0,61))]
    dts.reverse()
    prices["Date"] = dts

    file_path = path+'/problem5.csv'
    prices.to_csv(file_path,index=False)

    print('Problem5 written successfully.') 

generateData(sys.argv[1],sys.argv[2])