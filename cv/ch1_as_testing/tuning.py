import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from sklearn import metrics, svm
from test_function import *
from sklearn.metrics import accuracy_score

data = pd.read_csv("/storage/home/jkl5991/work/project/unannotated_omit_std.tsv", sep = "\t")
#omit = pd.read_csv("/storage/home/jkl5991/work/project/not_conflict/recessive_std.tsv", sep = "\t")
x_column = ['SIFT_pred','LRT_pred', 'MA_pred', 'PROVEN_pred', 'SLR_score', 'SIFT_score','LRT_omega', 
                'MA_score', 'PROVEN_score', 'Grantham', 'HMMEntropy','HMMRelEntropy', 'PredRSAB', 'PredRSAI', 
                'PredRSAE','PredBFactorF', 'PredBFactorM', 'PredBFactorS', 'PredStabilityH','PredStabilityM', 
                'PredStabilityL', 'PredSSE', 'PredSSH','PredSSC', 'dscore', 'phyloP_pri',
                'phyloP_mam','phyloP_ver','RNA_seq','UNEECON'] 
y_column = ['clinvar_result']


ch1 = data[data['location'].str.contains('chr1-')]
other = data.loc[~data['location'].isin(ch1['location'])]
X_test = ch1.loc[:,x_column]
X_train = other.loc[:,x_column]
y_test = ch1.loc[:,y_column].values.flatten()
y_train = other.loc[:,y_column].values.flatten()


with open("result_omit/%s%s%s.txt"%(sys.argv[1],sys.argv[2],sys.argv[3]),"w") as f:
     count_ = int(sys.argv[4])
     f.write("# Tuning hyper-parameters = kernal:%s, C:%0.03f, gamma:%0.03f\n"%(sys.argv[1],float(sys.argv[2]),float(sys.argv[3])))
     svc = SVC(kernel = sys.argv[1], C = float(sys.argv[2]), gamma = float(sys.argv[3]))
     svc.fit(X_train, y_train)
     train_pred = svc.predict(X_train)
     train_acc = accuracy_score(y_train,train_pred)
     test_pred = svc.predict(X_test)
     test_acc = accuracy_score(y_test, test_pred)
     #f.write("training accuracy = %r\n"%train_acc)
     f.write("testing accuracy = %0.3f\n"%test_acc )
     print('wrote %r of total 24 lines'%(count_))
     if count_ == 24:
        print('finished!')


