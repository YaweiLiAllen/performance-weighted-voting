#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
from torch.autograd import Variable

## read input_dataframe
import pickle
patient_file = open('/home/ylz0045/TCGA_pancancer/data/patient_matrix.pkl','rb')
input_dataframe = pickle.load(patient_file)
patient_file.close()

set_cancer = set(input_dataframe['cancer_type'])
word_to_idx_cancer = {word: i for i, word in enumerate(set_cancer)}
idx_to_word_cancer = {i: word for i, word in enumerate(set_cancer)}
cancer_id = [word_to_idx_cancer[w] for w in input_dataframe['cancer_type']]
input_dataframe.insert(0, 'cancer_type_id', cancer_id)

## select train_data, valid_data and test_data
from sklearn.model_selection import train_test_split
train_data, valid_test_data = train_test_split(input_dataframe, test_size = 0.4, random_state = None)
valid_data, test_data = train_test_split(valid_test_data, test_size = 0.5, random_state = None)
train_data = train_data.sample(frac = 1)
valid_data = valid_data.sample(frac = 1)
test_data = test_data.sample(frac = 1)



## parameter optimization
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

## LogisticRegression
clf1 = LogisticRegression().set_params(multi_class='ovr')
param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'class_weight': [None, 'balanced'],
              'penalty': ['l2']
             },
              {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
              'solver': ['liblinear', 'saga'],
              'class_weight': [None, 'balanced'],
              'penalty': ['l1']
             }]
grid_search1 = GridSearchCV(clf1, param_grid, n_jobs = 20, verbose = 0).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])
best_parameters1 = grid_search1.best_estimator_.get_params()

## SVM
clf2 = SVC().set_params(probability=True)
param_grid = [{'C': [0.01, 0.1, 1, 10, 100], 
              'kernel': ['poly', 'rbf','sigmoid'],
              'class_weight': [None, 'balanced'],
              'gamma': [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001]
             },
              {'C': [0.01, 0.1, 1, 10, 100], 
              'kernel': ['linear'],
              'class_weight': [None, 'balanced'],
             }]
grid_search2 = GridSearchCV(clf2, param_grid, n_jobs = 20, verbose = 0).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])
best_parameters2 = grid_search2.best_estimator_.get_params()

## randomforest
clf3 = RandomForestClassifier()
param_grid = {'n_estimators': list(range(10, 201, 10)),
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth' : range(10,50,1),
              'min_samples_leaf': range(1,5,1),
              'min_samples_split': range(1,15,1),
              'criterion' :['gini', 'entropy'],
              'class_weight': [None, 'balanced']
             }
grid_search3 = GridSearchCV(clf3, param_grid, n_jobs = 20, verbose = 0).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])  
best_parameters3 = grid_search3.best_estimator_.get_params()

## MLP
clf4 = MLPClassifier(shuffle=True)
param_grid = {'hidden_layer_sizes': [(256,256),(256,512),(256,1024),(512,256),(512, 512),
                                     (512, 1024),(1024,256),(1024,512),(1024,1024)],
              'learning_rate': ['adaptive', 'invscaling', 'constant'],
              'learning_rate_init': [0.0001, 0.001, 0.01, 0.1, 1],
              'momentum': [0.6, 0.7, 0.8, 0.9],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'sgd', 'adam'],
             }
grid_search4 = GridSearchCV(clf4, param_grid, n_jobs = 20, verbose = 0).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])
best_parameters4 = grid_search4.best_estimator_.get_params()

## XGBoost
clf5 = XGBClassifier()
param_grid = {'n_estimators': range(100, 301, 10),
              'max_depth': range(1, 15, 1),
              'learning_rate': [0.01, 0.1, 1],
              'min_child_weight': np.linspace(1,3,3),
              'gamma': np.linspace(0.1, 1, 8),
              'subsample': np.linspace(0.1, 1, 8),
             }
grid_search5 = GridSearchCV(clf5, param_grid, n_jobs = 20, verbose = 0) fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])   
best_parameters5 = grid_search5.best_estimator_.get_params()


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

## Logistic
clf1 = LogisticRegression(C=best_parameters1['C'], 
                         solver=best_parameters1['solver'], 
                         penalty=best_parameters1['penalty'],
                         class_weight= best_parameters1['class_weight']
                         ).set_params(n_jobs=10).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])

## SVM
clf2 = SVC(C=best_parameters2['C'], 
          kernel=best_parameters2['kernel'], 
          gamma=best_parameters2['gamma'],
          class_weight = best_parameters2['class_weight']
          ).set_params(probability=True).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])

## random_forest
clf3 = RandomForestClassifier(n_estimators=best_parameters3['n_estimators'], 
                             max_features=best_parameters3['max_features'], 
                             max_depth=best_parameters3['max_depth'],
                             min_samples_leaf=best_parameters3['min_samples_leaf'],
                             min_samples_split=best_parameters3['min_samples_split'],
                             criterion = best_parameters3['criterion'],
                             class_weight = best_parameters3['class_weight']
                             ).set_params(n_jobs=10).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])

## MLP
clf4 = MLPClassifier(hidden_layer_sizes=best_parameters4['hidden_layer_sizes'], 
                    activation=best_parameters4['activation'], 
                    solver=best_parameters4['solver'],
                    momentum=best_parameters4['momentum'],
                    learning_rate=best_parameters4['learning_rate'],
                    learning_rate_init=best_parameters4['learning_rate_init'],
                    ).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])

## XGBoost
clf5 = XGBClassifier(n_estimators=best_parameters5['n_estimators'], 
                    max_depth=best_parameters5['max_depth'], 
                    learning_rate=best_parameters5['learning_rate'],
                    min_child_weight=best_parameters5['min_child_weight'],
                    gamma=best_parameters5['gamma'],
                    subsample=best_parameters5['subsample'],
                    ).set_params(n_jobs=10).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])

## Voting
clf6 = VotingClassifier(estimators=[('lr',clf1),('svm',clf2),('rf',clf3),('mlp',clf4),('xgb',clf5)], 
                        voting='soft').set_params(n_jobs=10).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])
clf7 = VotingClassifier(estimators=[('lr',clf1),('svm',clf2),('rf',clf3),('mlp',clf4),('xgb',clf5)], 
                        voting='hard').set_params(n_jobs=10).fit(train_data[train_data.columns[2:]], train_data['cancer_type_id'])

clfs = [clf1, clf2, clf3, clf3, clf5]

for i in [1, 2, 3, 4, 5, 6]:
    locals()["predict{}_valid_proba".format(i)] = locals()["clf{}".format(i)].predict_proba(valid_data[train_data.columns[2:]])
    locals()["classifier{}_valid".format(i)] = pd.DataFrame(locals()["predict{}_valid_proba".format(i)], index = valid_data.index)
    locals()["classifier{}_valid".format(i)].insert(0 , "true_type", valid_data['cancer_type_id'])
    
    locals()["predict{}_test".format(i)] = locals()["clf{}".format(i)].predict(test_data[train_data.columns[2:]])
    locals()["accuracy{}_test".format(i)] = (locals()["predict{}_test".format(i)] == test_data['cancer_type_id']).mean()
    locals()["predict_proba{}".format(i)] = locals()["clf{}".format(i)].predict_proba(test_data[train_data.columns[2:]])
    locals()["classifier{}_test".format(i)] = pd.DataFrame(locals()["predict_proba{}".format(i)], index = test_data.index)
    locals()["classifier{}_test".format(i)].insert(0 , "true_type", test_data['cancer_type_id'])
    
predict7_test = clf7.predict(test_data[train_data.columns[2:]])
accuracy7_test = (predict7_test == test_data['cancer_type_id']).mean()


## the weight of performance weighted voting
from sklearn.preprocessing import LabelBinarizer 

encoder = LabelBinarizer(sparse_output = False)

valid_one_hot = encoder.fit_transform(valid_data['cancer_type_id'])
valid_one_hot = pd.DataFrame(valid_one_hot, index = valid_data.index)

test_one_hot = encoder.fit_transform(test_data['cancer_type_id'])
test_one_hot = pd.DataFrame(test_one_hot, index = test_data.index)


from scipy import linalg

para_weight = np.zeros((len(word_to_idx_cancer), len(clfs)), dtype = float)
for i in range(len(word_to_idx_cancer)):
    para_x = np.zeros((valid_data.shape[0], len(clfs)), dtype = float)
    para_y = np.array(valid_one_hot[i])
    for j in range(len(clfs)):
        para_x[:, j] = locals()["classifier{}_valid".format(j+1)][i]
    para_X = para_x.T.dot(para_x)
    para_Y = para_x.T.dot(para_y)
    weight = linalg.solve(para_X, para_Y)
    weight = weight / weight.sum()
    para_weight[i,:] = weight
    
para_weight = pd.DataFrame(para_weight, columns = ["LR", "SVM", "RF", "MLP", "XGBoost"])
para_weight.insert(0, "cancer_type", [idx_to_word_cancer[i] for i in range(14)])
cancer_id = list(map(word_to_idx_cancer.get, para_weight['cancer_type']))
para_weight.index = cancer_id
para_cancer = para_weight['cancer_type'].values
patients = [test_data[test_data['cancer_type'] == para_weight.loc[i, 'cancer_type']].shape[0] for i in range(14)]
para_weight.insert(1, 'patients', patients)
para_weight = para_weight.sort_index()


## performance weighted voting
key=para_weight
results = []
predict_prob = np.zeros((predict_proba1.shape[0], len(word_to_idx_cancer)), dtype=float)
max_prob1 = np.max(np.array(key.iloc[:,2:]), axis=1)
max_prob2 = np.sort(np.array(key.iloc[:,2:]),axis=1)[:,3]
max_prob3 = np.sort(np.array(key.iloc[:,2:]),axis=1)[:,2]
max_prob4 = np.sort(np.array(key.iloc[:,2:]),axis=1)[:,1]
max_prob5 = np.min(np.array(key.iloc[:,2:]),axis=1)
max_pos1 = np.argmax(np.array(key.iloc[:,2:]), axis=1)
max_pos2 = np.argsort(np.array(key.iloc[:,2:]),axis=1)[:,3]
max_pos3 = np.argsort(np.array(key.iloc[:,2:]),axis=1)[:,2]
max_pos4 = np.argsort(np.array(key.iloc[:,2:]),axis=1)[:,1]
max_pos5 = np.argmin(np.array(key.iloc[:,2:]),axis=1)
for i in range(predict_proba1.shape[0]):
    matrix = np.vstack((predict_proba1[i],predict_proba2[i],predict_proba3[i],predict_proba4[i],predict_proba5[i]))
    predict_list = []
    for j in range(len(word_to_idx_cancer)):
        normalize_para = max_prob1[j] +max_prob2[j] +max_prob3[j] +max_prob4[j] +max_prob5[j]
        predict_list.append((matrix[max_pos1[j],[j]]*max_prob1[j]/normalize_para
                            +matrix[max_pos2[j],[j]]*max_prob2[j]/normalize_para
                            +matrix[max_pos3[j],[j]]*max_prob3[j]/normalize_para
                            +matrix[max_pos4[j],[j]]*max_prob4[j]/normalize_para
                            +matrix[max_pos5[j],[j]]*max_prob5[j]/normalize_para
                           ).item())
    predict_list = [predict_list[i]/sum(predict_list) for i in range(len(predict_list))]
    results.append(np.argmax(predict_list))
    predict_prob[i] = predict_list
print('Accuracy:', (results == test_data['cancer_type_id']).mean())




