import pandas as pd
import scipy.stats
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import pickle
from nltk.metrics import scores

# global dataa_train, data_output_train, dataa_test, data_output_test


global x_train, y_train, x_test, y_test, data, X, Y


def prepro(df1):
   # df1 = df1.dropna(axis=1)
    df1=df1.fillna(0)
    # print(df1.describe())
    # print(df1.shape)
    df1 = df1.drop_duplicates(subset=None, keep='first', inplace=False)
    df1 = df1.replace({'diagnosis': {'B': 0, 'M': 1}})

    datau = df1['diagnosis']
    df1 = df1.drop(columns=['diagnosis'])
    df1['diagnosis_a'] = datau

    # print(dataa.info)
    return df1


def outlier(dataa, ft):
    q1 = dataa[ft].quantile(0.25)
    q3 = dataa[ft].quantile(0.75)
    IQR = q3 - q1
    lower = q1 - 1.5 * IQR
    upper = q3 + 1.5 * IQR
    ma = dataa.index[(dataa[ft] < lower) | (dataa[ft] > upper)]
    dataa = dataa.drop(ma)
    return dataa


# data = prepro()
data = pd.read_csv("Tumor Cancer Prediction_Data.csv")
# print(df1)
l = (
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17',
    'F18',
    'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30')
for features in l:
    data = outlier(data, features)

    # print(data.describe)
    # print(data.shape)

data = prepro(data)

X = data.iloc[:, 1:31]
Y = data['diagnosis_a']


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)

#print(x_test)
# print (x_train)
# print(y_train)


class classification:
    global x_train, y_train, x_test, y_test, data, X, Y
    global clf, logreg, tree, accuracy
    global model1, model2, model3

    # voting system

    def SVM(self, x, y, x_test, y_test):
        # SVM classification
        clf = svm.SVC(kernel='linear', C=0.0001)  # Linear Kernel
        clf.fit(x, y)
        y_train_pred = clf.predict(x)
        print(" svm train data Accuracy:", accuracy_score(y, y_train_pred))
        accuracy = clf.score(x_test, y_test)
        print(' SVM test data accuracy: ', accuracy)
        print(classification_report(y_test, clf.predict(x_test)))
        model1 = clf


        with open('model svm', 'wb') as filename:
            pickle.dump(model1, filename)
        return  model1

    def logisticRegression(self, x, y, x_test, y_test):
        # logisticRegression classification
        logreg = LogisticRegression(solver='lbfgs', max_iter=1000, C=.001)

        logreg.fit(x, y)
        y_train_pred = logreg.predict(x)
        print(" logistic regression train data Accuracy: ", accuracy_score(y, y_train_pred))
        accuracy = logreg.score(x_test, y_test)
        print(' logistic regression test data accuracy: ', accuracy)
        print(classification_report(y_test, logreg.predict(x_test)))
        model2 = logreg
        with open('model logistic regression', 'wb') as filename:
            pickle.dump(model2, filename)
        return model2

        # decisionTree classification

    def decisionTree(self, x, y, x_test, y_test):
        tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
        tree.fit(x, y)
        y_train_pred = tree.predict(x)
        print("decision tree train data Accuracy:", accuracy_score(y, y_train_pred))
        accuracy = tree.score(x_test, y_test)
        print(' decision tree test data accuracy: ', accuracy)
        print(classification_report(y_test, tree.predict(x_test)))
        model3 = tree
        with open('model decision tree', 'wb') as filename:
            pickle.dump(model3, filename)
        return model3

            #votingmodule
    def VotingSaveModule(self, x_train, y_train,model1,model2,model3):
            voting_cls_hard = VotingClassifier(estimators=[('logistic', model2), ('decisiontree', model3), ('SVM', model1)],
                                               voting="hard")
            voting_cls_hard.fit(x_train, y_train)
            with open("VOTINGMODULE", "wb") as file:
                pickle.dump(voting_cls_hard, file)





c = classification()

#
m1=c.SVM(x_train, y_train, x_test, y_test)
m3=c.logisticRegression(x_train, y_train, x_test, y_test)
m2=c.decisionTree(x_train, y_train, x_test, y_test)
c.VotingSaveModule(x_train, y_train,m1,m2,m3)
# filename="svm with model pickle"
# joblib.dump(model1, filename)

class test:

    data2 = pd.read_csv("Tumor Cancer Test data.csv")



    def prediction(self, x):
        with open('model svm', 'rb') as filename:
            svm = pickle.load(filename)
        with open('model svm', 'rb') as filename:
            logreg = pickle.load(filename)
        with open('model svm', 'rb') as filename:
            tree = pickle.load(filename)

        m1 = svm.predict(x)
        m2 = logreg.predict(x)
        m3 = tree.predict(x)
        print('svm model prediction',m1)
        print('logistic model prediction',m2)
        print('decision tree model prediction',m3)

    def VotingModule(self, x,y):
            vote = pickle.load(open("VOTINGMODULE", "rb"))
            predect = vote.predict(x)
            list1 = []
            for x in predect:
                if (x == 0):
                    list1.append("B")
                else:
                    list1.append("M")
            print("Y Prediction")
            print(list1)
            print("Prediction accuracy")
            score = accuracy_score(list(y), predect)
            print(score)


t = test()
t.data2 = prepro(t.data2)
#print(t.data2)



l = (
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17',
    'F18',
    'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30')
for features in l:
    t.data2 = outlier(t.data2, features)

#print(t.data2)
X1 = t.data2.iloc[:, 1:31]
Y1 = t.data2['diagnosis_a']

#t.prediction(X1)
#

