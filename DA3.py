#In[]
#Thêm thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn import metrics

from sklearn.tree import export_graphviz
from graphviz import Source

from sklearn.model_selection import GridSearchCV

train_test_split
lb = LabelEncoder()

class DataProcessing:
    #Đọc file
    def ReadFiles(self,file_names):
        return pd.read_csv(file_names)
       
    #Gọp file
    def FilesAggregation(self,fileA,fileB):
        return pd.concat([fileA,fileB],ignore_index=True)

    #Tách dữ liệu làm hai tập train set và test set theo Stratified
    def TrainTestSet(self, data, data_column, bins, labels):
        column_new = "Khoảng "+data_column
        data[column_new]=pd.cut(data[data_column],bins=bins,labels=labels)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
        for train_index, test_index in splitter.split(data, data[column_new]):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]
        
        #Vẽ
        data[column_new].hist(bins=6,figsize=(5,5))
        plt.title(column_new)
        plt.show()

        train_set[column_new].hist(bins=6,figsize=(5,5))
        plt.title("Train set")
        plt.show()

        test_set[column_new].hist(bins=6,figsize=(5,5))
        plt.title("Test set")
        plt.show()

        #Xóa feature mới
        for _set_ in (train_set, test_set):
            _set_.drop(columns="Khoảng G3", inplace=True)
        
        return train_set,test_set

    #Tách label
    def SplitLabel(self,data, column):
        data_label = data[column].copy()
        data = data.drop(columns=column,inplace=True)
        return data_label

    #Biến đổi label thành hai class 0: Fail, 1: Pass
    def LabelTransform(self, data_label):
        data_label.loc[data_label[:,]<10,]=0
        data_label.loc[data_label[:,]>=10,]=1
        return data_label

    #Biến đổi chữ sang số
    def FitTransform(self,cat_array, data):
        for i in cat_array:
            data.iloc[:,i] = lb.fit_transform(data.iloc[:,i])
        return data

class DecisionTreeExcute:
    #Tìm max depths
    def FindMaxDepths(self, max_depth,train_set_val, train_set_label):
        params = {'max_depth': list(range(1, max_depth))}
        grid_search_cv = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy", random_state=42), params, verbose=1, cv=3)
        grid_search_cv.fit(train_set_val, train_set_label)
        max_dept_params = grid_search_cv.best_params_
        return max_dept_params['max_depth'], grid_search_cv.best_estimator_

    #Tính cross value score
    def CalculatorAUC(self,model,train_set,train_set_label, test_set, test_set_label):
        dtc = model.fit(train_set,train_set_label)
        sc = cross_val_score(dtc, test_set, test_set_label, scoring='roc_auc',cv=5)
        result = "AUC: %0.2f (+/- %0.2f)" % (sc.mean(), sc.std() * 2)
        return result

    #Report
    def ReportDecisionTree(self,model,test_label, test_set):
        pd = model.predict(test_set)
        file = open("Report.txt","w")
        report = metrics.classification_report(test_label, pd)
        file.write(report)
        return metrics.classification_report(test_label, pd)

    #Export file PDF
    def ExportFigs(self, data, feature_names, class_names,fig_name):
        export_graphviz( data,
            out_file=r"Figs/"+fig_name+".dot",
            feature_names= feature_names,
            class_names= class_names,
            rounded=True, filled=True, leaves_parallel=True, 
            node_ids=True, proportion=False, precision=2 )
        Source.from_file(r"Figs/"+fig_name+".dot").render(r"Figs/"+fig_name, view=True, cleanup=True)

#In[]
#1. Xử lý dữ liệu
#1.1 #Đọc file 
dp = DataProcessing()
df_mat = dp.ReadFiles('student-mat.csv')
df_por = dp.ReadFiles('student-por.csv')

#1.2 Gọp file
df = dp.FilesAggregation(df_mat,df_por)

#1.3 Tách ra làm hai tập train set và test set
bins=[-1,5,11,15,20, np.inf]
labels=[5,11,15,20,25]
train_set, test_set = dp.TrainTestSet(df, "G3",bins,labels)
print("====================Train set============================")
print(train_set)
print("====================Test set============================")
print(test_set)

#1.4 Tách label G3
train_set_label = dp.SplitLabel(train_set,"G3")
test_set_label = dp.SplitLabel(test_set,"G3")

#1.5 Label thành hai class 1 và class 
dp.LabelTransform(train_set_label)
dp.LabelTransform(test_set_label)
print("====================Train set label============================")
print(train_set_label)
print("====================Test set label============================")
print(test_set_label)

#1.6 Đổi những feature chữ thành số 
cat_feat_index = [0,1,2,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]
processed_train_set_val = dp.FitTransform(cat_feat_index, train_set)
processed_test_set_val = dp.FitTransform(cat_feat_index, test_set)

#2 Thực thi Decision Tree
dte = DecisionTreeExcute()

#2.1 Tìm max depth
max_depths = np.linspace(1, 57, 57, endpoint=True)
model = tree.DecisionTreeClassifier
max_dep_index, max_dep_value = dte.FindMaxDepths(100,processed_train_set_val, train_set_label)
print("========================Max depth infor===============================")
print(max_dep_value)
print("========================Max depth infor===============================")
print(max_dep_index)

#2.2 Tính giá trị trung bình cross value score
model = tree.DecisionTreeClassifier(criterion="entropy",max_depth = max_dep_index,random_state=42)
auc = dte.CalculatorAUC(model, processed_train_set_val, train_set_label, processed_test_set_val, test_set_label)
print("========================AVG cross value score===============================")
print(auc)

#2.3 Viết Report trên file Report.txt
report = dte.ReportDecisionTree(model,test_set_label,processed_test_set_val)
print("========================Write file report in Report.txt===============================")

#2.4 Export ra file PDF Decision Tree với độ sâu của cây đã tìm ở 2.1
feature_names=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic","famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2"]
class_names=["Fail","Pass"]
dte.ExportFigs(model,feature_names,class_names,"tree")
print("========================Save file PDF tree with max depth in Figs file===============================")

#2.5 Export ra file PDF Decision Tree với toàn bộ
modelAll = tree.DecisionTreeClassifier(criterion="entropy",random_state=42)
modelAll.fit(processed_train_set_val,train_set_label)
dte.ExportFigs(modelAll,feature_names,class_names,"tree_all")
print("========================Save file PDF tree without max depth in Figs file===============================")



#In[]
#2.6 Tạo 1 bộ dữ liệu test và phần tích kết quả
# Dữ liệu thô chưa chuyển qua số [GB,F,17,U,GT3,A,0,0,health, service, home,father, 1,1,0,1,1,1,0,1,0,1,1,2,4,5,3,1,4,16,7,15]
# Dữ liệu sau khi xử lý [0,0,17,1,0,0,0,0, 1, 3, 1 ,0,1,1, 0,1,1,1,0,1,0,1,1,2,4,5,3,1,4,16,7,15]
n = [0,0,17,1,0,0,0,0, 1, 3, 1 ,0,1,1, 0,1,1,1,0,1,0,1,1,2,4,5,3,1,4,16,7,10]
predict = model.predict([n])
print(predict)

