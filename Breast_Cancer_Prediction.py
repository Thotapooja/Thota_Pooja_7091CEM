import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import time

#DATA COLLECTION
#load the data
breast_cancer_data = load_breast_cancer()

breast_cancer_dataFrame = pd.DataFrame(breast_cancer_data.data,columns=breast_cancer_data.feature_names)

breast_cancer_dataFrame['label'] = breast_cancer_data.target
breast_cancer_dataFrame.shape
breast_cancer_dataFrame.info()
breast_cancer_dataFrame.describe()

#DATA Pre-processing
#Checking the missing values
print(breast_cancer_dataFrame.isnull().sum())
breast_cancer_dataFrame['label'].value_counts()


#Displaying the count plot for the dataset
sb.countplot(breast_cancer_dataFrame['label'])
breast_cancer_data1 = sb.countplot(x='label',data = breast_cancer_dataFrame)
for p in breast_cancer_data1.patches:
    breast_cancer_data1.annotate('{:1.1f}'.format(p.get_height()),(p.get_x()+0.25,p.get_height()+0.01))
#plt.show()

breast_cancer = pd.read_csv("data.csv")
breast_cancer.shape
breast_cancer = breast_cancer.dropna(axis=1)
breast_cancer.shape
print(breast_cancer['diagnosis'].value_counts())




#Displaying the pair plot for the dataset
labelEncode_Y = LabelEncoder()
breast_cancer.iloc[:,1] = labelEncode_Y.fit_transform(breast_cancer.iloc[:,1].values)
sb.pairplot(breast_cancer.iloc[:,1:5])
#plt.show()

sb.pairplot(breast_cancer.iloc[:,1:5], hue="diagnosis")
#plt.show()

#Displaying the heatmap for the dataset
plt.figure(figsize=(100,100),num=" Heat Map for Breast Cancer")
sb.heatmap(breast_cancer.corr(),cmap="Blues", annot=True)
#plt.show()


plt.figure(figsize=(10,10),num=" Heat Map for Breast Cancer")
sb.heatmap(breast_cancer.iloc[:,1:10].corr(),annot=True, fmt=".0%")
#plt.show()



plot = plt.figure(figsize=(5,5),num=  "radius_mean Vs texture_mean")
sb.barplot(x='radius_mean', y='perimeter_mean', data = breast_cancer)
#plt.show()


plot = plt.figure(figsize=(5,5),num=  "radius_mean Vs texture_mean")
sb.boxplot(x='radius_mean', y='perimeter_mean', data = breast_cancer)
#plt.show()

#Removing unwanted columns
x_Value = breast_cancer.drop('diagnosis',axis=1)
y_Value = breast_cancer['diagnosis'].apply(lambda y_value: 1 if y_value >= 7 else 0)


X= breast_cancer.iloc[:,2:31].values
Y= breast_cancer.iloc[:,1].values
X.shape
Y.shape

#Training and testing the data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)


model = RandomForestClassifier()
model.fit(X_train, Y_train)


#Model Evaluation
X_test_prediction = model.predict(X_test)


#Accuracy Score
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(test_data_accuracy)


#Data Standardisation using standard scaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

def models(X_train,Y_train):
    logReg = LogisticRegression(random_state = 0)
    logReg.fit(X_train, Y_train)

    tree = DecisionTreeClassifier(random_state=0,criterion="entropy")
    tree.fit(X_train,Y_train)

    forest = RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
    forest.fit(X_train,Y_train)
    print(logReg.score(X_train,Y_train))
    print(tree.score(X_train,Y_train))
    return logReg,tree,forest


#Displaying the classification report and confusion matrix for the data
model = models(X_train,Y_train)
print(model)
print("Length os",len(model))
for i in range(len(model)):
    print("______________________________________________________________")
    print(model[i])
    print("______________________________________________________________")
    print(classification_report(Y_test,model[i].predict(X_test)))
    plt.figure(figsize=(12,8))
    pred = model[i].predict(X_test)
    cm = confusion_matrix(Y_test,pred)
    ax=plt.subplot()
    sb.heatmap(cm,annot=True,fmt='g',ax=ax)
    plt.show()

    print("Accuracy Score is : ",accuracy_score(Y_test,model[i].predict(X_test)))
    
num_folds = 10
results = []
names = []
models_list = []
models_list.append(('Decision Tree', DecisionTreeClassifier()))
models_list.append(('Logistic', LogisticRegression())) 
models_list.append(('Random Forest', RandomForestClassifier()))


for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123,shuffle=True )
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Download the decision tree for predicting the breast cancer
clf = tree.DecisionTreeClassifier(max_depth = 2, random_state =0)
clf.fit(X_train,Y_train)
clf.predict(X_test)
tree.plot_tree(clf)
fig,axes = plt.subplots(nrows = 1,ncols =1,figsize=(4,4),dpi=300)
tree.plot_tree(clf,class_names = breast_cancer_data.target_names, feature_names =breast_cancer_data.feature_names,filled= True)
fig.savefig('Breast Cancer Prediction Decision Tree.png')


#Testing the input data and predict whether it is a malignant or benign tumour
model1 = RandomForestClassifier()
model1.fit(X_train, Y_train)
X_test_prediction = model1.predict(X_test)

input_data =[10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
input_data_array = np.asarray(input_data)
input_reshaped_data = input_data_array.reshape(1,-1)
prediction = model1.predict(input_reshaped_data)
if(prediction[0] == 0):
    print("______________________________________________________________")
    print('the Breast Cancer is Malignant')
    print("______________________________________________________________")
else:
    print("______________________________________________________________")
    print('the Breast Cancer is Benign')
    print("______________________________________________________________")
