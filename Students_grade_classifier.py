import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#load the data set (adjust the path if you did not put it in the same dirctory)
dataset=pd.read_csv("xAPI-Edu-Data.csv");



#Data Preprocessing

# check if there is missing values
print(dataset.isnull().sum());  # there is no missing values

# convert strings to numrical catogries in each feature
dataset['gender'].replace({'M': 0, 'F': 1},inplace = True)
dataset['NationalITy'].replace({'KW': 0, 'lebanon': 1,'Egypt':2,'SaudiArabia':3,'USA':4,'Jordan':5,'venzuela':6,'Iran':7,'Tunis':8,'Morocco':9,'Syria':10,'Iraq':11,'Palestine':12,'Lybia':13},inplace = True)
dataset['PlaceofBirth'].replace({'KuwaIT': 0, 'lebanon': 1,'Egypt':2,'SaudiArabia':3,'USA':4,'Jordan':5,'venzuela':6,'Iran':7,'Tunis':8,'Morocco':9,'Syria':10,'Iraq':11,'Palestine':12,'Lybia':13},inplace = True)
dataset['StageID'].replace({'lowerlevel': 0, 'MiddleSchool':1 ,'HighSchool':2},inplace = True)
dataset['GradeID'].replace({'G-04': 0, 'G-07': 1,'G-08':2,'G-06':3,'G-05':4,'G-09':5,'G-12':6,'G-11':7,'G-10':8,'G-02':9},inplace = True)
dataset['SectionID'].replace({'A': 0, 'B': 1,'C':2},inplace = True)
dataset['Topic'].replace({'IT': 0, 'Math': 1,'Arabic':2,'Science':3,'English':4,'Quran':5,'Spanish': 6,'French':7,'History':8,'Biology':9,'Chemistry':10,'Geology':11},inplace = True)
dataset['Semester'].replace({'F': 0, 'S': 1,'C':2},inplace = True)
dataset['Relation'].replace({'Father': 0, 'Mum': 1},inplace = True)
dataset['ParentAnsweringSurvey'].replace({'Yes': 0, 'No': 1},inplace = True)
dataset['ParentschoolSatisfaction'].replace({'Good': 0, 'Bad': 1},inplace = True)
dataset['StudentAbsenceDays'].replace({'Under-7': 0, 'Above-7': 1},inplace = True)
dataset['Class'].replace({'L': 0,'M': 1,'H':2},inplace = True)


#store the features and Classes from the dataset

# put the features in X 
X = dataset.iloc[:, 0:16].values
# take the classes coulmn in y
y = dataset.iloc[:, 16].values

#split the dataset to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=93)


#train the model and predect classes of test dataset
model3 = RandomForestClassifier(n_estimators=900,max_depth=9,n_jobs=-1,random_state=27)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)


#the evaluation results of the model

print(classification_report(y_test, y_pred,target_names=['L', 'M', 'H']))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm ,display_labels=['L','M','H'])
disp.plot()  

