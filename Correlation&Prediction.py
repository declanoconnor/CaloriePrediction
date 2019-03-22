#Libraries for Processing
import numpy as np #Linear algebra
import pandas as pd #Data processing
from pandas.tools import plotting #CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt #For Plotting
import seaborn as sns #Additional Plotting
#import keras
#from keras.models import Sequential
#from keras.layers import Dense

#SKLEARN MODULES FOR MACHINE LEARNING
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn import metrics # for the check the error and accuracy of the model

#OTHER TESTED FEATURES
#from sklearn.model_selection import GridSearchCV# for tuning parameter
#from sklearn.metrics import  accuracy_score
#from sklearn.metrics import average_precision_score
#from sklearn.preprocessing import StandardScaler, LabelBinarizer
#from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LinearRegression

#CLEANING UP
import warnings
warnings.filterwarnings('ignore')

#%%

#READ IN OUR DATA
data = pd.read_csv("C:/Users/user/Desktop/mdmenu.csv",header=0) #here header 0 takes away first row

#TEST DATA IS AS SHOWS
#print(data.head(2))
#data.info()
#print(md.isnull().any())

#%% FOR PLOTS ###
features_all=list(data.columns[1:14])
print(features_all)
data.describe()

features_corgraph=list(data.columns[1:16])
print(features_corgraph)

data.groupby('Category')['Item'].count().plot(kind='bar')

sns.boxplot(data= data, x ='Category',y = 'Dietary Fiber')
plt.tight_layout
plt.show()

Max_Cal = data.groupby('Category').max().sort_values('Dietary Fiber',ascending=False)
sns.swarmplot(data =Max_Cal, x= Max_Cal.index,y = 'Calories', hue ='Item',size =10)
plt.tight_layout()

measures = ['Calories', 'Total Fat', 'Cholesterol','Sodium', 'Sugars', 'Carbohydrates']

for m in measures:   
    plot = sns.violinplot(x="Category", y=m, data=data)
    plt.setp(plot.get_xticklabels(), size=7)
    plt.title(m)
    plt.show()

for m in measures:
    g = sns.factorplot(x="Category", y=m,data=data, kind="swarm",size=5, aspect=2.5);
   
#%% DAILY RECOMENDATIONS ###
    
CaloriesPercentage = data['Calories'] / 2500 * 100

TotalFatPercentage = data['Total Fat'] / 66 * 100
SaturatedFatPercentage = data['Saturated Fat'] / 20 * 100
CholesterolPercentage = data['Cholesterol'] / 300 * 100
SodiumPercentage = data['Sodium'] / 2380 * 100
CarbohydratesPercentage = data['Carbohydrates'] / 310 * 100
DietaryFiberPercentage = data['Dietary Fiber'] / 30 * 100
#print(CaloriesPercentage)


#%% #KDE Plots ###

f, axes = plt.subplots(2, 3, figsize=(10, 6.666), sharex=True, sharey=True)

s = np.linspace(0, 3, 10)
cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

x = CaloriesPercentage
y = TotalFatPercentage
sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0,0])
axes[0,0].set(xlim=(-10, 50), ylim=(-30, 70), title = 'Fat')

map = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)


x = CaloriesPercentage
y = SaturatedFatPercentage
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,1])
axes[0,1].set(xlim=(-5, 50), ylim=(-10, 70),  title = 'Saturated Fat')

cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)

x = CaloriesPercentage
y = CholesterolPercentage
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,2])
axes[0,2].set(xlim=(-5, 50), ylim=(-10, 70),  title = 'Cholesterol')

cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)

x = CaloriesPercentage
y = SodiumPercentage
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[1,0])
axes[1,0].set(xlim=(-5, 50), ylim=(-10, 70),  title = 'Sodium')

cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)

x = CaloriesPercentage
y = CarbohydratesPercentage
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[1,1])
axes[1,1].set(xlim=(-5, 50), ylim=(-10, 70),  title = 'Carbohydrates')

cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)

x = CaloriesPercentage
y = DietaryFiberPercentage
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[1,2])
axes[1,2].set(xlim=(-5, 50), ylim=(-10, 70),  title = 'Dietary Fiber')

f.tight_layout()


#%% Correlation Graph ###

corr = data[features_corgraph].corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
          xticklabels= features_all, yticklabels= features_all,
          cmap= 'RdPu')

#coolwarm

#%% INITALISATION FOR MACHINE LEARNING ###

#Adding in features that will 
prediction_var1 = ['Dietary Fiber (% Daily Value)','Vitamin A (% Daily Value)', 'Vitamin C (% Daily Value)', 'Calcium (% Daily Value)', 'Iron (% Daily Value)']
prediction_var = ['Calories','Carbohydrates','Protein','Total Fat','Cholesterol',
                  'Sugars','Saturated Fat', 'Trans Fat',
                  'Dietary Fiber','Sodium']
train, test = train_test_split(data, test_size = 0.2)
(train.shape)
(test.shape)

#print(test.shape)
#print(train.shape)
train_X = train[prediction_var]
train_y= train.Calories
test_X= test[prediction_var]
test_y= test.Calories

#%% Random Forest Classifier ###

model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('')
print('------Random Forest------')
print('Accuracy Classification - ',metrics.accuracy_score(prediction,test_y))
print('Accuracy Regression - ', metrics.explained_variance_score(prediction,test_y))
print('Accuracy Clustering - ', metrics.adjusted_mutual_info_score(prediction,test_y))
print('Macro F1 Score - ',metrics.f1_score(prediction,test_y, average='macro'))

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(test_X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(test_X.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(train_X.shape[1]), indices)
plt.xlim([-1, test_X.shape[1]])
plt.show()

#%% Support Vector Machine ###

model = svm.SVC(kernel='linear')
model.fit(train_X,train_y)
print('')
print('------SVM------')
print('Accuracy Classification - ',metrics.accuracy_score(prediction,test_y))
print('Accuracy Regression - ', metrics.explained_variance_score(prediction,test_y))
print('Accuracy Clustering - ', metrics.adjusted_mutual_info_score(prediction,test_y))
print('Macro F1 Score - ',metrics.f1_score(prediction,test_y, average='macro'))

#%% LogisticRegression ###

model = LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('')
print('------Logistic Regression------')
print('Accuracy Classification - ',metrics.accuracy_score(prediction,test_y))
print('Accuracy Regression - ', metrics.explained_variance_score(prediction,test_y))
print('Accuracy Clustering - ', metrics.adjusted_mutual_info_score(prediction,test_y))
print('Macro F1 Score - ',metrics.f1_score(prediction,test_y, average='macro'))

#%% GaussianNaivebayes ###

model = GaussianNB()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('')
print('------Gaussian NaiveBayes------')
print('Accuracy Classification - ',metrics.accuracy_score(prediction,test_y))
print('Accuracy Regression - ', metrics.explained_variance_score(prediction,test_y))
print('Accuracy Clustering - ', metrics.adjusted_mutual_info_score(prediction,test_y))
print('Macro F1 Score - ',metrics.f1_score(prediction,test_y, average='macro'))

#%% DecisionTree ###

model = DecisionTreeClassifier(max_leaf_nodes=3)
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('')
print('------Decision Tree------')
print('Accuracy Classification - ',metrics.accuracy_score(prediction,test_y))
print('Accuracy Regression - ', metrics.explained_variance_score(prediction,test_y))
print('Accuracy Clustering - ', metrics.adjusted_mutual_info_score(prediction,test_y))
print('Macro F1 Score - ',metrics.f1_score(prediction,test_y, average='macro'))

#%% ExtraTrees ###

model = ExtraTreesClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('')
print('------Extra Tree Classifier------')
print('Accuracy Classification - ',metrics.accuracy_score(prediction,test_y))
print('Accuracy Regression - ', metrics.explained_variance_score(prediction,test_y))
print('Accuracy Clustering - ', metrics.adjusted_mutual_info_score(prediction,test_y))
print('Macro F1 Score - ',metrics.f1_score(prediction,test_y, average='macro'))

#%% KNN ###

model = KNeighborsClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('')
print('------KNN------')
print('Accuracy Classification - ',metrics.accuracy_score(prediction,test_y))
print('Accuracy Regression - ', metrics.explained_variance_score(prediction,test_y))
print('Accuracy Clustering - ', metrics.adjusted_mutual_info_score(prediction,test_y))
print('Macro F1 Score - ',metrics.f1_score(prediction,test_y, average='macro'))

#%% MLP ###
clf = MLPClassifier(#hidden_layer_sizes=(128,64,32), 
					activation='relu', 
                    solver='adam',
     				beta_1=0.6, 
     				beta_2=0.9,
                    alpha = 0.001,
                    early_stopping = True,
                    shuffle = True,
                    warm_start = True,
                    validation_fraction = 0.3,
     				learning_rate_init=0.01, 
     				max_iter = 14000, 
     				random_state = 1235, 
     				learning_rate='adaptive'   
     				)
model = clf
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('')
print('------MLP------')
print('Accuracy Classification - ',metrics.accuracy_score(prediction,test_y))
print('Accuracy Regression - ', metrics.explained_variance_score(prediction,test_y))
print('Accuracy Clustering - ', metrics.adjusted_mutual_info_score(prediction,test_y))
print('Macro F1 Score - ',metrics.f1_score(prediction,test_y, average='macro'))

#%% Scatter Plot Matrix ###

data['Category']=data['Category'].map({'Breakfast':0,'Beef & Pork':1,'Chicken & Fish':2,'Salads':3,'Snacks & Sides':4,'Desserts':5,'Smoothies & Shakes':6})
color_function = {0: "blue", 1: "red", 2: "red", 3: "red", 4: "red", 5: "red", 6: "red"}
colors = data["Category"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column
pd.plotting.scatter_matrix(data[prediction_var], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix

