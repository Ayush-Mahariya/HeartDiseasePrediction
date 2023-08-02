## HeartDiseasePrediction
It is an ML project for detecting Heart disease

# Explanation of Dataset:
age = age in years
sex = (1 = male; 0 = female)
cp = chest pain type
trestbps = resting blood pressure (in mm Hg on admission to the hospital)
chol = serum cholestoral in mg/dl
fbs = (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
restecg = resting electrocardiographic results (values 0,1,2)
thalach = maximum heart rate achieved
exang = exercise induced angina (1 = yes; 0 = no)
oldpeak = ST depression induced by exercise relative to rest
slope = the slope of the peak exercise ST segment
ca = number of major vessels (0-3) colored by flourosopy
thal = thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

# Results:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('dataset.csv')
print(df.head())
<img width="749" alt="1" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/4e834b1b-be16-45e1-b3a4-f4d2e386ff33">

print(df.info())
<img width="749" alt="2" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/ea7ed541-5704-4df4-b19a-ee59ed6c9284">

print(df.describe())
<img width="669" alt="3" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/d9593383-f992-4dc2-916c-49f715ce867c">

import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(16,16))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
<img width="361" alt="4" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/5634048b-897f-42fe-ab5e-bef0fbfaf434">

sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')
plt.show()
<img width="275" alt="5" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/601157c0-e5bf-46ee-8393-0a4935056884">

dataset = pd.get_dummies(df, columns = ['sex', 'cp', 
                                        'fbs','restecg', 
                                        'exang', 'slope', 
                                        'ca', 'thal'])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
dataset.head()
<img width="606" alt="6" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/e48aa71b-0594-4e35-8bb4-0dba5abca930">

y = dataset['target']
X = dataset.drop(['target'], axis = 1)

from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()
<img width="316" alt="7" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/2ed28ae1-1686-4463-90ab-69aae56f6bc4">

knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)
score.mean()
<img width="98" alt="8" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/29776074-9589-4d53-9dea-ad16444ebe00">

from sklearn.ensemble import RandomForestClassifier
randomforest_classifier= RandomForestClassifier(n_estimators=10)
score=cross_val_score(randomforest_classifier,X,y,cv=10)
score.mean()
<img width="104" alt="9" src="https://github.com/Ayush-Mahariya/HeartDiseasePrediction/assets/83781124/94ac75d9-47cb-43cd-801b-0e03b2f8cfe6">
