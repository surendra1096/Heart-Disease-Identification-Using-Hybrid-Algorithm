import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_curve
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

data = pd.read_csv('heart_data.csv', sep='\t')

data.plot(kind='density', subplots=True, layout=(4, 4), sharex=False,
          fontsize=8, figsize=(10, 10))
plt.tight_layout()
plt.title('Density map', y=1.1)

plt.show()

mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True
with sn.axes_style("ticks"):
    f, ax = plt.subplots(figsize=(9, 5))
    ax = sn.heatmap(data.corr(), mask=mask, vmax=.3, annot=True, fmt=".0%", linewidth=0.5, square=False)

plt.title('Correlation Map', y=1.1)

plt.show()

a = pd.get_dummies(data['cp'], prefix="cp")
b = pd.get_dummies(data['thal'], prefix="thal")
c = pd.get_dummies(data['slope'], prefix="slope")
d = pd.get_dummies(data['gender'], prefix="sex")

updated_clms = [data, a, b, c, d]
data = pd.concat(updated_clms, axis=1)
data.head()
data = data.drop(columns=['cp', 'thal', 'slope', 'gender'])

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
y = data.target
X = data.drop(['target'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
acc = dt_model.score(X_test, y_test) * 100
prediction = dt_model.predict(X_test)
print("********************** DECISION TREE ************************")
print("DT Accuracy Score: {:.2f}%".format(acc))
print(f"Precision Score  : {precision_score(prediction, y_test) * 100:.2f}% ")
print(f"Recall Score     : {recall_score(prediction, y_test) * 100:.2f}% ")
print("Confusion Matrix :\n", confusion_matrix(prediction, y_test))

# plot_confusion_matrix of DT
array = [[25, 8],
         [7, 27]]
df_cf_matrix = pd.DataFrame(array, index=[i for i in "01"],
                            columns=[i for i in "01"])
plt.figure(figsize=(2, 2))
sn.heatmap(df_cf_matrix, annot=True)
plt.title('Confusion matrix of  DECISION TREE ', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1, random_state=0)
rf.fit(X_test, y_test)
acc = rf.score(X_test, y_test) * 100
y_pred = rf.predict(X_test)
cf_matrix = confusion_matrix(y_test.T, y_pred)
print("********************** RANDOM FOREST ************************")
print("Accuracy of Random Forest: {:.2f}%".format(acc))
print(f"Precision Score  : {precision_score(y_pred, y_test) * 100:.2f}% ")
print(f"Recall Score     : {recall_score(y_pred, y_test) * 100:.2f}% ")
print("Confusion Matrix :\n", cf_matrix)
# plot_confusion_matrix of RF
array = [[26, 6],
         [2, 33]]
df_cf_matrix = pd.DataFrame(array, index=[i for i in "01"],
                            columns=[i for i in "01"])
plt.figure(figsize=(2, 2))
sn.heatmap(df_cf_matrix, annot=True)
plt.title('Confusion matrix of random forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_prediction = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_prediction)
print("********************** SUPPORT VECTOR MACHINE ************************")
print(f"Accuracy of SVM : {accuracy_score(y_test, y_prediction) * 100:.2f}% ")
print(f"Precision Score  : {precision_score(y_prediction, y_test) * 100:.2f}% ")
print(f"Recall Score     : {recall_score(y_prediction, y_test) * 100:.2f}% ")
print("Confusion Matrix : \n", cm)

# plot_confusion_matrix of  SVM
array = [[30, 2],
         [4, 31]]
df_confusion_matrix = pd.DataFrame(array, index=[i for i in "01"],
                                   columns=[i for i in "01"])
plt.figure(figsize=(2, 2))
sn.heatmap(df_confusion_matrix, annot=True)
plt.title('Confusion matrix of  SVM', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
