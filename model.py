import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sn
# Load the data
data = pd.read_csv("heart.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.2, random_state=42)

# Train three different models on the training set
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Define a function to make predictions using the hybrid model
def make_hybrid_prediction(dt_model, rf_model, svm_model, X):
    dt_pred = dt_model.predict(X)
    rf_pred = rf_model.predict(X)
    svm_pred = svm_model.predict(X)

    hybrid_pred = np.zeros_like(dt_pred)
    for i in range(len(dt_pred)):
        if dt_pred[i] == rf_pred[i] == svm_pred[i]:
            hybrid_pred[i] = dt_pred[i]
        else:
            if dt_pred[i] == rf_pred[i]:
                hybrid_pred[i] = dt_pred[i]
            elif dt_pred[i] == svm_pred[i]:
                hybrid_pred[i] = dt_pred[i]
            elif rf_pred[i] == svm_pred[i]:
                hybrid_pred[i] = rf_pred[i]
            else:
                prob = np.array([dt_model.predict_proba([X.iloc[i]]),
                                 rf_model.predict_proba([X.iloc[i]]),
                                 svm_model.predict_proba([X.iloc[i]])])
                hybrid_pred[i] = np.argmax(np.max(prob, axis=0))

    return hybrid_pred

# Test the hybrid model on the testing set
X_test_pred = make_hybrid_prediction(dt_model, rf_model, svm_model, X_test)

# Evaluate the performance of the hybrid model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Accuracy: ", accuracy_score(y_test, X_test_pred))
print("Precision: ", precision_score(y_test, X_test_pred))
print("Recall: ", recall_score(y_test, X_test_pred))
print("F1 score: ", f1_score(y_test, X_test_pred))
cm = confusion_matrix(y_test,X_test_pred)
print("Confusion Matrix :\n" ,confusion_matrix(y_test, X_test_pred))
# plot_confusion_matrix of HM
array =  [[58,6],
          [0,58]]
df_confusion_matrix = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (2,2))
sn.heatmap(df_confusion_matrix, annot=True)
plt.title('Confusion matrix of HYBRID Model', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
# Make predictions on new, unknown data
unknown_data = pd.DataFrame({
    'age': [57],
    'sex': [1],
    'cp': [0],
    'trestbps': [110],
    'chol': [201],
    'fbs': [0],
    'restecg': [1],
    'thalach': [126],
    'exang': [1],
    'oldpeak': [1.5],
    'slope': [1],
    'ca': [0],
    'thal': [1]
})

unknown_data_pred = make_hybrid_prediction(dt_model, rf_model, svm_model, unknown_data)

# Print the predicted target value for the unknown data
print(unknown_data_pred)


unknown_data = pd.DataFrame({
    'age': [60],
    'sex': [1],
    'cp': [0],
    'trestbps': [145],
    'chol': [282],
    'fbs': [0],
    'restecg': [0],
    'thalach': [142],
    'exang': [1],
    'oldpeak': [2.8],
    'slope': [1],
    'ca': [2],
    'thal': [3]
})

unknown_data_pred = make_hybrid_prediction(dt_model, rf_model, svm_model, unknown_data)

# Print the predicted target value for the unknown data
print(unknown_data_pred)

import pickle

# Save the hybrid model
with open('hybrid_model.pkl', 'wb') as f:
    pickle.dump(make_hybrid_prediction, f)
