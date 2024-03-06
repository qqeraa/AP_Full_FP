import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

diabetes_data = pd.read_csv('diabetes.csv')
diabetes_data.head()
diabetes_data.info()
diabetes_data['Outcome'].value_counts()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=diabetes_data, palette='viridis', alpha=0.7)
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('BMI vs Glucose')
plt.legend(title='Outcome', loc='upper right')
plt.show()


corr_matrix = diabetes_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heat map')
plt.show()



features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

plt.figure(figsize=(20, 25))
for i, feature in enumerate(features, start=1):
    plt.subplot(4, 2, i)
    sns.histplot(data=diabetes_data, x=feature, hue='Outcome', kde=True, palette=['green', 'red'], edgecolor='black')
    plt.title(f"Distribution of {feature}", fontsize=18, fontweight='bold')
    plt.xlabel(feature, fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.tight_layout()

plt.show()



X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled, X_test_scaled


param_lr_C = {'C': [0.01, 0.1, 1, 10, 100, 1000]}
grid_search_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_lr_C, cv=5)
grid_search_lr.fit(X_train_scaled, y_train)

best_param_lr_C = grid_search_lr.best_params_['C']
print("Best C parameter for LR:", best_param_lr_C)


plt.figure(figsize=(10, 6))
sns.lineplot(x=[0.01, 0.1, 1, 10, 100, 1000], y=grid_search_lr.cv_results_['mean_test_score'], marker='o')
plt.xscale('log')
plt.xlabel('C parameter')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C parameter')
plt.grid(True)
plt.show()



param_knn_K = {'n_neighbors': np.arange(1, 31)}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_knn_K, cv=5)
grid_search_knn.fit(X_train_scaled, y_train)

best_param_knn_K = grid_search_knn.best_params_['n_neighbors']
print("Best K parameter for KNN:", best_param_knn_K)


plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(1, 31), y=grid_search_knn.cv_results_['mean_test_score'], marker='o')
plt.xlabel('K parameter')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K parameter')
plt.grid(True)
plt.xticks(np.arange(1, 31))
plt.show()



param_svm_C = {'C': [0.01, 0.1, 1, 10, 100, 1000]}
grid_search_svm = GridSearchCV(SVC(), param_svm_C, cv=5)
grid_search_svm.fit(X_train_scaled, y_train)

best_param_svm_C = grid_search_svm.best_params_['C']
print("Best C parameter for SVM:", best_param_svm_C)

plt.figure(figsize=(10, 6))
sns.lineplot(x=[0.01, 0.1, 1, 10, 100, 1000], y=grid_search_svm.cv_results_['mean_test_score'], marker='o')
plt.xscale('log')
plt.xlabel('C parameter')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C parameter')
plt.grid(True)
plt.show()



param_dt_max_depth = {'max_depth': np.arange(1, 21)}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=27), param_dt_max_depth, cv=5)
grid_search_dt.fit(X_train_scaled, y_train)

best_param_dt_max_depth = grid_search_dt.best_params_['max_depth']
print("Best max depth parameter for DT:", best_param_dt_max_depth)

plt.figure(figsize=(10, 6))
sns.lineplot(x=np.arange(1, 21), y=grid_search_dt.cv_results_['mean_test_score'], marker='o')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max depth parameter')
plt.grid(True)
plt.xticks(np.arange(1, 21))
plt.show()



lr_pred = grid_search_lr.predict(X_test_scaled)
knn_pred = grid_search_knn.predict(X_test_scaled)
svm_pred = grid_search_svm.predict(X_test_scaled)
dt_pred = grid_search_dt.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)

evaluation_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall'])

evaluation_df.loc['Logistic Regression'] = [lr_accuracy, lr_precision, lr_recall]
evaluation_df.loc['KNN'] = [knn_accuracy, knn_precision, knn_recall]
evaluation_df.loc['SVM'] = [svm_accuracy, svm_precision, svm_recall]
evaluation_df.loc['Decision Tree'] = [dt_accuracy, dt_precision, dt_recall]

print("Evaluation Metrics:")
print(evaluation_df)

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
evaluation_df['Accuracy'].plot(kind='barh', color='skyblue')
plt.title('Accuracy')
plt.xlim(0, 1)

plt.subplot(3, 1, 2)
evaluation_df['Precision'].plot(kind='barh', color='salmon')
plt.title('Precision')
plt.xlim(0, 1)

plt.subplot(3, 1, 3)
evaluation_df['Recall'].plot(kind='barh', color='lightgreen')
plt.title('Recall')
plt.xlim(0, 1)

plt.tight_layout()
plt.show()

