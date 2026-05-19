# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the spam dataset and preprocess the data.

2.Convert text messages into numerical form using TF-IDF Vectorizer.

3.Train the SVM classifier using training and testing data.

4.Predict the output and display the confusion matrix using a heatmap.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ishwarya s
RegisterNumber:  212225240053
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Customer Segmentation using K-Means")

plt.show()
```

## Output:
<img width="748" height="440" alt="Screenshot 2026-05-19 144351" src="https://github.com/user-attachments/assets/d128869b-cded-4d6b-8367-b88760f0be84" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
