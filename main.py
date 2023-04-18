from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv(r"datasets/CC GENERAL.csv")  # Read in data from file

# Data Preprocessing - Creates new column so ID will be int instead of string
dataset.rename({'CUST_ID': 'OLD_ID'}, axis='columns', inplace=True)  # Update column name to destroy
dataset.insert(0, "CUST_ID", 1)  # Insert new column
dataset['CUST_ID'] = dataset['CUST_ID'].astype(int)  # Update value to int

for i, row in dataset.iterrows():  # Update values from OLD_ID to CUST_ID
    tmp = row['OLD_ID']
    dataset.at[i, 'CUST_ID'] = int(tmp[1:])

del dataset['OLD_ID']  # Delete old id column
# print(dataset.head())

dataset = dataset.dropna(axis=0)  # Drop any rows with no values

X = dataset.drop('TENURE', axis=1).values  # Separate column used for training and test
y = dataset['TENURE']  # Saves values of dropped column

pca = PCA(2)  # Create instance of PCA class
X = pca.fit_transform(X)  # Fit to training data

# Applying k-means
nclusters = 2  # Use two due to findings from elbow method
km = KMeans(n_clusters=nclusters)  # Input number of clusters to KMeans
km.fit(X)  # Fit data to model

y_cluster_kmeans = km.predict(X)  # Predict the cluster on each data point
score = metrics.silhouette_score(X, y_cluster_kmeans)  # Find silhouette_score
print("Silhouette score of data before scaling: ", score)  # Print results

scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled_array = scaler.transform(X)

# Applying k-means again
nclusters = 2  # Use two due to findings from elbow method
km = KMeans(n_clusters=nclusters)  # Input number of clusters to KMeans
km.fit(X_scaled_array)  # Fit data to model

y_cluster_kmeans = km.predict(X)  # Predict the cluster on each data point
score = metrics.silhouette_score(X, y_cluster_kmeans)  # Find silhouette_score
print("Silhouette score of data after scaling: ", score)  # Print results

# Question 2
dataset = pd.read_csv("datasets/pd_speech_features.csv")  # Read in data from file

X = dataset.drop('class', axis=1).values  # Get x values separate from y
y = dataset['class'].values  # Get y value for classification

scaler = StandardScaler()  # Create instance of scaler class
X_Scale = scaler.fit_transform(X)  # Fit transform scaler values

pca3 = PCA(n_components=3)  # Create pca instance with 3 components
x_pca = pca3.fit_transform(X_Scale)  # Fit and transform values
df2 = pd.DataFrame(data=x_pca)  # Create dataframe from fit and transformed values
finaldf = pd.concat([df2, dataset[['class']]], axis=1)  # Get PCA result

X = finaldf.drop('class', axis=1).values  # Get values separate from y
y = finaldf['class'].values  # Get values of class column for spilt

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)  # Spilt up data

classifier_svc = SVC()  # Create instance of class
classifier_svc.fit(X_train, y_train)  # Fit model
y_pred = classifier_svc.predict(X_test)  # Make predictions
print('Accuracy for SVC: ', accuracy_score(y_pred, y_test))  # Accuracy score

# Question 3
dataset = pd.read_csv("datasets/Iris.csv")  # Read in data from file

X = dataset.drop('Species', axis=1).values  # Get target seperate from class
y = dataset['Species'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)  # Spilt up data

scaler = StandardScaler()  # Create instance of StandardScaler class
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)  # Transform test data

lda = LinearDiscriminantAnalysis(n_components=2)  # Create instance of LDA class with 2 components
X_train = lda.fit_transform(X_train, y_train)  # Transform training data and fit with LDA
X_test = lda.transform(X_test)  # Transform test data with LDA

# Add LogisticRegression to get accuracy
lr = LogisticRegression()  # Create instance of LogisticRegression class
lr.fit(X_train, y_train)  # Fit training model

y_pred = lr.predict(X_test)  # Make predictions for model

print('Accuracy for Linear Discriminant Analysis: ', accuracy_score(y_pred, y_test))  # Print accuracy score
