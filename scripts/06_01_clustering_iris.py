import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load iris dataset
iris = datasets.load_iris()
samples = iris.data

# Visualize the data 
df = pd.DataFrame(samples, columns=iris.feature_names)
pd.plotting.scatter_matrix(df, figsize = [8, 8])

# Visualize the data (only petal_length and petal_width)
plt.scatter(samples[:,2],samples[:,3], label='True Position') 
plt.show()

#Create Clusters
model = KMeans(n_clusters=3)
model.fit(samples)
print(model.cluster_centers_) 
print(model.labels_) 

# Visualize how the data has been clustered (only petal_length and petal_width)
plt.scatter(samples[:,2],samples[:,3], c=model.labels_, cmap='rainbow')  
# plot the centroid coordinates of each cluster
plt.scatter(model.cluster_centers_[:,2], model.cluster_centers_[:,3], color='black')
plt.show()

#Cluster labels for new samples
new_samples = [[5.7, 4.4, 1.5, 0.4],[6.5, 3., 5.5, 1.8],  [ 5.8, 2.7, 5.1, 1.9]]
new_labels = model.predict(new_samples)
print(new_labels)
