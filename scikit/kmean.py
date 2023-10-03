import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

kmeans = KMeans(n_cluster = 10, random_state=0)
kmeans.fit
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)

for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

plt.show()
