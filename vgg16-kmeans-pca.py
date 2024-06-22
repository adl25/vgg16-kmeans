import cv2
import os
import numpy as np
from keras.models import Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import matplotlib.image as mpimg
from sklearn.decomposition import PCA

def get_model(layer='fc2'):
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)
    return model

def get_files(path_to_files, size):
    fn_imgs = []
    files = [file for file in os.listdir(path_to_files)]
    for file in files:
        img = cv2.resize(cv2.imread(path_to_files + file), size)
        fn_imgs.append([file, img])
    return dict(fn_imgs)

def feature_vector(img_arr, model):
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis=2)
    arr4d = np.expand_dims(img_arr, axis=0)
    arr4d_pp = preprocess_input(arr4d)
    return model.predict(arr4d_pp)[0, :]

def feature_vectors(imgs_dict, model):
    f_vect = {}
    for fn, img in imgs_dict.items():
        f_vect[fn] = feature_vector(img, model)
    return f_vect

# Load images and prepare the model
path_to_files = 'train/'
size = (224, 224)
imgs_dict = get_files(path_to_files, size)
model = get_model()
img_feature_vector = feature_vectors(imgs_dict, model)

# Reduce dimensions with PCA
images = list(img_feature_vector.values())
df = np.array(images)
pca = PCA(2)
data = pca.fit_transform(df)

# K-means Clustering
sum_of_squared_distances = []
K = range(1, 30)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data)
    sum_of_squared_distances.append(km.inertia_)
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Optimal k value')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal k')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
kmeans.fit(data)
y_kmeans = kmeans.predict(data)
file_names = list(imgs_dict.keys())

# Visualize the results
centroids = kmeans.cluster_centers_
u_labels = np.unique(y_kmeans)

plt.scatter(data[:, 0], data[:, 1], s=10, color='k')
plt.title('2D Scatter Plot of Images')
plt.show()

for i in u_labels:
    plt.scatter(data[y_kmeans == i, 0], data[y_kmeans == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=10, color='k')
plt.title('Clustered Images')
plt.legend()
plt.show()

# Save clustered images
n_clusters = 5
cluster_path = 'kmeans_clusters/'
if not os.path.exists(cluster_path):
    os.mkdir(cluster_path)

for c in range(n_clusters):
    if not os.path.exists(cluster_path + 'cluster_' + str(c)):
        os.mkdir(cluster_path + 'cluster_' + str(c))

for fn, cluster in zip(file_names, y_kmeans):
    img = cv2.imread(path_to_files + fn)
    cv2.imwrite(cluster_path + 'cluster_' + str(cluster) + '/' + fn, image)

# Visualize clustered images
fig = plt.figure(figsize=(20, 20))

cluster_0_path = cluster_path + 'cluster_0/'
images = [file for file in os.listdir(cluster_0_path)]

for cnt, data in enumerate(images[1:18]):
    print(data)
    y = fig.add_subplot(3, 6, cnt + 1)
    img = mpimg.imread(cluster_0_path + data)
    y.imshow(img)
    plt.title('cluster_0')
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
