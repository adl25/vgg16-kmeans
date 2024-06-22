# Image Clustering for Architectural Research

This project demonstrates an image clustering approach using the VGG16 model to extract feature vectors from images. The project applies K-means clustering to these feature vectors and visualizes the results. This method is particularly useful in the field of architecture, where clustering can help analyze and categorize various architectural elements and styles.

## Project Overview

Architectural research often involves analyzing a large number of images to identify patterns and categorize different styles or elements. This project uses a pre-trained VGG16 model to extract high-dimensional feature vectors from images, reduces the dimensionality using PCA, and applies K-means clustering to group similar images together. The results are visualized and saved in clustered folders for further analysis.

## Prerequisites

- Python 3.6 or higher
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/image-clustering-architectural-research.git
    cd image-clustering-architectural-research
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your images in a directory named `train/`.
2. Run the script:
    ```bash
    python image_clustering.py
    ```

## Code Explanation

### 1. Importing Libraries

The necessary libraries for image processing, feature extraction, clustering, and visualization are imported.

### 2. Defining Functions

- `get_model(layer='fc2')`: Loads the VGG16 model and extracts features from the specified layer.
- `get_files(path_to_files, size)`: Loads and resizes images from the specified directory.
- `feature_vector(img_arr, model)`: Extracts feature vectors from an image using the specified model.
- `feature_vectors(imgs_dict, model)`: Extracts feature vectors for all images in the dictionary.

### 3. Loading Images and Preparing the Model

Images are loaded from the `train/` directory and resized. The VGG16 model is loaded to extract feature vectors from the images.

### 4. PCA for Dimensionality Reduction

PCA is applied to reduce the dimensionality of the feature vectors for better visualization and clustering performance.

### 5. K-means Clustering

K-means clustering is applied to the PCA-transformed data to group similar images together. The elbow method is used to determine the optimal number of clusters.

### 6. Visualizing and Saving Results

The clustered results are visualized and saved in separate directories named `kmeans_clusters/cluster_i` where `i` is the cluster number.

## Results

The clustered images are saved in the `kmeans_clusters/` directory. Each subdirectory contains images belonging to the same cluster, which can be used for further architectural analysis.

## Conclusion

This project provides a method to cluster architectural images based on their visual features. By using a pre-trained VGG16 model and K-means clustering, similar architectural elements can be grouped together, facilitating further research and analysis in the field of architecture.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was inspired by the need for effective image analysis tools in architectural research. Special thanks to the contributors and the open-source community for their invaluable resources and support.

