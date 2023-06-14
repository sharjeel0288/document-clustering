import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Function to read the files from the folder
def read_files(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            files.append(text)
    return files

def preprocess(files):
    # Text cleaning
    files = [re.sub(r'\W+', ' ', text) for text in files]
    
    # Stemming or lemmatization
    stemmer = PorterStemmer()
    files = [' '.join([stemmer.stem(word) for word in text.split()]) for text in files]
    
    # Lowercasing
    files = [text.lower() for text in files]
    
    # Removing rare words
    word_freq = {}
    for text in files:
        for word in text.split():
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    files = [' '.join([word for word in text.split() if word_freq[word] > 1]) for text in files]
    
    # Convert the text into a matrix of token counts
    global vectorizer
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(files)
    
    # Transform the token count matrix into a tf-idf representation
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    
    return X


# Function to perform KMeans clustering
def kmeans_clustering(X, num_clusters):
    # Perform KMeans clustering with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
    kmeans.fit(X)
    
    # Return the labels assigned to each document
    return kmeans.labels_

# Function to calculate purity
def calculate_purity(labels, y):
    n = len(labels)
    k = len(set(labels))
    
    # Construct the contingency table
    p_ij = np.zeros((k, n))
    for i in range(k):
        for j in range(n):
            if labels[j] == i:
                p_ij[i][j] = 1
                
    m_ij = np.zeros((k, len(set(y))))
    for i in range(k):
        for j in range(n):
            if y[j] == "C1":
                m_ij[i][0] += p_ij[i][j]
            elif y[j] == "C2":
                m_ij[i][1] += p_ij[i][j]
            elif y[j] == "C3":
                m_ij[i][2] += p_ij[i][j]
            elif y[j] == "C4":
                m_ij[i][3] += p_ij[i][j]
            elif y[j] == "C5":
                m_ij[i][4] += p_ij[i][j]
    
    # Calculate the purity
    max_in_row = np.max(m_ij, axis=1)
    purity = np.sum(max_in_row)/n
    
    return purity


# Path of the folder containing the documents
doc_folder = "./Doc50"

# Path of the folder containing the ground truth labels
gt_folder = "./Doc50_GT"

# Read the ground truth labels
gt_labels = []
for i in range(1, 6):
    folder_path = os.path.join(gt_folder, f"C{i}")
    files = read_files(folder_path)
    gt_labels += [f"C{i}"]*len(files)
    
    # Print the number of files and the class label for the ground truth folder
    print(f"Ground truth folder {i}: {folder_path}")
    print(f"Number of files: {len(files)}")
    print(f"Class label: {f'C{i}'}\n")

# Read the documents
files = read_files(doc_folder)

# Preprocess the documents
X = preprocess(files)

# Perform KMeans clustering
num_clusters = 5
labels = kmeans_clustering(X, num_clusters)

# Calculate purity
purity = calculate_purity(labels, gt_labels)
print("Purity:", purity)

# Print the number of documents in each class

# Print the number of documents in each class
for i in range(1, 6):
    folder_path = os.path.join(gt_folder, f"C{i}")
    files = read_files(folder_path)
    print(f"Number of documents in class C{i}: {len(files)}")

# Print the ground truth labels
print("\nGround Truth Labels:")
print(gt_labels)

# Print the predicted labels
print("\nPredicted Labels:")
print(labels)

# Print the number of documents in each cluster
for i in range(num_clusters):
    print(f"Number of documents in cluster {i+1}: {sum(labels==i)}")

# Print the purity
print("\nPurity:", purity)

# Grading Criteria:
# Preprocessing
# Reading the documents
print("\nReading the documents...")
files = read_files(doc_folder)

# Preprocessing the documents
print("\nPreprocessing the documents...")
X = preprocess(files)

# Feature Selection
# Get the top 10 most frequent words
print("\nGetting the top 10 most frequent words...")
features = vectorizer.get_feature_names_out()
word_counts = X.sum(axis=0).tolist()[0]
top_words_indices = sorted(range(len(word_counts)), key=lambda i: word_counts[i], reverse=True)[:10]
top_words = [features[i] for i in top_words_indices]
print(f"Top 10 most frequent words: {top_words}")

# K-Mean Implementation
print(f"\nPerforming KMeans clustering with {num_clusters} clusters...")
labels = kmeans_clustering(X, num_clusters)

# Evaluation
print("\nEvaluating the clustering results...")
print(f"Purity: {purity}")
