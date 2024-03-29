import numpy as np
import pandas as pd
import re
import csv
from collections import Counter, defaultdict
import time
import random
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import concurrent.futures

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        self.tol = 1.0e-4
        self.iterations = 0
        self.loss_history = []

    def initialize_parameters(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_predicted):
        # Compute the cross-entropy loss
        n_samples = len(y)
        loss = -(1 / n_samples) * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        return loss

    def fit(self, X, y, method, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
            
        n_samples, n_features = X.shape
        # initialize w
        self.initialize_parameters(n_features)
        last_loss = 0
        # Batch Gradient Descent - BGD
        for i in range(self.num_iterations):
            self.iterations += 1
            # compute y p_value
            epsilon = 1e-10
            y_predicted = self.predict_prob(X)
            y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)

            # compute loss
            loss = self.compute_loss(y, y_predicted)
            self.loss_history.append(loss)

            # compute gradients
            gw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            gb = (1 / n_samples) * np.sum(y_predicted - y)

            if method == 'inverse':
                self.learning_rate = self.learning_rate / (1 + 0.01 * self.iterations)
            
            # update parameters
            self.w -= self.learning_rate * gw
            self.b -= self.learning_rate * gb
            last_loss = loss
            
            # compute the difference between the current loss and the last loss
            if abs(loss - last_loss) < self.tol:
                break
            
    def predict_prob(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_prob(X)
        return np.where(probabilities >= threshold, 1, 0)

    
start_time = time.time()     

preprocessing_re = re.compile(r'[^a-zA-Z\s]')
stop_words = set(['did', 'a', "mightn't", 'these', 'to', 'just', 'his', 'into', 'but', 't', 'mustn', 'other', "won't", 'nor', 'himself', 'mightn', 'by', 've', 'very', 'so', "doesn't", 'which', 'off', 'an', 'with', 'at', 'below', 'your', 'shouldn', 'it', 'are', "needn't", 'hasn', 'that', 'me', 'more', 'no', 'do', 'herself', 'this', 'there', 'under', 'o', 'both', 'some', 'hers', 'over', 'between', 'them', 'been', 'because', 'myself', "don't", 'd', "didn't", 'only', 'on', 'how', 'am', 'who', 'their', "hasn't", 's', 'ours', 'you', "you'd", 'above', 'few', 'was', 'our', 'can', "hadn't", 'shan', 'now', 'once', 'being', 'hadn', 'were', 'whom', "isn't", 'ain', 'will', 'for', 'yours', "that'll", 'should', 'haven', 'those', 'couldn', 'while', 'same', 'themselves', 'itself', 'having', 'where', 'when', 'they', 'had', 'he', 'any', 'the', "should've", 'after', 'or', 'wasn', 'won', 'has', 'does', 'not', "shouldn't", 'than', 're', 'own', "mustn't", "it's", 'have', 'why', 'is', 'and', 'about', 'him', 'doing', 'theirs', 'wouldn', 'll', 'my', 'in', 'of', 'aren', 'needn', 'from', 'up', 'then', "she's", 'ma', "haven't", "you'll", "wasn't", 'y', 'against', 'here', 'further', "you're", 'yourself', 'down', 'before', 'such', 'until', 'isn', 'each', 'its', 'if', 'all', "you've", 'her', 'didn', 'doesn', 'what', "weren't", 'weren', "wouldn't", 'she', 'too', "aren't", 'most', "couldn't", 'i', 'during', 'ourselves', 'through', 'we', 'm', 'as', "shan't", 'out', 'yourselves', 'be', 'don', 'again'])

def preprocessing(text):
    text = text.strip().lower()
    text = text.replace("/", " ")
    text = re.sub(preprocessing_re, ' ', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    remove_text = Counter(filtered_tokens)
    remove_text = ' '.join(remove_text.keys())
    return remove_text    

# crete relevant and non-relevant query-passage pairs
# create relevant and non-relevant dictionaries
def create_rel_and_non_rel_dict(file_path):
    rel_qid_query_pid_passage = defaultdict(lambda: defaultdict(dict))
    non_rel_qid_query_pid_passage = defaultdict(lambda: defaultdict(dict))
    qid_pid_relevance_dict = defaultdict(dict)
    
    with open(file_path, 'r',newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  
        for line in reader:
            qid, pid, query, passage, relevancy = line
            processed_query = preprocessing(query)
            processed_passage = preprocessing(passage)
            if float(relevancy) == 1:
                rel_qid_query_pid_passage[qid][processed_query][pid] = processed_passage
                qid_pid_relevance_dict[qid][pid] = 1
            elif float(relevancy) == 0:
                non_rel_qid_query_pid_passage[qid][processed_query][pid] = processed_passage
                qid_pid_relevance_dict[qid][pid] = 0
    return rel_qid_query_pid_passage, non_rel_qid_query_pid_passage,qid_pid_relevance_dict

validation_rel_qid_query_pid_passage, validation_non_rel_qid_query_pid_passage,validation_rel_dict = create_rel_and_non_rel_dict('validation_data.tsv')
train_rel_qid_query_pid_passage, train_non_rel_qid_query_pid_passage, train_rel_dict  = create_rel_and_non_rel_dict('train_data.tsv')

end_time = time.time()  
print("程序运行时间1：" + str(end_time - start_time))

def count_pid_per_query(data):
    query_pid_count = 0
    for qid, queries in data.items():
        for query, pid_passage in queries.items():
            query_pid_count += len(pid_passage)
    return query_pid_count

# 4797
rel_query_pid_count = count_pid_per_query(train_rel_qid_query_pid_passage) 
# 4359542
non_rel_query_pid_count = count_pid_per_query(train_non_rel_qid_query_pid_passage)
print("Number of relevant query-passage pairs:", rel_query_pid_count)
print("Number of non-relevant query-passage pairs:", non_rel_query_pid_count)

#  use Random Negative Sampling for generating a subset of training data 1:5
def sampling_non_rel_passage(data, sample_size):
    new_data = {}
    for qid, queries in data.items():
        new_data[qid] = {}
        for query, pid_passage in queries.items():
            # take all if less than five
            sampled_pids = random.sample(list(pid_passage.keys()), min(sample_size, len(pid_passage)))
            new_data[qid][query] = {pid: pid_passage[pid] for pid in sampled_pids}
    return new_data
sampled_train_non_rel_qid_query_pid_passage = sampling_non_rel_passage(train_non_rel_qid_query_pid_passage, 9)

# word embedding model training
def create_query_passage_list(sampled_train_non_rel_qid_query_pid_passage, sampled_train_rel_qid_query_pid_passage):
    query_set = set()
    passage_list = []

    qid_to_words = {}  
    pid_to_words = {} 

    for sample in [sampled_train_non_rel_qid_query_pid_passage, sampled_train_rel_qid_query_pid_passage]:
        for qid, query_dict in sample.items():
            for query, passage_dict in query_dict.items():
                if qid not in qid_to_words:
                    query_set.add(qid)
                    qid_to_words[qid] = query.split()

                for pid, passage in passage_dict.items():
                    if pid not in pid_to_words:
                        passage_list.append(passage)
                        pid_to_words[pid] = passage.split()

    new_query_list = [qid_to_words[qid] for qid in query_set]
    new_passage_list = [text.split() for text in passage_list]

    return new_query_list, new_passage_list, qid_to_words, pid_to_words

train_query_list, train_passage_list, train_qid_to_terms, train_pid_to_terms = create_query_passage_list(sampled_train_non_rel_qid_query_pid_passage, train_rel_qid_query_pid_passage)
validation_query_list, validation_passage_list, validation_qid_to_terms, validation_pid_to_terms = create_query_passage_list(validation_non_rel_qid_query_pid_passage, validation_rel_qid_query_pid_passage)

del sampled_train_non_rel_qid_query_pid_passage, train_rel_qid_query_pid_passage, validation_non_rel_qid_query_pid_passage, validation_rel_qid_query_pid_passage

# print("Number of unique queries in training data:", len(train_query_list))
# print("Number of unique queries in validation data:", len(validation_query_list))
# print("Number of unique passages in training data:", len(train_passage_list))
# print("Number of unique passages in validation data:", len(validation_passage_list))

training_corpus = train_query_list + train_passage_list
# iter 10 times
train_model = Word2Vec(sentences=training_corpus, vector_size=100, window=5, min_count=1, sg=1, workers=4, epochs=10)

end_time = time.time()  
print("程序运行时间2：" + str(end_time - start_time))

# Compute query(/passage) embeddings
def get_average_embedding(word2vec_model, terms):
    
    embeddings = [word2vec_model.wv[term] for term in terms if term in word2vec_model.wv]
    
    if embeddings:
        embedding = np.mean(embeddings, axis=0)
    else:
        embedding = np.zeros(word2vec_model.vector_size)

    return embedding

train_qid_embeddings = {qid: get_average_embedding(train_model, terms) for qid, terms in train_qid_to_terms.items()}
train_pid_embeddings = {pid: get_average_embedding(train_model, terms) for pid, terms in train_pid_to_terms.items()}
validation_qid_embeddings = {qid: get_average_embedding(train_model, terms) for qid, terms in validation_qid_to_terms.items()}
validation_pid_embeddings = {pid: get_average_embedding(train_model, terms) for pid, terms in validation_pid_to_terms.items()}

end_time = time.time()  
print("程序运行时间3：" + str(end_time - start_time))

def calculate_features_and_labels(qid_embeddings, pid_embeddings, qid_to_terms, pid_to_terms, rel_dict):
    x = []
    y = []

    for qid, pid_rel_dict in rel_dict.items():
        for pid, rel in pid_rel_dict.items():
            if qid in qid_embeddings and pid in pid_embeddings:
                qid_embedding = qid_embeddings[qid]
                pid_embedding = pid_embeddings[pid]

                # use embedding and term frequency as features
                feature_vector = np.concatenate([qid_embedding, pid_embedding])

                x.append(feature_vector)
                y.append(rel)

    return np.array(x), np.array(y)

train_X, train_y = calculate_features_and_labels(train_qid_embeddings, train_pid_embeddings, train_qid_to_terms, train_pid_to_terms, train_rel_dict)
validation_X, validation_y = calculate_features_and_labels(validation_qid_embeddings, validation_pid_embeddings, validation_qid_to_terms, validation_pid_to_terms, validation_rel_dict)

print("Shape of training features:", train_X.shape)
print("Shape of training labels:", train_y.shape)
print("Shape of validation features:", validation_X.shape)
print("Shape of validation labels:", validation_y.shape)

learning_rates = [100, 10, 1, 0.1, 0.01, 0.001]
loss_histories = {}  

end_time = time.time()  
print("程序运行时间4：" + str(end_time - start_time))

for lr in learning_rates:
    model = LogisticRegression(learning_rate=lr, num_iterations=300)
    model.fit(train_X, train_y, None, learning_rate=lr)
    loss_histories[lr] = model.loss_history

# for lr, loss_history in loss_histories.items():
#     plt.plot(loss_history, label=f'LR={lr}')

# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Loss vs. Iterations for different learning rates')
# plt.show()

end_time = time.time()  
print("程序运行时间5：" + str(end_time - start_time))
