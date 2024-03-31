import numpy as np
import pandas as pd
import re
import csv
from collections import Counter, defaultdict
import time
import random
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import copy
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        self.tol = 1.0e-5
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

    def fit(self, X, y, method, learning_rate=None, fix = False):
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
                self.learning_rate = self.learning_rate / (1 + 0.1 * self.iterations)
            
            # update parameters
            self.w -= self.learning_rate * gw
            self.b -= self.learning_rate * gb
            
            # compute the difference between the current loss and the last loss
            if not fix and abs(loss - last_loss) < self.tol:
                break
                
            last_loss = loss   
            
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
    text = preprocessing_re.sub(' ', text)
    tokens = text.split()
    filtered_tokens = set(word for word in tokens if word not in stop_words)
    return ' '.join(filtered_tokens)

# crete relevant and non-relevant query-passage pairs
# create qid and pid relevancy dictionaries
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
# print("Number of relevant query-passage pairs:", rel_query_pid_count)
# print("Number of non-relevant query-passage pairs:", non_rel_query_pid_count)

#  use Random Negative Sampling for generating a subset of training data
def sampling_non_rel_passage(data, sample_size):
    new_data = {}
    for qid, queries in data.items():
        new_data[qid] = {}
        for query, pid_passage in queries.items():
            sampled_pids = random.sample(list(pid_passage.keys()), min(sample_size, len(pid_passage)))
            new_data[qid][query] = {pid: pid_passage[pid] for pid in sampled_pids}
    return new_data
# 比例
sampled_train_non_rel_qid_query_pid_passage = sampling_non_rel_passage(train_non_rel_qid_query_pid_passage, 1)
# print(len(sampled_train_non_rel_qid_query_pid_passage))

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

def build_qid_pid_rel_dict(data, rel):
    qid_pid_rel = {}
    for qid, query_dict in data.items():
        qid_pid_rel[qid] = {}
        for query, pid_passage in query_dict.items():
            for pid, passage in pid_passage.items():
                qid_pid_rel[qid][pid] = rel 
    return qid_pid_rel

# 使用函数构建字典
sampled_train_non_rel_qid_pid_rel = build_qid_pid_rel_dict(sampled_train_non_rel_qid_query_pid_passage, 0)
train_rel_qid_pid_rel = build_qid_pid_rel_dict(train_rel_qid_query_pid_passage,1)

def merge_dicts(dict1, dict2):
    merged_dict = {**dict1}  # 复制第一个字典
    for qid, pid_rel in dict2.items():
        if qid in merged_dict:
            # 如果 qid 已存在，则合并 pid 和 rel 值
            merged_dict[qid].update(pid_rel)
        else:
            # 如果 qid 不存在，直接添加到结果字典中
            merged_dict[qid] = pid_rel
    return merged_dict

# 使用函数合并字典
merged_qid_pid_rel = merge_dicts(sampled_train_non_rel_qid_pid_rel, train_rel_qid_pid_rel)

del sampled_train_non_rel_qid_query_pid_passage, train_rel_qid_query_pid_passage, validation_non_rel_qid_query_pid_passage, validation_rel_qid_query_pid_passage

# print("Number of unique queries in training data:", len(train_query_list))
# print("Number of unique queries in validation data:", len(validation_query_list))
# print("Number of unique passages in training data:", len(train_passage_list))
# print("Number of unique passages in validation data:", len(validation_passage_list))

training_corpus = train_query_list + train_passage_list
# print(len(training_corpus))
# print(train_query_list[:1])
# print(train_passage_list[:1])
# iter 10 times
train_model = Word2Vec(sentences=training_corpus, vector_size=100, window=5, min_count=1, sg=1, workers=4, epochs=10)

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

def calculate_features_and_labels(qid_embeddings, pid_embeddings, qid_to_terms, pid_to_terms, rel_dict):
    rows = []

    for qid, pid_rel_dict in rel_dict.items():
        for pid, rel in pid_rel_dict.items():
            if qid in qid_embeddings and pid in pid_embeddings:
                qid_terms_embedding = qid_embeddings[qid]
                pid_terms_embedding = pid_embeddings[pid]
                # use embedding
                feature_vector = np.concatenate([qid_terms_embedding, pid_terms_embedding])
                # append qid, pid, feature_vector, and label to the rows list
                rows.append((qid, pid, feature_vector, rel))

    # Create a DataFrame from the rows
    df = pd.DataFrame(rows, columns=['qid', 'pid', 'features', 'label'])

    return df

# 使用更新的函数获取DataFrame、特征矩阵和标签数组
train_data_df = calculate_features_and_labels(train_qid_embeddings, train_pid_embeddings, train_qid_to_terms, train_pid_to_terms, merged_qid_pid_rel)
validation_data_df = calculate_features_and_labels(validation_qid_embeddings, validation_pid_embeddings, validation_qid_to_terms, validation_pid_to_terms, validation_rel_dict)

print("Shape of training features:", np.stack(train_data_df['features'].values).shape)
print("Shape of training labels:", train_data_df['label'].values.shape)
print("Shape of validation features:", np.stack(validation_data_df['features'].values).shape)
print("Shape of validation labels:",validation_data_df['label'].values.shape)

learning_rates = [0.001, 0.01, 0.1, 0.5, 0.8 , 1]
loss_histories = {}
stop_iteration_times = {}  

train_X = np.stack(train_data_df['features'].values)
train_y = train_data_df['label'].values
validation_X = np.stack(validation_data_df['features'].values)

mean = np.mean(train_X, axis=0)
std = np.std(train_X, axis=0)
std += 1e-10
train_X_normalized = (train_X - mean) / std
validation_X_normalized = (validation_X - mean) / std
# print("train_X_normalized: ", train_X_normalized[:1])
# print("validation_X_normalized: ", validation_X_normalized[:1])
# print("train_X", train_X[:1])
# print("validation_X", validation_X[:1])

validation_qids = validation_data_df['qid'].values
validation_pids = validation_data_df['pid'].values

# store results for each learning rate
results_dfs = {}  
for lr in learning_rates:
    model = LogisticRegression(learning_rate=lr, num_iterations=300)
    model.fit(train_X_normalized, train_y, None, learning_rate=lr)
    # loss histories for each learning rate
    loss_histories[lr] = model.loss_history
    # print(model.w, model.b)
    stop_iteration_times[lr] = model.iterations

    # predict probabilities for the validation set in each learning rate train model
    predicted_probs = model.predict_prob(validation_X_normalized)

    results = []
    for qid, pid, prob in zip(validation_qids, validation_pids, predicted_probs):
        results.append((qid, pid, prob))

    results_dfs[lr] = pd.DataFrame(results, columns=['qid', 'pid', 'prob'])   

sorted_results_dfs_prob = {}  

for lr, df in results_dfs.items():
    # sort by qid and then by probability in descending order
    sorted_df = df.sort_values(by=['qid', 'prob'], ascending=[True, False])
    sorted_results_dfs_prob[lr] = sorted_df.reset_index(drop=True)
  
sorted_results_dfs_binrel = copy.deepcopy(sorted_results_dfs_prob)  

for lr, df in sorted_results_dfs_binrel.items():
    df['prob'] = np.where(df['prob'] >= 0.5, 1.0, 0.0)  

# for lr, df in sorted_results_dfs_binrel.items():    
#     print(lr , ":sorted_results_dfs_pro: \n", sorted_results_dfs_prob[lr][:1])
    # print("sorted_results_dfs_rel: \n", sorted_results_dfs_binrel[lr][:1]) 
# print("sorted_results_def_len len: ", len(sorted_results_dfs_prob[0.1]))

lr_groups = [learning_rates[:3], learning_rates[3:6]]
# lr_groups = [learning_rates[:6]]

for i, lrs in enumerate(lr_groups):
    plt.figure(figsize=(10, 6))
    for lr in lrs:
        plt.plot(loss_histories[lr], label=f'LR={lr}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss vs. Iterations for learning rates')
    plt.savefig(f"loss_group_unfix-1{i+1}.pdf", format='pdf')
    plt.show()

# qid_to_pid_rel = lr: {qid:{pid:rel}} 
qid_to_pid_rel = {}
for lr, df in sorted_results_dfs_binrel.items():
    qid_to_pid_rel[lr] = {qid: dict(zip(group['pid'], group['prob']))
                          for qid, group in df.groupby('qid')}

def cal_AP(LR_rank, qid_and_pid_and_rel):
    total_AP_for_each_query = 0
    for qid,pids in qid_and_pid_and_rel.items():
        counted_passage = 0
        found_rel_passage = 0
        cur_query_precision = 0
        for pid,rel in LR_rank[qid].items():
            counted_passage += 1
            if rel == 1.0 and qid_and_pid_and_rel[qid][pid] == 1:
                found_rel_passage += 1
                cur_query_precision += found_rel_passage / counted_passage  
        if(found_rel_passage != 0):
            total_AP_for_each_query += cur_query_precision / found_rel_passage   
                
    return total_AP_for_each_query / len(qid_and_pid_and_rel)

def cal_NDCG(LR_rank, qid_and_pid_and_rel):
    total_NDCG_for_each_query = 0
    for qid, pids in qid_and_pid_and_rel.items():
        counted_passage = 0
        counted_passage_opt = 0
        rel_count = 0
        DCG = 0
        opt_DCG = 0
        for pid,rel in LR_rank[qid].items():
            counted_passage += 1
            reli = 0
            if qid_and_pid_and_rel[qid][pid] == 1:
                rel_count += 1
                reli = 1 
            DCG += (2 ** reli - 1) / np.log2(counted_passage + 1)   
        total_passage_count = len(LR_rank[qid])
        # create an array set rel_count length as 1 and rest as 0
        optimal_relevance = [1 if i < rel_count else 0 for i in range(total_passage_count)]    
        for rel in optimal_relevance:
            counted_passage_opt += 1
            opt_DCG += (2 ** rel - 1) / np.log2(counted_passage_opt + 1)    
             
        NDCG = DCG / opt_DCG if opt_DCG != 0 else 0
        total_NDCG_for_each_query += NDCG       

    return total_NDCG_for_each_query / len(qid_and_pid_and_rel)

# LR_AP = {lr:AP}
LR_AP = {}
LR_NDCG = {}
for lr in qid_to_pid_rel.keys():
    LR_AP[lr] = cal_AP(qid_to_pid_rel[lr], validation_rel_dict)
    LR_NDCG[lr] = cal_NDCG(qid_to_pid_rel[lr], validation_rel_dict)
    
print('AP',LR_AP)    
print('NDCG',LR_NDCG)
print(stop_iteration_times)

end_time = time.time()  
print("程序运行时间5：" + str(end_time - start_time))
