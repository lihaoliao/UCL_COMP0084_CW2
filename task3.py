import itertools
import random
import pandas as pd
import numpy as np
import re
from collections import OrderedDict, defaultdict
import csv
import xgboost as xgb
from gensim.models import Word2Vec

stop_words = set(['did', 'a', "mightn't", 'these', 'to', 'just', 'his', 'into', 'but', 't', 'mustn', 'other', "won't", 'nor', 'himself', 'mightn', 'by', 've', 'very', 'so', "doesn't", 'which', 'off', 'an', 'with', 'at', 'below', 'your', 'shouldn', 'it', 'are', "needn't", 'hasn', 'that', 'me', 'more', 'no', 'do', 'herself', 'this', 'there', 'under', 'o', 'both', 'some', 'hers', 'over', 'between', 'them', 'been', 'because', 'myself', "don't", 'd', "didn't", 'only', 'on', 'how', 'am', 'who', 'their', "hasn't", 's', 'ours', 'you', "you'd", 'above', 'few', 'was', 'our', 'can', "hadn't", 'shan', 'now', 'once', 'being', 'hadn', 'were', 'whom', "isn't", 'ain', 'will', 'for', 'yours', "that'll", 'should', 'haven', 'those', 'couldn', 'while', 'same', 'themselves', 'itself', 'having', 'where', 'when', 'they', 'had', 'he', 'any', 'the', "should've", 'after', 'or', 'wasn', 'won', 'has', 'does', 'not', "shouldn't", 'than', 're', 'own', "mustn't", "it's", 'have', 'why', 'is', 'and', 'about', 'him', 'doing', 'theirs', 'wouldn', 'll', 'my', 'in', 'of', 'aren', 'needn', 'from', 'up', 'then', "she's", 'ma', "haven't", "you'll", "wasn't", 'y', 'against', 'here', 'further', "you're", 'yourself', 'down', 'before', 'such', 'until', 'isn', 'each', 'its', 'if', 'all', "you've", 'her', 'didn', 'doesn', 'what', "weren't", 'weren', "wouldn't", 'she', 'too', "aren't", 'most', "couldn't", 'i', 'during', 'ourselves', 'through', 'we', 'm', 'as', "shan't", 'out', 'yourselves', 'be', 'don', 'again'])
preprocessing_re = re.compile(r'[^a-zA-Z\s]')

def preprocessing(text):
    text = text.strip().lower()
    text = preprocessing_re.sub(' ', text)
    tokens = text.split()
    filtered_tokens = set(word for word in tokens if word not in stop_words)
    return ' '.join(filtered_tokens)

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

def read_data(file_path, num_rows=None):
    return pd.read_csv(file_path, sep='\t', nrows=num_rows)

validation_rel_qid_query_pid_passage, validation_non_rel_qid_query_pid_passage, validation_rel_dict = create_rel_and_non_rel_dict('validation_data.tsv')
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

#  use Random Negative Sampling for generating a subset of training data
def sampling_non_rel_passage(data, sample_size):
    new_data = {}
    for qid, queries in data.items():
        new_data[qid] = {}
        for query, pid_passage in queries.items():
            sampled_pids = random.sample(list(pid_passage.keys()), min(sample_size, len(pid_passage)))
            new_data[qid][query] = {pid: pid_passage[pid] for pid in sampled_pids}
    return new_data

sampled_train_non_rel_qid_query_pid_passage = sampling_non_rel_passage(train_non_rel_qid_query_pid_passage, 5)

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

sampled_train_non_rel_qid_pid_rel = build_qid_pid_rel_dict(sampled_train_non_rel_qid_query_pid_passage, 0)
train_rel_qid_pid_rel = build_qid_pid_rel_dict(train_rel_qid_query_pid_passage,1)

def merge_dicts(dict1, dict2):
    merged_dict = {**dict1}
    for qid, pid_rel in dict2.items():
        if qid in merged_dict:
            merged_dict[qid].update(pid_rel)
        else:
            merged_dict[qid] = pid_rel
    return merged_dict

merged_qid_pid_rel = merge_dicts(sampled_train_non_rel_qid_pid_rel, train_rel_qid_pid_rel)
del sampled_train_non_rel_qid_query_pid_passage, train_rel_qid_query_pid_passage, validation_non_rel_qid_query_pid_passage, validation_rel_qid_query_pid_passage

training_corpus = train_query_list + train_passage_list
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

# 'qid', 'pid', 'features', 'label'
train_data_df = calculate_features_and_labels(train_qid_embeddings, train_pid_embeddings, train_qid_to_terms, train_pid_to_terms, merged_qid_pid_rel)
validation_data_df = calculate_features_and_labels(validation_qid_embeddings, validation_pid_embeddings, validation_qid_to_terms, validation_pid_to_terms, validation_rel_dict)

del train_qid_to_terms, train_pid_to_terms, validation_qid_to_terms, validation_pid_to_terms, train_qid_embeddings, train_pid_embeddings, validation_qid_embeddings, validation_pid_embeddings

sorted_train_data_df = train_data_df.sort_values(by='qid').reset_index(drop=True)
sorted_train_qid_counts = sorted_train_data_df.groupby('qid').size()

sorted_validation_data_df = validation_data_df.sort_values(by='qid').reset_index(drop=True)
sorted_validation_qid_counts = sorted_validation_data_df.groupby('qid').size()

group_train = []
group_valid = []
for qid, count in sorted_train_qid_counts.items():
    group_train.append(count)
for qid, count in sorted_validation_qid_counts.items():
    group_valid.append(count)
 
train_X = np.stack(sorted_train_data_df['features'].values)
train_y = sorted_train_data_df['label'].values
train_qid_list = sorted_train_data_df['qid'].values
train_pid_list = sorted_train_data_df['pid'].values

validation_X = np.stack(sorted_validation_data_df['features'].values)
validation_y = sorted_validation_data_df['label'].values

validation_qid_list = sorted_validation_data_df['qid'].values
validation_pid_list = sorted_validation_data_df['pid'].values

mean = np.mean(train_X, axis=0)
std = np.std(train_X, axis=0)
std += 1e-10
train_X_normalized = (train_X - mean) / std

mean_v = np.mean(validation_X, axis=0)
std_v = np.std(validation_X, axis=0)
std += 1e-10
validation_X_normalized = (validation_X - mean_v) / std_v

train_dmatrix = xgb.DMatrix(train_X_normalized, label=train_y)
valid_dmatrix = xgb.DMatrix(validation_X_normalized, label=validation_y)

cv_X = np.concatenate([train_X_normalized, validation_X_normalized])
cv_qid_list = np.concatenate([train_qid_list, validation_qid_list])
cv_pid_list = np.concatenate([train_pid_list, validation_pid_list])
cv_y = np.concatenate([train_y, validation_y])

train_dmatrix.set_group(group_train)
valid_dmatrix.set_group(group_valid)

train_validation_rel_dict = merge_dicts(merged_qid_pid_rel, validation_rel_dict)

LM_AP = 0
LM_NDCG = 0

def cal_AP(LM_rank, qid_and_pid_and_rel):
    total_AP_for_each_query = 0
    for qid,pids in qid_and_pid_and_rel.items():
        counted_passage = 0
        found_rel_passage = 0
        cur_query_precision = 0
        for pid, prob in LM_rank[qid]:
            counted_passage += 1
            if qid_and_pid_and_rel[qid][pid] == 1.0:
                found_rel_passage += 1
                cur_query_precision += found_rel_passage / counted_passage  
        if(found_rel_passage != 0):
            total_AP_for_each_query += cur_query_precision / found_rel_passage   
                
    return total_AP_for_each_query / len(qid_and_pid_and_rel)

def cal_NDCG(LM_rank, qid_and_pid_and_rel):
    total_NDCG_for_each_query = 0
    for qid, pids in qid_and_pid_and_rel.items():
        counted_passage = 0
        counted_passage_opt = 0
        rel_count = 0
        DCG = 0
        opt_DCG = 0
        for pid, prob in LM_rank[qid]:
            counted_passage += 1
            reli = 0
            if qid_and_pid_and_rel[qid][pid] == 1.0:
                rel_count += 1
                reli = 1 
            DCG += (2 ** reli - 1) / np.log2(counted_passage + 1)   
        total_passage_count = len(LM_rank[qid])
        # create an array set rel_count length as 1 and rest as 0
        optimal_relevance = [1 if i < rel_count else 0 for i in range(total_passage_count)]    
        for rel in optimal_relevance:
            counted_passage_opt += 1
            opt_DCG += (2 ** rel - 1) / np.log2(counted_passage_opt + 1)    
             
        NDCG = DCG / opt_DCG if opt_DCG != 0 else 0
        total_NDCG_for_each_query += NDCG       

    return total_NDCG_for_each_query / len(qid_and_pid_and_rel)

best_AP = 0
best_NDCG = 0
best_eta = 0
best_gamma = 0
best_min_child_weight = 0
best_max_depth = 0

# adjusting here for the number of iterations and other hyperparameters
for i in range(10):
    print("iteration: ", i)
    eta_range = [0.001, 0.01, 0.1, 0.5, 1]
    gamma_range = [0, 0.1, 0.5, 1.0]
    min_child_weight_range = [1, 3, 5]
    max_depth_range = [4, 6, 8, 10]

    eta = random.choice(eta_range)
    gamma = random.choice(gamma_range)
    min_child_weight = random.choice(min_child_weight_range)   
    max_depth = random.choice(max_depth_range)
    
    params = {
        # 'objective': 'rank:ndcg',
        # 'objective': 'rank:map',
        'objective': 'rank:pairwise',
        'eta': eta,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'max_depth': max_depth
    }
    
    combined_Xy_qid_pid = np.column_stack((cv_X, cv_qid_list, cv_pid_list, cv_y))
    np.random.shuffle(combined_Xy_qid_pid)
    num_features = cv_X.shape[1]
    
    cv_time = 3  
    split_data = np.array_split(combined_Xy_qid_pid, cv_time)
    
    train_X_part_one = split_data[0][:, :num_features]
    train_qid_part_one = split_data[0][:, num_features]
    train_pid_part_one = split_data[0][:, num_features + 1]
    train_y_part_one = split_data[0][:, -1]
    
    train_X_part_two = split_data[1][:, :num_features]
    train_qid_part_two = split_data[1][:, num_features]
    train_pid_part_two = split_data[1][:, num_features + 1]
    train_y_part_two = split_data[1][:, -1]
    
    train_X_part_three = split_data[2][:, :num_features]
    train_qid_part_three = split_data[2][:, num_features]
    train_pid_part_three = split_data[2][:, num_features + 1]
    train_y_part_three = split_data[2][:, -1]
    
    one_two_X = np.concatenate([train_X_part_one, train_X_part_two])
    two_three_X = np.concatenate([train_X_part_two, train_X_part_three])
    one_three_X = np.concatenate([train_X_part_one, train_X_part_three])
    
    ot_qid = np.concatenate([train_qid_part_one, train_qid_part_two])
    tt_qid = np.concatenate([train_qid_part_two, train_qid_part_three])
    oth_qid = np.concatenate([train_qid_part_one, train_qid_part_three])
    
    ot_pid = np.concatenate([train_pid_part_one, train_pid_part_two])
    tt_pid = np.concatenate([train_pid_part_two, train_pid_part_three])
    oth_pid = np.concatenate([train_pid_part_one, train_pid_part_three])
    
    ot_y = np.concatenate([train_y_part_one, train_y_part_two])
    tt_y = np.concatenate([train_y_part_two, train_y_part_three])
    oth_y = np.concatenate([train_y_part_one, train_y_part_three])
    
    one_dmatrix = xgb.DMatrix(train_X_part_one, label=train_y_part_one)
    two_dmatrix = xgb.DMatrix(train_X_part_two, label=train_y_part_two)
    three_dmatrix = xgb.DMatrix(train_X_part_three, label=train_y_part_three)
    
    ot_dmatrix = xgb.DMatrix(one_two_X, label=ot_y)
    tt_dmatrix = xgb.DMatrix(two_three_X, label=tt_y)
    oth_dmatrix = xgb.DMatrix(one_three_X, label=oth_y)

    AP_list = []
    NDCG_list = []
    
    # round 1
    xgb_model = xgb.train(params, tt_dmatrix, num_boost_round=4,
                        evals=[(one_dmatrix, 'validation')], early_stopping_rounds=2)

    
    pred = xgb_model.predict(one_dmatrix)

    qid_pid_prob_map = defaultdict(dict)
    for i in range(len(train_qid_part_one)):
        qid_pid_prob_map[train_qid_part_one[i]][train_pid_part_one[i]] = pred[i]
        
    sorted_qid_pid_prob_map = defaultdict(list) 
    for qid, pid_prob_map in qid_pid_prob_map.items():
        sorted_pids = sorted(pid_prob_map.items(), key=lambda x: x[1], reverse=True)
        sorted_qid_pid_prob_map[qid] = sorted_pids    
    
    
    LM_AP = cal_AP(sorted_qid_pid_prob_map, train_validation_rel_dict)
    LM_NDCG = cal_NDCG(sorted_qid_pid_prob_map, train_validation_rel_dict)
    AP_list.append(LM_AP)
    NDCG_list.append(LM_NDCG)
    
    # round 2
    xgb_model = xgb.train(params, oth_dmatrix, num_boost_round=4,
                        evals=[(two_dmatrix, 'validation')], early_stopping_rounds=2)

    
    pred = xgb_model.predict(two_dmatrix)

    qid_pid_prob_map = defaultdict(dict)
    for i in range(len(train_qid_part_two)):
        qid_pid_prob_map[train_qid_part_two[i]][train_pid_part_two[i]] = pred[i]
        
    sorted_qid_pid_prob_map = defaultdict(list) 
    for qid, pid_prob_map in qid_pid_prob_map.items():
        sorted_pids = sorted(pid_prob_map.items(), key=lambda x: x[1], reverse=True)
        sorted_qid_pid_prob_map[qid] = sorted_pids    
    
    LM_AP = cal_AP(sorted_qid_pid_prob_map, train_validation_rel_dict)
    LM_NDCG = cal_NDCG(sorted_qid_pid_prob_map, train_validation_rel_dict)
    AP_list.append(LM_AP)
    NDCG_list.append(LM_NDCG)
    
    # round 3
    xgb_model = xgb.train(params, oth_dmatrix, num_boost_round=4,
                        evals=[(three_dmatrix, 'validation')], early_stopping_rounds=2)

    
    pred = xgb_model.predict(three_dmatrix)

    qid_pid_prob_map = defaultdict(dict)
    for i in range(len(train_qid_part_three)):
        qid_pid_prob_map[train_qid_part_three[i]][train_pid_part_three[i]] = pred[i]
        
    sorted_qid_pid_prob_map = defaultdict(list) 
    for qid, pid_prob_map in qid_pid_prob_map.items():
        sorted_pids = sorted(pid_prob_map.items(), key=lambda x: x[1], reverse=True)
        sorted_qid_pid_prob_map[qid] = sorted_pids    
    
    LM_AP = cal_AP(sorted_qid_pid_prob_map, train_validation_rel_dict)
    LM_NDCG = cal_NDCG(sorted_qid_pid_prob_map, train_validation_rel_dict)
    AP_list.append(LM_AP)
    NDCG_list.append(LM_NDCG)
    
    mean_AP = np.mean(AP_list)
    mean_NDCG = np.mean(NDCG_list)
    # if mean_AP > best_AP:
    #     best_AP = mean_AP
    #     best_NDCG = mean_NDCG
    #     best_eta = eta
    #     best_gamma = gamma
    #     best_min_child_weight = min_child_weight
    #     best_max_depth = max_depth
    
    if mean_NDCG > best_NDCG:
        best_AP = mean_AP
        best_NDCG = mean_NDCG
        best_eta = eta
        best_gamma = gamma
        best_min_child_weight = min_child_weight
        best_max_depth = max_depth    

# LR_AP = {lr:AP}
LM_AP = best_AP
LM_NDCG = best_NDCG
print("BEST_LM_AP: ", LM_AP)
print("BEST_LM_NDCG: ", LM_NDCG)
print("BEST_eta: ", best_eta)
print("BEST_gamma: ", best_gamma)
print("BEST_min_child_weight: ", best_min_child_weight)
print("BEST_max_depth: ", best_max_depth)

# ============================================ LM.txt ============================================
test_qid_terms = {}
with open('test-queries.tsv', 'r', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')  
    for row in tsv_reader:
        qid = row[0]  
        terms = preprocessing(row[1])  
        test_qid_terms[qid] = terms

candidate_pid_terms = {}
qid_pid_connection = {}        
with open('candidate_passages_top1000.tsv', 'r', encoding='utf-8') as file:     
    tsv_reader = csv.reader(file, delimiter='\t') 
    for row in tsv_reader:
        qid = row[0]
        if row[0] in test_qid_terms:
            pid = row[1]
            terms = preprocessing(row[3])
            candidate_pid_terms[pid] = terms
            if qid not in qid_pid_connection:
                qid_pid_connection[qid] = []
            qid_pid_connection[qid].append(pid)        

test_qid_embeddings = {qid: get_average_embedding(train_model, terms) for qid, terms in test_qid_terms.items()}
test_pid_embeddings = {pid: get_average_embedding(train_model, terms) for pid, terms in candidate_pid_terms.items()}

def calculate_features_and_labels(qid_embeddings, pid_embeddings):
    rows = []

    for qid, pid_list in qid_pid_connection.items():
        for pid in pid_list:
            if qid in qid_embeddings and pid in pid_embeddings:
                qid_terms_embedding = qid_embeddings[qid]
                pid_terms_embedding = pid_embeddings[pid]
                # use embedding
                feature_vector = np.concatenate([qid_terms_embedding, pid_terms_embedding])
                # append qid, pid, feature_vector, and label to the rows list
                rows.append((qid, pid, feature_vector))

    # Create a DataFrame from the rows
    df = pd.DataFrame(rows, columns=['qid', 'pid', 'features'])

    return df

qid_pid_features_df = calculate_features_and_labels(test_qid_embeddings, test_pid_embeddings)

qid_pid_X = np.stack(qid_pid_features_df['features'].values)
test_qid_list = qid_pid_features_df['qid'].values
test_pid_list = qid_pid_features_df['pid'].values

mean = np.mean(qid_pid_X, axis=0)
std = np.std(qid_pid_X, axis=0)
std += 1e-10
qid_pid_X_normalized = (qid_pid_X - mean) / std

params = {
        # 'objective': 'rank:ndcg',
        # 'objective': 'rank:map',
        'objective': 'rank:pairwise',
        'eta': best_eta,
        'gamma': best_gamma,
        'min_child_weight': best_min_child_weight,
        'max_depth': best_max_depth
    }
xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4,
                    evals=[(valid_dmatrix, 'validation')], early_stopping_rounds=2)

test_dmatrix = xgb.DMatrix(qid_pid_X_normalized)
test_pred = xgb_model.predict(test_dmatrix)

test_qid_pid_prob_map = defaultdict(dict)
test_sorted_qid_pid_prob_map = defaultdict(dict)

for i in range(len(test_qid_list)):
    test_qid_pid_prob_map[test_qid_list[i]][test_pid_list[i]] = test_pred[i]
for qid, pid_prob_map in test_qid_pid_prob_map.items():
    sorted_pids = sorted(pid_prob_map.items(), key=lambda x: x[1], reverse=True)
    test_sorted_qid_pid_prob_map[qid] = dict(sorted_pids) 

top_100_pids_per_qid = {}
for qid, pid_prob_map in test_sorted_qid_pid_prob_map.items():
    top_100_pids = dict(itertools.islice(pid_prob_map.items(), 100))
    top_100_pids_per_qid[qid] = top_100_pids

top_100_pids_per_qid = OrderedDict(sorted(top_100_pids_per_qid.items(), key=lambda x: x[0]))
with open('LM.txt', 'w') as file:
    last_qid = 0
    for qid, pids in top_100_pids_per_qid.items():
        if qid != last_qid:
            rank = 1
        for pid, prob in pids.items():
            file.write(f'{qid} A2 {pid} {rank} {prob} LM\n')
            rank += 1    