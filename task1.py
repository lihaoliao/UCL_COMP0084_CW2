from collections import Counter, defaultdict
import json
import re
import csv
import numpy as np
import time

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
    with open('remove_stop_word_vocabulary.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(remove_text.keys()))
    return remove_text

def read_and_process_tsv(file_path):
    data = []
    unique_data = {}
    qid_and_pid_and_rel = defaultdict(lambda: defaultdict(float))
    
    with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  
        for row in reader:
            data.append(row)
            qid, pid, relevancy = row[0], row[1], row[4]
            if pid not in qid_and_pid_and_rel[qid]:
                qid_and_pid_and_rel[qid][pid] = relevancy
            if pid not in unique_data: 
                unique_data[pid] = row
    
    return list(unique_data.values()), qid_and_pid_and_rel,data

unique_validation_data, qid_and_pid_and_rel, validation_data = read_and_process_tsv('validation_data.tsv')

#  get the passage info
def get_passage_info(data):
    passage_list = [row[3] for row in data]
    passage = " ".join(passage_list)
    return passage
passage_info = get_passage_info(unique_validation_data)

# preprocessing the passage info and return the terms
remove_number_of_terms = preprocessing(passage_info)

# create the connection between query and terms
def preprocessing_passage(passage):
    passage = passage.strip().lower()
    passage = passage.replace("/", " ")
    passage = re.sub(preprocessing_re, ' ', passage)
    return passage

queries_id_and_terms_info = {}
for row in validation_data:
    row[2] = preprocessing_passage(row[2])
    qid, query = row[0], row[2]
    tokens = query.split()
    query_contain_token = [token for token in tokens if token in remove_number_of_terms]
    if query_contain_token:
        queries_id_and_terms_info[qid] = query_contain_token
    else:
        # no quert contain token found, optimise the query
        query_contain_token = [token for token in tokens if token not in stop_words]
        queries_id_and_terms_info[qid] = query_contain_token    
            
# create the dict about passage and its terms after preprocessing
passages_id_and_terms_info = {
    row[1]: [token for token in preprocessing_passage(row[3]).split() if token in remove_number_of_terms]
    for row in unique_validation_data
}        

# create the passage inverted_index store the pid and term count
inverted_index = defaultdict(dict)
for pid, terms in passages_id_and_terms_info.items():
    term_frequency = Counter(terms)
    for term, frequency in term_frequency.items():
        inverted_index[term][pid] = frequency

# with open('inverted_index.json', 'w', encoding='utf-8') as file:
#     json.dump(inverted_index, file, ensure_ascii=False, indent=3)

# with open('passages_id_and_terms_info.json', 'w', encoding='utf-8') as file:
#     json.dump(passages_id_and_terms_info, file, ensure_ascii=False, indent=3)      

# create the inverted_index_query
inverted_index_query = defaultdict(dict)   
# each_query_terms_sum = {}     

for qid, terms in queries_id_and_terms_info.items():
    term_frequency = Counter(terms)
    # each_query_terms_sum[qid] = sum(term_frequency.values())
    for term, frequency in term_frequency.items():
        inverted_index_query[term][qid] = frequency  
        
# ===================================================BM25==========================================================
k1, k2, b = 1.2, 100, 0.75
dl = passages_id_and_terms_info
avdl = 0
for total_words_in_passage in dl.values():
    avdl += len(total_words_in_passage)
avdl = avdl / len(dl)

def calculate_BM25(n, f, qf, R, r, N, dl, avdl, k1, k2, b, pid):
    K = k1 * ((1 - b) + (b * (len(dl[pid])/avdl)))
    BM25_part1 = np.log( ((r+0.5) / (R-r+0.5)) / ((n-r+0.5) / (N-n-R+r+0.5)) )
    BM25_part2 = ((k1+1) * f) / (K + f)
    BM25_part3 = ((k2 + 1) * qf) / (k2 + qf)
    return BM25_part1 * BM25_part2 * BM25_part3

def calculate_BM25_score(qid_and_pid_rel, queries_id_and_terms_info, inverted_index, inverted_index_query, dl, avdl, k1, k2, b):
    BM25_score = defaultdict(dict)
    for qid, pids in qid_and_pid_rel.items():
        for pid in pids:
            BM25_cur_qid_score = 0
            for term in queries_id_and_terms_info[qid]:
                BM25_cur_qid_score += calculate_BM25(len(inverted_index[term]), inverted_index[term].get(pid, 0), inverted_index_query[term][qid], 0, 0, len(dl), dl, avdl, k1, k2, b, pid)
            if pid not in BM25_score[qid]:
                BM25_score[qid][pid] = {
                    'BM25_score': BM25_cur_qid_score,
                    'relevancy': qid_and_pid_and_rel[qid][pid]
                }
    return BM25_score

BM25_score = calculate_BM25_score(qid_and_pid_and_rel, queries_id_and_terms_info, inverted_index, inverted_index_query, dl, avdl, k1, k2, b)

sorted_BM25_by_score = {}
for qid, pids in BM25_score.items():
    sorted_pids = sorted(pids.items(), key=lambda x: x[1]['BM25_score'], reverse=True)
    sorted_BM25_by_score[qid] = {pid: info for pid, info in sorted_pids}

def cal_AP(sorted_BM25_by_score, qid_and_pid_and_rel):
    total_AP_for_each_query = 0
    for qid,pids in qid_and_pid_and_rel.items():
        counted_passage = 0
        found_rel_passage = 0
        cur_query_precision = 0
        for info in sorted_BM25_by_score[qid].values():
            counted_passage += 1
            if float(info['relevancy']) == 1:
                found_rel_passage += 1
                cur_query_precision += found_rel_passage / counted_passage
        if(found_rel_passage != 0):
            total_AP_for_each_query += cur_query_precision / found_rel_passage             
    return total_AP_for_each_query / len(qid_and_pid_and_rel)           
        
BM25_AP = cal_AP(sorted_BM25_by_score, qid_and_pid_and_rel)

def cal_NDCG(sorted_BM25_by_score, qid_and_pid_and_rel):
    total_NDCG_for_each_query = 0
    for qid, pids in qid_and_pid_and_rel.items():
        counted_passage = 0
        counted_passage_opt = 0
        rel_count = 0
        DCG = 0
        opt_DCG = 0
        for info in sorted_BM25_by_score[qid].values():
            counted_passage += 1
            reli = 0
            if float(info['relevancy']) == 1:
                rel_count += 1
                reli = 1
            DCG += (2 ** reli - 1) / np.log2(counted_passage + 1)   
        total_passage_count = len(sorted_BM25_by_score[qid])
        # create an array set rel_count length as 1 and rest as 0
        optimal_relevance = [1 if i < rel_count else 0 for i in range(total_passage_count)]    
        for info in optimal_relevance:
            counted_passage_opt += 1
            opt_DCG += (2 ** info - 1) / np.log2(counted_passage_opt + 1)    
             
        NDCG = DCG / opt_DCG if opt_DCG != 0 else 0
        total_NDCG_for_each_query += NDCG    

    return total_NDCG_for_each_query / len(qid_and_pid_and_rel)

BM25_NDCG = cal_NDCG(sorted_BM25_by_score, qid_and_pid_and_rel)
end_time = time.time()  
print("BM25_AP: ", BM25_AP)
print("BM25_NDCG: ", BM25_NDCG)
print("程序运行时间：" + str(end_time - start_time))
