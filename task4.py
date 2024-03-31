import random
import pandas as pd
import numpy as np
import torch
import torchtext
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torchtext.vocab as vocab

start_time = time.time() 
def end():
    end_time = time.time()  
    print("程序运行时间5：" + str(end_time - start_time))
    
stop_words = set(['did', 'a', "mightn't", 'these', 'to', 'just', 'his', 'into', 'but', 't', 'mustn', 'other', "won't", 'nor', 'himself', 'mightn', 'by', 've', 'very', 'so', "doesn't", 'which', 'off', 'an', 'with', 'at', 'below', 'your', 'shouldn', 'it', 'are', "needn't", 'hasn', 'that', 'me', 'more', 'no', 'do', 'herself', 'this', 'there', 'under', 'o', 'both', 'some', 'hers', 'over', 'between', 'them', 'been', 'because', 'myself', "don't", 'd', "didn't", 'only', 'on', 'how', 'am', 'who', 'their', "hasn't", 's', 'ours', 'you', "you'd", 'above', 'few', 'was', 'our', 'can', "hadn't", 'shan', 'now', 'once', 'being', 'hadn', 'were', 'whom', "isn't", 'ain', 'will', 'for', 'yours', "that'll", 'should', 'haven', 'those', 'couldn', 'while', 'same', 'themselves', 'itself', 'having', 'where', 'when', 'they', 'had', 'he', 'any', 'the', "should've", 'after', 'or', 'wasn', 'won', 'has', 'does', 'not', "shouldn't", 'than', 're', 'own', "mustn't", "it's", 'have', 'why', 'is', 'and', 'about', 'him', 'doing', 'theirs', 'wouldn', 'll', 'my', 'in', 'of', 'aren', 'needn', 'from', 'up', 'then', "she's", 'ma', "haven't", "you'll", "wasn't", 'y', 'against', 'here', 'further', "you're", 'yourself', 'down', 'before', 'such', 'until', 'isn', 'each', 'its', 'if', 'all', "you've", 'her', 'didn', 'doesn', 'what', "weren't", 'weren', "wouldn't", 'she', 'too', "aren't", 'most', "couldn't", 'i', 'during', 'ourselves', 'through', 'we', 'm', 'as', "shan't", 'out', 'yourselves', 'be', 'don', 'again'])

class TextDataset(Dataset):
    def __init__(self, dataframe, glove_vectors, tokenizer, max_length):
        self.dataframe = dataframe
        self.glove_vectors = glove_vectors
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qid_pid_rel_map, self.num_relevancy_ones = self.build_qid_pid_rel_map()
        self.sample_data = self.sample_data(proportions = 1)

    def build_qid_pid_rel_map(self):
        qid_pid_rel_map = {}
        num_relevancy_ones = 0  
        for _, row in self.dataframe.iterrows():
            qid = row['qid']
            pid = row['pid']
            relevancy = row['relevancy']
            if relevancy == 1:
                num_relevancy_ones += 1
            if qid not in qid_pid_rel_map:
                qid_pid_rel_map[qid] = {}
            qid_pid_rel_map[qid][pid] = relevancy
        # print(num_relevancy_ones)    
        return qid_pid_rel_map, num_relevancy_ones
    
    def sample_data(self, proportions):
        # 抽取每个qid中rel为1的数据和随机一条rel为0的数据，并形成{qid: {pid: rel}}结构
        sample_data = {}
        for qid, pid_rel_map in self.qid_pid_rel_map.items():
            rel_ones = {pid: rel for pid, rel in pid_rel_map.items() if rel == 1.0}
            rel_zeros = {pid: rel for pid, rel in pid_rel_map.items() if rel == 0.0}
        
            if rel_ones:
                sample_data[qid] = {**rel_ones}  # 将rel为1的数据加入样本

            if rel_zeros:
                zero_sample = random.sample(list(rel_zeros.items()), 1)  # 随机选择一条rel为0的数据
                sample_data[qid] = {**sample_data.get(qid, {}), **dict(zero_sample)}

        # 额外选择rel为0的数据以满足所需比例
        total_data_needed = int(self.num_relevancy_ones * (proportions + 1)) - sum(len(data_value) for data_value in sample_data.values())
        if total_data_needed > 0:
            all_zeros = [(qid, pid, rel) for qid, pid_rel_map in self.qid_pid_rel_map.items() for pid, rel in pid_rel_map.items() if rel == 0.0]
            extra_zeros = random.sample(all_zeros, total_data_needed)
            for qid, pid, rel in extra_zeros:
                if qid in sample_data:
                    sample_data[qid][pid] = rel
                else:
                    sample_data[qid] = {pid: rel}
        # print('sample data', len(sample_data))
        return sample_data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        query_vec = self.text_to_embedding(row['queries'])
        passage_vec = self.text_to_embedding(row['passage'])
        combine_vec  = self.text_to_embedding(row['queries'] + row['passage'])
        # label
        relevancy = row['relevancy']
        qid = row['qid']  
        pid = row['pid'] 
        return torch.tensor(query_vec).unsqueeze(dim=0), torch.tensor(passage_vec).unsqueeze(dim=0), torch.tensor(combine_vec).unsqueeze(dim=0), torch.tensor(relevancy, dtype=torch.float), qid, pid

    def text_to_embedding(self, text):
        tokens = self.tokenizer(text)
        tokens = set(word for word in tokens if word not in stop_words)
        embeddings = []
        # 遍历tokens中的每个单词
        for t in tokens:
        # 检查单词是否在GloVe词汇表中
            if t in self.glove_vectors.stoi:
            # 获取GloVe向量并转换为NumPy数组
                vector = self.glove_vectors[t].numpy()
                # 将向量添加到列表中
                embeddings.append(vector)
                
        # 如果生成的词向量列表超出了最大长度，进行截断
        if len(embeddings) > self.max_length:
            embeddings = embeddings[:self.max_length]
        if len(embeddings) < self.max_length:
        # 添加(padding)向量直到达到max_length
            embeddings += [np.zeros(self.glove_vectors.dim) for _ in range(self.max_length - len(embeddings))]    
        if len(embeddings) == 0:
            return np.zeros((self.max_length, self.glove_vectors.dim))
        embeddings = np.stack(embeddings)
        return embeddings

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

embedding_dim = 100
iteration = 1
   
# class_num = 2
lr = 0.001     
out_channel = 2
max_length = 40
glove = vocab.GloVe(name='6B', dim=embedding_dim)

def read_data(file_path, num_rows=None):
    return pd.read_csv(file_path, sep='\t', nrows=num_rows)


train_data_path = 'train_data.tsv'
validation_data_path = 'validation_data.tsv'
# train_df = read_data('train_data.tsv')
# validation_df = read_data('validation_data.tsv')
train_df = read_data('train_data.tsv')
validation_df = read_data('validation_data.tsv')
class_num = train_df['relevancy'].nunique()    

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def dict_to_df(data_dict, original_df):
    data_list = []
    for qid, pid_rel_map in data_dict.items():
        for pid, rel in pid_rel_map.items():
            data_list.append((qid, pid, rel))

    # 使用列表创建DataFrame
    filtered_df = pd.DataFrame(data_list, columns=['qid', 'pid', 'relevancy'])

    # 显示DataFrame

    # 如果你还需要query和passage列，你可能需要根据qid和pid从原始数据中获取这些信息
    # 假设original_df是你的原始DataFrame，包含所有的数据
    # 你可以如下合并它们以获取完整的信息
    complete_df = pd.merge(filtered_df, original_df, on=['qid', 'pid', 'relevancy'], how='left')
    complete_df = complete_df[['qid', 'pid', 'queries', 'passage', 'relevancy']]
    # print(len(complete_df))
    return complete_df

# dataset and dataloader
train_dataset = TextDataset(train_df, glove, tokenizer,max_length)
sample_train_dateset = dict_to_df(train_dataset.sample_data, train_dataset.dataframe)
sample_train_dateset = TextDataset(sample_train_dateset, glove, tokenizer, max_length)
train_loader = DataLoader(sample_train_dateset, batch_size=1, shuffle=True)

validation_dataset = TextDataset(validation_df, glove, tokenizer,max_length)
validation_loader = DataLoader(validation_dataset, batch_size=1)

# end()

# for i in range(1):
#     for query, passage, combine, relevancy in train_loader:
#         print('query:', query.shape, 'passage:', passage.shape, 'combine:', combine.shape, 'relevancy:', relevancy.shape)
#         break
         
class Block(nn.Module):
    def __init__(self,kernel_size, embedding_dim, out_channel, max_length):
        super().__init__()
        # CNN, batch_size, in_channel, max_length, embedding_dim, 输出卷成一个条
        # 输出通道可以优化
        self.cnn = nn.Conv2d(in_channels=1, out_channels=out_channel, kernel_size=(kernel_size, embedding_dim))
        # ACT
        self.act = nn.ReLU()
        # MaxPool
        self.maxpool = nn.MaxPool1d(kernel_size=(max_length - kernel_size + 1))
    def forward(self, combine_vec):
        # print('combine_vec', combine_vec.shape)
        c = self.cnn.forward(combine_vec)
        # print('c', c.shape)
        a = self.act.forward(c)
        # print('a', a.shape)
        a = a.squeeze(dim=-1)
        # print('a', a.shape)
        m = self.maxpool.forward(a)
        # print('m', m.shape)
        m = m.squeeze(dim=-1)
        # print('m', m.shape)
        return m  
      
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, class_num,out_channel,max_length):
        super().__init__()
        # 卷积核可以优化
        self.block1 = Block(2, embedding_dim,out_channel,max_length)
        self.block2 = Block(3, embedding_dim,out_channel,max_length)
        self.block3 = Block(4, embedding_dim,out_channel,max_length)
        
        self.classifier = nn.Linear(out_channel * 3, class_num)   
        self.loss_fun = nn.CrossEntropyLoss()   
        
    def forward(self, combine_vec, label=None):
        b1_result = self.block1.forward(combine_vec)
        b2_result = self.block2.forward(combine_vec)
        b3_result = self.block3.forward(combine_vec)
        
        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)
        # print('feature', feature.shape)
        prob = self.classifier(feature)
        
        # 
        if label is not None:
            label = label.long()
            loss = self.loss_fun(prob,label)
            return loss
        else:
            return torch.argmax(prob, dim=-1), prob
           
model = TextCNN(embedding_dim, class_num,out_channel,max_length).to(device)
opt = torch.optim.Adam(model.parameters(), lr)
       
# initial_state = model.state_dict()
                  
for i in range(iteration):      
    # train mode but not work in this case
    model.train()
    for query, passage, combine, relevancy, qid, pid in train_loader:
        combine = combine.float()
        combine = combine.to(device)
        relevancy = relevancy.to(device) 
        opt.zero_grad()
        loss = model.forward(combine, relevancy)
        loss.backward()
        opt.step()
        # print('loss', loss)
    # 假设你想监控的是名为'block1.cnn'的第一个卷积层
    # cnn_weights = model.block1.cnn.weight.data.cpu().numpy()
    # cnn_biases = model.block1.cnn.bias.data.cpu().numpy()

    # 计算权重和偏置的平均值
    # avg_weight = cnn_weights.mean()
    # avg_bias = cnn_biases.mean()

    # print(f'Epoch {i+1}, Loss: {loss.item()}')
    # print(f'Average CNN Weight: {avg_weight}')
    # print(f'Average CNN Bias: {avg_bias}')   
    
# train_state = model.state_dict()
        
model.eval()  
qid_pid_prob_map = defaultdict(dict)
with torch.no_grad():
    for query, passage, combine, relevancy, qid, pid in validation_loader:
        combine = combine.float()
        combine = combine.to(device)
        relevancy = relevancy.to(device) 
        prob_rel, prob = model.forward(combine)
        prob = torch.softmax(prob, dim=1)
        rel_prob = prob[0][1].item()
        # prob[0][0] is irrelevant, prob[0][1] is relevant
        qid_pid_prob_map[qid.item()][pid.item()] = rel_prob
  
sorted_qid_pid_prob_map = {}
for qid, pid_prob_map in qid_pid_prob_map.items():
    # 按prob进行降序排序，并存储结果
    sorted_pids = sorted(pid_prob_map.items(), key=lambda x: x[1], reverse=True)
    sorted_qid_pid_prob_map[qid] = sorted_pids

end()    
   
        
        
        
