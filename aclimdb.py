import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import re
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_WORD = 20000  # 只保留最高频的20000词
MAX_LEN = 200     # 句子统一长度为200
word_count={}     # 词-词出现的词数 词典

#清理文本，去标点符号，转小写,其实使用的是下面那个函数，这个没用
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()

def normalizeString(s):
    """字符串规范化函数, 参数s代表传入的字符串"""
    # 使字符变为小写并去除两侧空白符
    s = s.lower().strip()
    # 在.!?前加一个空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中不是大小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
# 分词方法
def tokenizer(sentence):
    return sentence.split()

#  数据预处理过程，返回词典
def data_process(filename): # 根据文本路径生成文本的标签

    print("data preprocess")
    # file_pro = open(text_path,'w',encoding='utf-8')

    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # lines_out = [[normalizeString(s) for s in l.split('\t') if len(s) > 10] for l in lines]
    lines_out=[]
    for line in lines:
        l=line.split('\t')
        lines_temp=[]
        lines_temp.append(normalizeString(l[0]))
        lines_temp.append(l[1])
        lines_out.append(lines_temp)

    for line in lines_out:
        tokens = tokenizer(line[0]) # 分词统计词数
        for token in tokens:
            if token in word_count.keys():
                word_count[token] = word_count[token] + 1
            else:
                word_count[token] = 1   #之前没出现过就设为1

    print("build vocabulary")

    vocab = {"<UNK>": 0, "<PAD>": 1}

    word_count_sort = sorted(word_count.items(), key=lambda item : item[1], reverse=True) # 对词进行排序，过滤低频词，只取前MAX_WORD个高频词
    word_number = 1
    for word in word_count_sort:
        if word[0] not in vocab.keys():
            vocab[word[0]] = len(vocab)
            word_number += 1
        if word_number > MAX_WORD:
            break
    return vocab

# 定义Dataset
class MyDataset(Dataset):
    def __init__(self, text_path):
        file = open(text_path, 'r', encoding='utf-8')
        # self.text_with_tag = file.readlines()  # 文本标签与内容
        lines = file.readlines()
        lines_out=[]
        for line in lines:
            l=line.split('\t')
            lines_temp=[]
            lines_temp.append(normalizeString(l[0]))
            lines_temp.append(l[1])
            lines_out.append(lines_temp)
        
        self.text_with_tag = lines_out
        file.close()

    def __getitem__(self, index): # 重写getitem
        line = self.text_with_tag[index] # 获取一个样本的标签和文本信息
        
        text = line[0]  # 文本信息
        label = int(line[1]) # 标签信息
        # print(text,label)

        return text, label

    def __len__(self):
        return len(self.text_with_tag)


# 根据vocab将句子转为定长MAX_LEN的tensor
def text_transform(sentence_list, vocab):
    sentence_index_list = []
    for sentence in sentence_list:
        sentence_idx = [vocab[token] if token in vocab.keys() else vocab['<UNK>'] for token in tokenizer(sentence)] # 句子分词转为id

        if len(sentence_idx) < MAX_LEN:
            for i in range(MAX_LEN-len(sentence_idx)): # 对长度不够的句子进行PAD填充
                sentence_idx.append(vocab['<PAD>'])

        sentence_idx = sentence_idx[:MAX_LEN] # 取前MAX_LEN长度
        sentence_index_list.append(sentence_idx)
        # length = len(sentence_index_list)
    return torch.LongTensor(sentence_index_list) # 将转为idx的词转为tensor


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size) # embedding层

        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False)
        self.decoder = nn.Linear(num_hiddens, 1)
        self.activation=nn.Sigmoid()

    def forward(self, inputs):
        # inputs的形状是（批量大小，词数），因此LSTM需要将序列长度（Seq_len）作为第一维，所以将输入转置后 再提取词特征
        embeddings = self.embedding(inputs) # .permute(1,0) permute(1,0)交换维度
        # LSTM只传入输入embeddings,因此只返回最后一层的隐藏层再各时间步的隐藏状态
        # outputs的形状是（词数，批量大小， 隐藏单元个数）
        outputs, _ = self.encoder(embeddings)
        # 连接初时间步和最终时间步的隐藏状态作为全连接层的输入。形状为(批量大小， 隐藏单元个数)
        # encoding = outputs[-1] # 取LSTM最后一层结果
        linear = self.decoder(outputs[-1]) # 线性层
        # print(outs.shape) [256,1]
        output = 10*self.activation(linear)

        return output

# 模型训练
def train(model, train_data, test_data, vocab, epoch=10):
    print('train model')
    model = model.to(device)
    loss_sigma = 0.0
    correct = 0.0
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3,weight_decay=0.0001)

    for epoch in tqdm(range(epoch)):
        model.train()
        avg_loss = 0  # 平均损失
        avg_acc = 0  # 平均准确率
        for idx, (text, label) in enumerate(tqdm(train_data)):

            train_x = text_transform(text, vocab).to(device)
            train_y = label.float().to(device)

            optimizer.zero_grad()
            pred = model(train_x.t()).reshape(250)  #250：batch_size大小，代码并没有很完善
            
            loss = criterion(pred, train_y)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += accuracy(pred, train_y)

        # 一个epoch结束后，计算平均loss和评平均acc
        avg_loss = avg_loss / len(train_data)
        avg_acc = avg_acc / len(train_data)

        test_acc = test(model=model, test_data=test_data, vocab=vocab)

        print("avg_loss:", avg_loss, " train_avg_acc:,", avg_acc ,"test_acc",test_acc)
        # 保存训练完成后的模型参数
        # torch.save(model.state_dict(), 'LSTM_IMDB_parameter.pkl')


# 模型测试
def test(model, test_data, vocab):
    print('test model')
    model = model.to(device)
    model.eval()
    avg_acc = 0
    for idx, (text, label) in enumerate(tqdm(test_data)):
        train_x = text_transform(text, vocab).to(device)
        train_y = label.to(device)
        pred = model(train_x.t()).reshape(250)
        # pred = train_y
        avg_acc += accuracy(pred, train_y)
    avg_acc = avg_acc / len(test_data)
    return avg_acc

# 计算预测准确性
def accuracy(y_pred, y_true):
    y_pred = y_pred.cpu().detach().numpy()

    y_true = y_true.cpu().numpy()

    acc=0
    for i in range(len(y_true)):
        if y_pred[i]>=y_true[i]-1 and y_pred[i]<=y_true[i]+1:
            acc+=1

    return acc / len(y_pred)

def main():

    train_path = r'./dataset/aclImdb-train.txt'  # 原训练集文件地址

    test_path = r'./dataset/aclImdb-test.txt'  # 预处理后的训练集文件地址

    # vocab = data_process(train_path) # 数据预处理
    # data_process(test_path, test_dir)
    # np.save('vocab.npy', vocab) # 词典保存为本地
    vocab = np.load('vocab.npy', allow_pickle=True).item()  # 加载本地已经存储的vocab

    # 构建MyDataset实例
    train_data = MyDataset(text_path=train_path)
    test_data = MyDataset(text_path=test_path)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=250, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=250, shuffle=False)

    # 生成模型
    model = LSTM(vocab=vocab, embed_size=300, num_hiddens=128, num_layers=2)  # 定义模型

    train(model=model, train_data=train_loader, test_data = test_loader, vocab=vocab, epoch=20)
    

    # 加载训练好的模型
    #model.load_state_dict(torch.load('LSTM_IMDB_parameter.pkl', map_location=torch.device('cpu')))

    # 测试结果
    acc = test(model=model, test_data=test_loader, vocab=vocab)
    print(acc)

if __name__ == '__main__':
    main()

