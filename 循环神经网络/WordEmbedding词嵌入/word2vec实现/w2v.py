import torch
from torch import nn


# 连续词袋
class Word2Vec_CBOW(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=100):
        super().__init__()
        # 嵌入层
        # num_embeddings: 词库字数
        # embedding_dim: 嵌入维度
        # padding_idx: 填充索引，若指定，则此索引处的权重不会被更新
        # max_norm: 词向量的最大长度，指的是词嵌入后的向量长度
        # norm_type: 归一化类型，默认为2
        # scale_grad_by_freq: 根据词频缩放梯度，词频越大梯度越小，反之越大，用于平衡词频对训练的影响
        # sparse: 是否使用稀疏梯度，默认为False
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 线性层
        self.fc = nn.Linear(embedding_dim, num_embeddings)
        self.softmax = nn.Softmax(dim=-1)

    # x 形状 (N, L=4) x 代表要嵌入的文字索引
    def forward(self, x):
        # embedding 输出的结果叫 词向量 词嵌入
        x = self.embedding(x)
        # (N, L, embedding_dim)
        # 加法，将所有词向量相加
        # x = torch.sum(x, dim=1)
        # 除了求和以外，也可以求平均嵌入
        x = torch.mean(x, dim=1)
        # (N, embedding_dim)
        y = self.fc(x)
        # (N, num_embeddings)
        # softmax 求概率分布
        y = self.softmax(y)
        return y


# 跳元模型
class Word2Vec_SkipGram(nn.Module):
    # context_len: 上下文长度
    def __init__(self, num_embeddings, embedding_dim=100, context_len=4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.context_len = context_len
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc_list = nn.ModuleList([nn.Linear(embedding_dim, num_embeddings) for i in range(context_len)])
        self.softmax = nn.Softmax(dim=-1)

    # x 形状 (N)
    def forward(self, x):
        N = x.shape[0]

        # 嵌入层
        x = self.embedding(x)
        # (N, embedding_dim)

        # 声明一个空的输出张量
        outputs = torch.empty(N, self.context_len, self.num_embeddings)

        # 线性变换
        for i, fc in enumerate(self.fc_list):
            # 预测一个字
            y = fc(x)
            y = self.softmax(y)
            # (N, num_embeddings)
            outputs[:, i] = y

        return outputs


if __name__ == '__main__':
    all_text = 'i love you 我 爱 你 们'
    words = all_text.split()
    # model = Word2Vec_CBOW(len(words), embedding_dim=100)
    # x = torch.tensor([
    #     [0, 2],
    #     [3, 5]
    # ])
    # y = model(x)
    # print(y)
    # print(y.shape)
    model = Word2Vec_SkipGram(len(words), context_len=4)
    x = torch.tensor([1, 4])
    y = model(x)
    print(y)
    print(y.shape)
    # 训练完成后，调用 model.embedding() 就可以得到词向量了
    word_vector = model.embedding(x)
