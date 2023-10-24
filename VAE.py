import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器部分
        self.fc1 = nn.Linear(input_dim, 400)  # 输入维度到编码器中间层
        self.fc21 = nn.Linear(400, latent_dim)  # 输出均值
        self.fc22 = nn.Linear(400, latent_dim)  # 输出对数方差
        # 解码器部分
        self.fc3 = nn.Linear(latent_dim, 400)  # 潜在向量到解码器中间层
        self.fc4 = nn.Linear(400, input_dim)  # 解码器输出

    def encode(self, x):
        # 编码器：将输入 x 映射到潜在空间
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # 返回均值和对数方差

    def reparameterize(self, mu, logvar):
        # 从潜在空间采样并返回潜在向量
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 解码器：将潜在向量 z 映射回原始数据空间
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 返回解码结果

    def forward(self, x):
        # 前向传播过程，将输入数据编码为潜在向量，然后解码生成输出
        mu, logvar = self.encode(x.view(-1, 784))  # 对输入数据编码
        z = self.reparameterize(mu, logvar)  # 从均值和对数方差中采样
        return self.decode(z), mu, logvar  # 返回解码后的数据、均值和对数方差

