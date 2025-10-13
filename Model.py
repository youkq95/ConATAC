# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstConvLayer(nn.Module):
    def __init__(self, first_conv_out_channels, first_kernel_size, dropout_rate=0.3):
        super(FirstConvLayer, self).__init__()
        # 定义二维卷积层 (1 表示虚拟的单通道)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=first_conv_out_channels, kernel_size=first_kernel_size)
        self.gelu = nn.GELU()  # 激活函数
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass for the convolutional layer.
        x: 输入形状为 (batch_size, seq_length, in_channels) -> (batch_size, 19, 34)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, 19, 34)使输入形状变为 (batch_size, 1, seq_length, in_channels)
        x = self.conv2d(x)  # 卷积后形状为 (batch_size, out_channels=100, new_height=18, new_width=1)
        x = self.gelu(x)  # 应用激活函数
        x = self.dropout(x)
        x = x.squeeze(-1)  # (batch_size, 32, 18)变为 (batch_size, out_channels, new_height)

        return x


class SecondConvLayer(nn.Module):
    def __init__(self, second_conv_out_channels, second_kernel_size, dropout_rate=0.3):
        super(SecondConvLayer, self).__init__()
        # 定义二维卷积层
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=second_conv_out_channels, kernel_size=second_kernel_size)
        self.gelu = nn.GELU()  # 激活函数
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        x: 输入形状为 (batch_size, channels, seq_length)
        """
        # 先转置，使特征通道在第二个维度
        x = x.permute(0, 2, 1)  # 变为 (batch_size, seq_length, channels)
        # 增加一个维度，转换为2D卷积需要的输入形状
        x = x.unsqueeze(1)  # 变为 (batch_size, 1, channels, seq_length)
        x = self.conv2d(x)  # 2D卷积
        x = self.gelu(x)  # 应用激活函数
        x = self.dropout(x)
        # 移除多余的维度
        x = x.squeeze(-1)  # 移除最后一个维度

        return x


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Attention Layer:
        input_dim: 输入特征数量（即通道数，如上一个卷积层的输出通道数）
        output_dim: 最终输出特征数量（即 V 的行数）
        """
        super(AttentionLayer, self).__init__()
        self.output_dim = output_dim
        # 定义用于生成 Q, K, V 的线性变换
        self.query_layer = nn.Linear(input_dim, output_dim)  # Q 的线性变换
        self.key_layer = nn.Linear(input_dim, output_dim)  # K 的线性变换
        self.value_layer = nn.Linear(input_dim, output_dim)  # V 的线性变换

    def forward(self, x):
        """
        Forward pass for the attention mechanism
        x: 输入张量，形状为 (batch_size, channels, seq_length)，即 (batch_size, 32, 18)
        """
        # 转置 x 为 (batch_size, seq_length, channels) -> 常用于线性层
        x = x.transpose(1, 2)  # 转置后形状为 (batch_size, 18, 32)

        # 生成 Q, K, V
        Q = self.query_layer(x)  # Q 的形状为 (batch_size, 18, output_dim)
        K = self.key_layer(x)  # K 的形状为 (batch_size, 18, output_dim)
        V = self.value_layer(x)  # V 的形状为 (batch_size, 18, output_dim)

        # 转置 K 为 (batch_size, output_dim, seq_length) 以便计算 Q·K^T
        K_T = K.transpose(1, 2)  # K_T 形状为 (batch_size, output_dim, 18)

        # Scaled Dot-Product Attention
        # 计算注意力分数：Q·K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K_T)  # (batch_size, 18, 18)
        d_k = Q.size(-1)  # d_k 为 Q 或 K 的最后一维大小
        scaled_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # 缩放

        # 对每一行（时间步）进行 softmax，得到注意力权重
        attention_weights = F.softmax(scaled_scores, dim=-1)  # (batch_size, 18, 18)

        # 注意力权重加权 V
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, 18, output_dim)

        return attention_output  # 输出形状为 (batch_size, 18, output_dim)


# 联合卷积层和 Attention 层
class ConvAndAttentionModel(nn.Module):
    def __init__(self,
                 first_conv_out_channels,
                 first_kernel_size,
                 second_conv_out_channels,
                 second_kernel_size,
                 attention_output_dim,
                 maxpool_kernel_size):
        super(ConvAndAttentionModel, self).__init__()
        self.conv_layer = FirstConvLayer(first_conv_out_channels=first_conv_out_channels, first_kernel_size=first_kernel_size)
        self.second_conv_layer = SecondConvLayer(second_conv_out_channels=second_conv_out_channels, second_kernel_size=second_kernel_size)
        self.attention_layer = AttentionLayer(input_dim=second_conv_out_channels, output_dim=attention_output_dim)
        self.pool1d = nn.MaxPool1d(kernel_size=maxpool_kernel_size)  # 使用 kernel_size=16进行池化

        # 全连接层
        self.fc1 = nn.Linear(17 * 1, 32)  # 输入维度是池化后展平的特征 output_dim，映射到 32
        self.fc2 = nn.Linear(32, 2)  # 最后的全连接层输出 2（对应二分类）


    def forward(self, x):
        # 输入数据 -> 卷积层
        x = self.conv_layer(x)  # 2D 卷积层输出形状为 (batch_size, conv_out_channels=64, seq_length=18)
        x = self.second_conv_layer(x)
        # 注意力机制 -> 输出最终的注意力结果
        x = self.attention_layer(x)  # 输出每个时间步的注意力加权结果
        # 每个时间步的 16 个特征中取一个最大值
        x = self.pool1d(x)  # 参数 kernel_size=16 -> (batch_size, 1, 18)
        x = x.permute(0, 2, 1)  # 恢复到 (batch_size, 18, 1)

        x = x.flatten(start_dim=1)  # 展平成 (batch_size, 18*1)
        # 全连接层
        x = F.gelu(self.fc1(x))  # 第一层全连接 + GELU 激活函数
        x = self.fc2(x)  # 第二层全连接，输出 (batch_size, 2)
        return x


# if __name__ == "__main__":
#     batch_size = 32
#     seq_length = 19
#     in_channels = 23
#     first_conv_out_channels = 32
#     first_kernel_size = (2, 23)
#     second_conv_out_channels = 64
#     second_kernel_size = (2, 32)
#     attention_output_dim = 16  # V 的行数
#     maxpool_kernel_size = 16
#     # 输入数据：形状为 (batch_size, seq_length, in_channels)
#     input_data = torch.randn(batch_size, seq_length, in_channels)
#     # 定义联合模型
#     model = ConvAndAttentionModel(
#                                   first_conv_out_channels=first_conv_out_channels,
#                                   first_kernel_size=first_kernel_size,
#                                   second_conv_out_channels=second_conv_out_channels,
#                                   second_kernel_size=second_kernel_size,
#                                   attention_output_dim=attention_output_dim,
#                                   maxpool_kernel_size=maxpool_kernel_size)
#     # 前向传播
#     output = model(input_data)
#     print("Input shape: ", input_data.shape)  # (batch_size, 19, 23)
#     print("Output shape: ", output.shape)  # (batch_size, 2)
#     print(model)