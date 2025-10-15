# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstConvLayer(nn.Module):
    def __init__(self, first_conv_out_channels, first_kernel_size, dropout_rate=0.3):
        super(FirstConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=first_conv_out_channels, kernel_size=first_kernel_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.conv2d(x)  
        x = self.gelu(x)  
        x = self.dropout(x)
        x = x.squeeze(-1)  

        return x


class SecondConvLayer(nn.Module):
    def __init__(self, second_conv_out_channels, second_kernel_size, dropout_rate=0.3):
        super(SecondConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=second_conv_out_channels, kernel_size=second_kernel_size)
        self.gelu = nn.GELU() 
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.conv2d(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = x.squeeze(-1)

        return x


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        K_T = K.transpose(1, 2)
        attention_scores = torch.matmul(Q, K_T)
        d_k = Q.size(-1)
        scaled_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scaled_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output

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
        self.pool1d = nn.MaxPool1d(kernel_size=maxpool_kernel_size)

        self.fc1 = nn.Linear(17 * 1, 32)
        self.fc2 = nn.Linear(32, 2)


    def forward(self, x):
        x = self.conv_layer(x)
        x = self.second_conv_layer(x)
        x = self.attention_layer(x)
        x = self.pool1d(x)
        x = x.permute(0, 2, 1)

        x = x.flatten(start_dim=1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x