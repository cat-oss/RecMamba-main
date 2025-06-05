import torch
import torch.nn.functional as F

class ChannelWeighting(torch.nn.Module):
    def __init__(self):
        super(ChannelWeighting, self).__init__()

    def forward(self, x):
        # 计算平均池化和最大池化，得到每个通道的权重
        avg_pool = F.adaptive_avg_pool1d(x, 1)  # (B, D, L) -> (B, D, 1)
        max_pool, _ = torch.max(x, dim=2, keepdim=True)  # (B, D, L) -> (B, D, 1)

        # 将平均池化和最大池化结果拼接，得到通道的权重
        channel_weights = torch.cat([avg_pool, max_pool], dim=2)  # (B, D, 2)
        channel_weights = torch.mean(channel_weights, dim=2)  # 对拼接后的维度求均值，得到(B, D)

        # 对通道权重进行归一化处理
        channel_weights = F.softmax(channel_weights, dim=1)  # 归一化通道权重，使其和为1
        return channel_weights.unsqueeze(2)

class ChannelFusion(torch.nn.Module):
    def __init__(self, in_channels):
        super(ChannelFusion, self).__init__()
        # 使用卷积层进行通道融合，kernel_size设置为1，表示对每个位置的通道进行卷积
        # 通过卷积学习通道之间的关系
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)  # 通过卷积融合通道信息
        return x