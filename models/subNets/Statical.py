import torch
import torch.nn as nn
import torch.nn.functional as F



def div_metric(pos, neg, mode="kl"):
    if mode=="kl":
        return torch.mean(pos) - torch.log(torch.mean(torch.exp(neg)))
    
    elif mode=="f":
        return torch.mean(pos) - torch.log(torch.mean(torch.exp(neg))-1)
    elif mode=="w":
        return torch.mean(pos) - torch.log(torch.sum(neg) + 1e-8).subtract(torch.log(torch.tensor(neg.size(0), dtype=torch.float32)))  # 负样本的数量
    else:
        raise ValueError("Unknown mode")
        


def shuffle_batch(x):
    batch_size = x.size(0)
    perm_idx = torch.randperm(batch_size)
    x_shuffled = x[perm_idx]
    return x_shuffled


def create_fake_samples(x_text, x_audio, x_video):
    """
    构造伪联合样本:
      - 分别对 x_text、x_audio、x_video 做随机打乱
      - 再拼接成一个“伪联合”向量, 维度保持在 [batch_size, dim * 3]

    x_text, x_audio, x_video: 均为 [batch_size, dim] 形状
    返回值: z_fake, 形状为 [batch_size, 3 * dim]
    """
    xt_shuf = shuffle_batch(x_text)
    xa_shuf = shuffle_batch(x_audio)
    xv_shuf = shuffle_batch(x_video)

    # z_fake = torch.cat([xt_shuf, xa_shuf, xv_shuf], dim=-1)
    return xt_shuf,xa_shuf,xv_shuf

class StatisticNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(StatisticNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),

            nn.Linear(input_dim // 2, input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            nn.Linear(input_dim, input_dim ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            nn.Linear(input_dim, input_dim // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            nn.Linear(input_dim // 4, input_dim // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            
            nn.Linear(input_dim // 4, 1),
        )

    def forward(self, x):
        scores = self.mlp(x)
        scores = nn.functional.sigmoid(scores)  
        # assert positive
        # assert torch.all(scores >= 0), "Scores should be non-negative"
        return scores
