from torch import nn
import torch.nn.functional as F

def alignment_loss(shared_t, shared_a, shared_v):
    #对齐损失
    la_ta = (shared_t - shared_a).pow(2).sum(dim=1).mean()
    la_tv = (shared_t - shared_v).pow(2).sum(dim=1).mean()
    la_av = (shared_a - shared_v).pow(2).sum(dim=1).mean()
    return (la_ta + la_tv + la_av) / 3

def orth_loss(shared_t, private_t):
    # 正交损失
    dot = (shared_t * private_t).sum(dim=1)  # shape [batch_size]
    return dot.pow(2).mean()

def rec_loss(shared_t, private_t,origin):
    #重构损失
    rec = shared_t + private_t
    rec_l = F.mse_loss(rec, origin)
    return rec_l

class AutoEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(AutoEncoder, self).__init__()  
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.4),
        )
        
    def forward(self, x):
        return self.encoder(x)
class Decouple(nn.Module):
    def __init__(self, hidden_dim):
        super(Decouple, self).__init__()
        self.share_t = AutoEncoder(hidden_dim)
        self.share_v = AutoEncoder(hidden_dim)
        self.share_a = AutoEncoder(hidden_dim)
        self.private_t = AutoEncoder(hidden_dim)
        self.private_v = AutoEncoder(hidden_dim)
        self.private_a = AutoEncoder(hidden_dim)
        
    def forward(self, T, A, V):
        t_shared = self.share_t(T)
        v_shared = self.share_v(V)
        a_shared = self.share_a(A)
        t_private = self.private_t(T)
        v_private = self.private_v(V)
        a_private = self.private_a(A)
        return t_shared, a_shared, v_shared, t_private, a_private, v_private
    