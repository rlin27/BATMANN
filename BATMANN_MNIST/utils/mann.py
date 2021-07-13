import torch.nn.functional as F
import torch
import os
import shutil


def save_checkpoint(state, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.path.tar')
    torch.save(state, filename)


class KeyValueMemory(object):
    """
    kv is a dictionary, for each item, the key is the support vector, and the
    corresponding value is its one-hot label.
    """

    def __init__(self, x, x_labels):
        super().__init__()
        x_one_hot_labels = F.one_hot(x_labels)
        self.kv = dict()
        for i in range(len(x)):
            self.kv[x[i, :]] = x_one_hot_labels[i, :]

    def mem_size(self):
        return len(self.kv)


def softabs(alpha):
    """ The sharpening function used in Nat Comm """
    beta = 10
    sa = 1 / torch.exp(-(beta * (alpha - 0.5))) + 1 / torch.exp(-(beta * (-alpha - 0.5)))
    return sa


def sim_comp(kv, batch_features):
    """
    Input:
    - kv: a dictionary, see KeyValueMemory.
    - batch_features: a matrix, which is of size [batch * m, d].
    """

    ks = []
    vs = []
    for k, v in kv.items():
        ks.append(k)
        vs.append(v)
    ks = torch.stack(ks).cuda()  # a matrix, which is of size [mn, d]
    vs = torch.stack(vs).float().cuda()  # a matrix, which is of size [mn, m]

    # Cosine Similarity
    inner_product = torch.matmul(batch_features, ks.t())  # [batch * m, mn]
    ks_norm = torch.norm(ks, dim=1).unsqueeze(0)  # ks: [mn, d], ks_norm: [1, mn]
    feature_norm = torch.norm(batch_features, dim=1).unsqueeze(1)  # [batch * m, 1]
    norm_product = ks_norm * feature_norm  # [batch * m, mn]
    K = torch.squeeze(inner_product / (norm_product + 1e-8))

    # Calculating softabs
    K_exp = softabs(K)
    w = K_exp / torch.sum(K_exp, 1, keepdim=True)  # [batch * m, mn]

    # normalization
    w = (w - w.mean([0, 1], keepdim=True)) / w.std([0, 1], keepdim=True)

    ws = torch.matmul(w, vs)  # [batch * m, m]

    return ws
