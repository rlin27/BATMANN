import torch


def sim_comp_softmax(kv, batch_features):
    """
    Use softmax instead of softabs as the sharpening function

    Input:
    - kv: a dictionary, see KeyValueMemory.
    - batch_features: a matrix, which is of size [batch * mn, d].
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
    K_exp = torch.exp(K)
    w = K_exp / torch.sum(K_exp, 1, keepdim=True)  # [batch * m, mn]

    # normalization
    w = (w - w.mean([0, 1], keepdim=True)) / w.std([0, 1], keepdim=True)

    ws = torch.matmul(w, vs)  # [batch * m, m]

    return ws


def sim_comp_approx(kv, batch_features, binary_id=1):
    """
    Use softmax instead of softabs as the sharpening function

    Input:
    - kv: a dictionary, see KeyValueMemory.
    - batch_features: a matrix, which is of size [batch * m, d].
    - binary_id: an int, binary_id=1 means the features are formed by {-1,1}^dim,
      binary_id=2 means the features are formed by {0,1}^dim.
    """
    ks = []
    vs = []
    for k, v in kv.items():
        ks.append(k)
        vs.append(v)
    ks = torch.stack(ks).cuda()  # a matrix, which is of size [mn, d]
    vs = torch.stack(vs).float().cuda()  # a matrix, which is of size [mn, m]

    # Dot Similarity
    # Case 1: called bipolar in the Nat Comm paper (feature vectors only contain {-1, 1})
    if binary_id == 1:
        w = 1 / batch_features.size(1) * torch.matmul(batch_features, ks.t())  # [batch * m, mn]

    # Case 2: called binary in the Nat Comm paper (feature vectors only contain {0, 1})
    elif binary_id == 2:
        w = 1/2 + 1 / (2 * batch_features.size(1)) * torch.matmul(batch_features, ks.t())

    ws = torch.matmul(w, vs)  # [batch * m, m]

    return ws

