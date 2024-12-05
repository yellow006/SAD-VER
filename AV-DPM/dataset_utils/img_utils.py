import torch

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    # rescale在不改变图像归一化(0,1)分布的情况下，将数据的范围化为(-1,1)
    # inverse_rescale则相反。
    if config.data.rescaled:
        X = 2 * X - 1.0

    return X

def inverse_data_transform(config, X):
    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
