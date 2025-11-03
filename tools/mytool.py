import torch.nn as nn

def unwrap_model(m):
    if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return m.module
    return m