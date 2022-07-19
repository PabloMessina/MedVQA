from torch.optim import Adam, AdamW, SGD

def create_optimizer(name, params, lr):
    print(f'create_optimizer(): name = {name}')
    if name == 'adam':
        optimizer = Adam(params, lr=lr)
    elif name == 'adamw':
        optimizer = AdamW(params, lr=lr)
    elif name == 'sgd':
        optimizer = SGD(params, lr=lr)
    else:
        assert False, f'Unknown optimizer {name}'
    return optimizer