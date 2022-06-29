def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False