class AnswerDecoding:
    TEACHER_FORCING = 'teacher-forcing'
    GREEDY_SEARCH = 'greedy-search'
    BEAM_SEARCH = 'beam-search'
    
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def load_model_state_dict(model, state_dict, ignore_size_mismatch=True):
    if ignore_size_mismatch:
        model_state_dict = model.state_dict()
        for k in state_dict.keys():
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}")
                    del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
