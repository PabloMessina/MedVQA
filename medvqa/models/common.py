class AnswerDecoding:
    TEACHER_FORCING = 'teacher-forcing'
    GREEDY_SEARCH = 'greedy-search'
    BEAM_SEARCH = 'beam-search'
    
def freeze_parameters(model, ignore_name_regex=None):
    for name, param in model.named_parameters():
        if ignore_name_regex is not None and ignore_name_regex.search(name):
            print(f"Skip freezing parameter: {name}")
            continue
        param.requires_grad = False

def set_inplace_flag(module, inplace_value):
    for submodule in module.modules():
        if hasattr(submodule, 'inplace'):
            submodule.inplace = inplace_value