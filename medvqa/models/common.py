import random
from medvqa.utils.logging import print_orange

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

def load_model_state_dict(model, state_dict, ignore_size_mismatch=True, strict=False):
    if ignore_size_mismatch:
        model_state_dict = model.state_dict()
        to_delete = []
        for k in state_dict.keys():
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}")
                    to_delete.append(k)
        for k in to_delete:
            del state_dict[k]
    # count intersection over union of keys
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    intersection = model_keys & state_dict_keys
    union = model_keys | state_dict_keys
    if len(intersection) != len(union):
        print_orange(f"Warning: model state dict has {len(model_keys)} keys, "
            f"loaded state dict has {len(state_dict_keys)} keys, "
            f"intersection has {len(intersection)} keys, "
            f"union has {len(union)} keys.")
        missing_keys = list(model_keys - state_dict_keys)
        if len(missing_keys) > 0:
            print_orange("Examples of keys in model but not in loaded state dict:")
            missing_keys = random.sample(missing_keys, min(10, len(missing_keys)))
            for k in missing_keys:
                print_orange(f"  {k}")
        missing_keys = list(state_dict_keys - model_keys)
        if len(missing_keys) > 0:
            print_orange("Examples of keys in loaded state dict but not in model:")
            missing_keys = random.sample(missing_keys, min(10, len(missing_keys)))
            for k in missing_keys:
                print_orange(f"  {k}")
    model.load_state_dict(state_dict, strict=strict)