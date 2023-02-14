from medvqa.training.vqa import get_engine as _get_vqa_engine

def get_engine(**kwargs):
    # Override the default VQA engine to use the visual module only
    # This way we avoid code duplication :)
    return _get_vqa_engine(**kwargs, use_visual_module_only=True)