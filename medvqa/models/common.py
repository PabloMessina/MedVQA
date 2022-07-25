class AnswerDecoding:
    TEACHER_FORCING = 'teacher-forcing'
    GREEDY_SEARCH = 'greedy-search'
    BEAM_SEARCH = 'beam-search'
    
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False