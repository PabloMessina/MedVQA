from medvqa.utils.constants import CHEXPERT_LABELS

def average_ignoring_nones(values):
    acc = 0
    count = 0
    for x in values:
        if x is not None:
            acc += x
            count += 1
    return acc / count if count > 0 else 0

def chexpert_label_array_to_string(label_array):
    return ', '.join(CHEXPERT_LABELS[i] for i, label in enumerate(label_array) if label == 1)

def chest_imagenome_label_array_to_string(label_array, label_names):
    return '\n'.join(f'({", ".join(label_names[i])})' for i, label in enumerate(label_array) if label == 1)

def question_label_array_to_string(questions, label_array):
    assert len(questions) == len(label_array)
    return ', '.join(questions[i] for i, label in enumerate(label_array) if label == 1)