import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from medvqa.utils.metrics import chexpert_label_array_to_string
from medvqa.datasets.image_processing import inv_normalize
from medvqa.utils.constants import CHEXPERT_GENDERS, CHEXPERT_ORIENTATIONS
from medvqa.utils.files import get_cached_json_file
from medvqa.datasets.iuxray import IUXRAY_CACHE_DIR
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR

def inspect_chexpert_vision_trainer(chexpert_vision_trainer, i):
    instance = chexpert_vision_trainer.dataset[i]
    idx = instance['idx']
    print('idx:', idx)
    print('chexpert labels:', instance['l'])
    print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['l']))
    print('orientation:', instance['o'], CHEXPERT_ORIENTATIONS[instance['o']])
    print('sex:', instance['g'], CHEXPERT_GENDERS[instance['g']])
    print('image from tensor:')
    img = Image.fromarray((inv_normalize(instance['i']).permute(1,2,0) * 255).numpy().astype(np.uint8))
    plt.imshow(img)
    plt.show()
    print('image from path:')
    print(chexpert_vision_trainer.dataset.images[idx])
    img = Image.open(chexpert_vision_trainer.dataset.images[idx]).convert('RGB')
    plt.imshow(img)
    plt.show()

def inspect_iuxray_vision_trainer(iuxray_vision_trainer, split, i):
    if split == 'train':
        dataset = iuxray_vision_trainer.train_dataset
    else:
        assert split == 'validation'
        dataset = iuxray_vision_trainer.val_dataset
    instance = dataset[i]
    idx = instance['idx']
    print('idx:', idx)
    print('chexpert labels:', instance['chexpert'])
    print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['chexpert']))
    print('orientation:', instance['orientation'])
    print('question labels:', instance['qlabels'])
    print('image from tensor:')
    img = Image.fromarray((inv_normalize(instance['i']).permute(1,2,0) * 255).numpy().astype(np.uint8))
    plt.imshow(img)
    plt.show()
    print('image from path:')
    print(iuxray_vision_trainer.images[idx])
    img = Image.open(iuxray_vision_trainer.images[idx]).convert('RGB')
    plt.imshow(img)
    plt.show()    

def inspect_mimiccxr_vision_trainer(mimiccxr_vision_trainer, split, i):
    if split == 'train':
        dataset = mimiccxr_vision_trainer.train_dataset
    else:
        assert split == 'validation'
        dataset = mimiccxr_vision_trainer.val_dataset
    instance = dataset[i]
    idx = instance['idx']
    print('idx:', idx)
    print('chexpert labels:', instance['chexpert'])
    print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['chexpert']))
    print('orientation:', instance['orientation'])
    print('question labels:', instance['qlabels'])
    print('image from tensor:')
    img = Image.fromarray((inv_normalize(instance['i']).permute(1,2,0) * 255).numpy().astype(np.uint8))
    plt.imshow(img)
    plt.show()
    print('image from path:')
    print(mimiccxr_vision_trainer.images[idx])
    img = Image.open(mimiccxr_vision_trainer.images[idx]).convert('RGB')
    plt.imshow(img)
    plt.show()

def _inspect_vqa_trainer(vqa_trainer, cache_dir, split, i):
    if split == 'train':
        dataset = vqa_trainer.train_dataset
    else:
        assert split == 'validation'
        dataset = vqa_trainer.val_dataset
    instance = dataset[i]
    idx = instance['idx']    
    print('idx:', idx)
    print('chexpert labels:', instance['chexpert'])
    print('chexpert labels (verbose):', chexpert_label_array_to_string(instance['chexpert']))
    print('orientation:', instance['orientation'])
    print('question labels:', instance['qlabels'])

    qa_reports = get_cached_json_file(os.path.join(cache_dir, vqa_trainer.qa_adapted_reports_filename))
    print('question:', instance['q'], qa_reports['questions'][instance['q']])
    assert instance['qlabels'][instance['q']] == 1
    print('answer:', vqa_trainer.tokenizer.ids2string(instance['a']))

    print('image from tensor:')
    img = Image.fromarray((inv_normalize(instance['i']).permute(1,2,0) * 255).numpy().astype(np.uint8))
    plt.imshow(img)
    plt.show()
    print('image from path:')
    print(vqa_trainer.images[idx])
    img = Image.open(vqa_trainer.images[idx]).convert('RGB')
    plt.imshow(img)
    plt.show()

def inspect_iuxray_vqa_trainer(iuxray_vqa_trainer, split, i):
    _inspect_vqa_trainer(iuxray_vqa_trainer, IUXRAY_CACHE_DIR, split, i)

def inspect_mimiccxr_vqa_trainer(mimiccxr_vqa_trainer, split, i):
    _inspect_vqa_trainer(mimiccxr_vqa_trainer, MIMICCXR_CACHE_DIR, split, i)