import numpy as np
import argparse
import torch
from tqdm import tqdm
from medvqa.datasets.nli.nli_dataset_management import _LABEL_TO_INDEX, NLI_Trainer, _INDEX_TO_LABEL
from medvqa.models.nlp.fact_encoder import FactEncoder
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata
from medvqa.models.nlp.nli import BertBasedNLI
from medvqa.utils.common import parsed_args_to_dict
from medvqa.datasets.dataloading_utils import get_fact_embedding_collate_batch_fn, get_nli_collate_batch_fn
from medvqa.utils.files import load_jsonl
from medvqa.utils.logging import print_blue, print_bold

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_folder_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='GPU')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--train_mode', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    return parser.parse_args(args=args)

def evaluate(
    checkpoint_folder_path,
    batch_size,
    num_workers,
    device='GPU',
    plot_confusion_matrix=False,
    train_mode=False,
    dev_mode=False,
    test_mode=False,
    compare_with_gpt4_predictions=False,    
    gpt4_predictions_path=None,
    use_bert_based_nli=False,
    use_mscxrt=False,
    use_radnli_test=False,
    sns_font_scale=1.8,
    font_size=25,
):
    print_blue('----- Evaluating model ------', bold=True)

    assert sum([train_mode, dev_mode, test_mode]) == 1 # Only one mode can be True

    metadata = load_metadata(checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']

    # device
    print_bold('device = ', device)
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    
    # Create model
    print_bold('Create model')
    if use_bert_based_nli:
        model = BertBasedNLI(**model_kwargs)
    else:
        model = FactEncoder(**model_kwargs)
    model = model.to(device)

    # Load model weights
    print_bold('Load model weights')
    model_checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    print('model_checkpoint_path = ', model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Create nli trainer
    if use_bert_based_nli:
        nli_collate_batch_fn = get_nli_collate_batch_fn(**collate_batch_fn_kwargs)
    else:
        nli_collate_batch_fn = get_fact_embedding_collate_batch_fn(**collate_batch_fn_kwargs['nli'])
    nli_trainer = NLI_Trainer(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_batch_fn=nli_collate_batch_fn,
        train_mode=train_mode,
        test_mode=test_mode,
        dev_mode=dev_mode,
        use_mscxrt=use_mscxrt,
        use_radnli_test=use_radnli_test,
    )
    if train_mode:
        dataloader = nli_trainer.train_dataloader
        dataset = nli_trainer.train_dataset
    elif dev_mode:
        dataloader = nli_trainer.dev_dataloader
        dataset = nli_trainer.dev_dataset
    elif test_mode:
        dataloader = nli_trainer.test_dataloader
        dataset = nli_trainer.test_dataset
    else: assert False

    # Run evaluation
    print_bold('Run evaluation')
    labels = []
    pred_labels = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            tok_p = batch['tokenized_premises']
            tok_h = batch['tokenized_hypotheses']
            if use_bert_based_nli:
                tok_p = {k: v.to(device) for k, v in tok_p.items()}
                tok_h = {k: v.to(device) for k, v in tok_h.items()}
                logits = model(tok_p, tok_h)
            else:
                p_input_ids = tok_p['input_ids'].to(device)
                p_attention_mask = tok_p['attention_mask'].to(device)
                h_input_ids = tok_h['input_ids'].to(device)
                h_attention_mask = tok_h['attention_mask'].to(device)
                logits = model.nli_forward(
                    p_input_ids=p_input_ids,
                    p_attention_mask=p_attention_mask,
                    h_input_ids=h_input_ids,
                    h_attention_mask=h_attention_mask,
                )
            batch_labels = batch['labels'].cpu().numpy()
            batch_pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
            labels.extend(batch_labels)
            pred_labels.extend(batch_pred_labels)
    labels = np.array(labels)
    pred_labels = np.array(pred_labels)
    
    # Accuracy
    accuracy = np.mean(labels == pred_labels)
    print(f'Accuracy = {accuracy:.4f}')

    # Plot a confusion matrix
    if plot_confusion_matrix:
        import seaborn as sns
        sns.set(font_scale=sns_font_scale)
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(labels, pred_labels)
        # Use label names instead of indices
        label_names = [_INDEX_TO_LABEL[label] for label in range(3)]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted', fontsize=font_size)
        plt.ylabel('True', fontsize=font_size)
        plt.show()
    
    output = {
        'accuracy': accuracy,
        'premises': dataset.premises,
        'hypotheses': dataset.hypotheses,
        'labels': [_INDEX_TO_LABEL[label] for label in labels],
        'pred_labels': [_INDEX_TO_LABEL[label] for label in pred_labels],
    }
    if compare_with_gpt4_predictions:
        assert gpt4_predictions_path is not None
        gpt4_predictions = load_jsonl(gpt4_predictions_path)
        gpt4_query_to_label = {x['metadata']['query'] : x['parsed_response'] for x in gpt4_predictions}
        queries = [f'Premise: {p} | Hypothesis: {h}' for p, h in zip(dataset.premises, dataset.hypotheses)]
        gpt4_labels = [_LABEL_TO_INDEX[gpt4_query_to_label[query]] for query in queries]
        gpt4_labels = np.array(gpt4_labels)
        gpt4_vs_gt_accuracy = np.mean(gpt4_labels == labels)
        gpt4_vs_pred_accuracy = np.mean(gpt4_labels == pred_labels)
        output['gpt4_vs_gt_accuracy'] = gpt4_vs_gt_accuracy
        output['gpt4_vs_pred_accuracy'] = gpt4_vs_pred_accuracy
        print(f'GPT4 vs GT accuracy = {gpt4_vs_gt_accuracy:.4f}')
        print(f'GPT4 vs Pred accuracy = {gpt4_vs_pred_accuracy:.4f}')
        output['gpt4_labels'] = [_INDEX_TO_LABEL[label] for label in gpt4_labels]

        if plot_confusion_matrix:
            # Plot a confusion matrix between GPT4 and Pred
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            cm = confusion_matrix(gpt4_labels, pred_labels)
            # Use label names instead of indices
            label_names = [_INDEX_TO_LABEL[label] for label in range(3)]
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
            plt.xlabel('Predicted', fontsize=font_size)
            plt.ylabel('GPT4', fontsize=font_size)
            plt.show()
    return output 

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)