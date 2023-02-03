import torch

class SimpleTemplateRGModel:
    def __init__(self, labels, templates, thresholds, label_order):

        # sanity checks
        assert len(thresholds) == len(labels)
        assert len(templates) == len(labels)
        # check that label_order is a subset of labels
        assert set(label_order).issubset(set(labels))
        # check that templates have values for 0 and 1
        for label in labels:
            if not set([0, 1]) == set(templates[label].keys()):
                raise Exception(f'Values missing for {label}: {templates[label].keys()}')
        
        self.labels = labels
        self.templates = templates
        self.thresholds = thresholds
        self.label_order = [labels.index(l) for l in label_order]

    def __call__(self, pred_probs):
        """Transforms classification scores into fixed templates."""
        # pred_probs shape: batch_size, n_labels (e.g. 100x14). Values in [0, 1]        
        if torch.is_tensor(pred_probs) or torch.is_tensor(pred_probs[0]):
            thresholds = torch.tensor(self.thresholds, device=pred_probs[0].device)
        else:
            thresholds = self.thresholds
        assert type(pred_probs[0]) == type(thresholds)        
        reports = []
        for i in range(len(pred_probs)):
            sample_probs = pred_probs[i] # shape: n_labels (e.g. 14)            
            sample_binary = sample_probs >= thresholds
            report = []
            for idx in self.label_order:
                label_name = self.labels[idx]
                sentence = self.templates[label_name][sample_binary[idx].item()]
                report.append(sentence)
            reports.append(report)
        return reports