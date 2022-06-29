class SimpleTemplateRGModel:
    def __init__(self, diseases, templates, order):       

        # sanity checks
        if not set(order).issubset(set(diseases)):
            raise Exception(f'Order contains invalid diseases: {order} vs {self.diseases}')

        for disease in order:
            if not set([0, 1]).issubset(set(templates[disease].keys())):
                raise Exception(f'Values missing for {disease}: {templates[disease].keys()}')        
        
        self.templates = templates
        self.diseases = diseases
        self.disease_order = [self.diseases.index(d) for d in order]

    def __call__(self, labels):
        """Transforms classification scores into fixed templates."""
        # labels shape: batch_size, n_diseases (binary)

        reports = []
        for i in range(len(labels)):
            sample_predictions = labels[i]
            # shape: n_diseases

            report = []
            for disease_index in self.disease_order:
                pred_value = sample_predictions[disease_index].item()
                disease_name = self.diseases[disease_index]
                sentence = self.templates[disease_name][pred_value]
                report.append(sentence)

            reports.append(report)

        return reports