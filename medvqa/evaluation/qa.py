from medvqa.utils.files import get_cached_json_file
from medvqa.utils.metrics import chexpert_label_array_to_string
import random

class QAExamplePlotter:

    def __init__(self, dataset_name, results,
                qa_adapted_reports_file_path=None,
        ):
        dataset = results[f'{dataset_name}_dataset']
        tokenizer = results['tokenizer']
        metrics_dict = results[f'{dataset_name}_metrics'] 
        idxs = metrics_dict['idxs']        

        self.idxs = idxs
        self.metrics_dict = metrics_dict        
        self.questions = [tokenizer.ids2string(tokenizer.clean_sentence(dataset.questions[i])) for i in idxs]
        self.answers = [tokenizer.ids2string(tokenizer.clean_sentence(dataset.answers[i])) for i in idxs]
        self.pred_questions = [tokenizer.ids2string(x) for x in metrics_dict['pred_questions']]
        self.pred_answers = [tokenizer.ids2string(x) for x in metrics_dict['pred_answers']]
        self.dataset = dataset

        # optional
        if qa_adapted_reports_file_path is not None:
            self.reports = get_cached_json_file(qa_adapted_reports_file_path)
        else:
            self.reports = None

    def inspect_example(self, metrics_to_inspect, metrics_to_rank=None, idx=None, question=None, mode='random'):

        if idx is None:
            indices = [i for i, q in enumerate(self.questions) if q == question]
            assert len(indices) > 0, f'no match for question {question}'
            
            if mode == 'random':
                idx = random.choice(indices)
            else:
                if metrics_to_rank is None:
                    metrics_to_rank = metrics_to_inspect
                indices.sort(key=lambda i : sum(self.metrics_dict[m][i] for m in metrics_to_rank))
                if mode == 'best':
                    idx = indices[-1]
                else:
                    idx = indices[0]
        
        if self.reports:
            rid = self.dataset.report_ids[self.idxs[idx]]
            report = self.reports['reports'][rid]
            report = '. '.join(report['sentences'][i] for i in report['matched'])
            print('Report:\n')
            print(report)
            print("\n===================")
        
        print('idx:', idx)
        print('--')
        print('question:', self.questions[idx])
        print('pred_question:', self.pred_questions[idx])
        print('--')
        print('answer:', self.answers[idx])
        print('pred_answer:', self.pred_answers[idx])
        print('--')
        print('chexpert_labels_gt:', self.metrics_dict['chexpert_labels_gt'][idx])
        print('chexpert_labels_gen:', self.metrics_dict['chexpert_labels_gen'][idx])
        print('chexpert_labels_gt (verbose):', chexpert_label_array_to_string(self.metrics_dict['chexpert_labels_gt'][idx]))
        print('chexpert_labels_gen (verbose):', chexpert_label_array_to_string(self.metrics_dict['chexpert_labels_gen'][idx]))
        print('--')
        for m in metrics_to_inspect:
            print(f'{m}:', self.metrics_dict[m][idx])        