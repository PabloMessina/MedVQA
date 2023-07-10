from sklearn.metrics.pairwise import cosine_similarity
from medvqa.utils.files import load_json
from medvqa.utils.logging import print_orange, print_red
from Levenshtein import distance as levenshtein_distance

EPS = 1e-6

class AnatomicalLocationTripletRanker:
    def __init__(self, anatloc2embedding):
        self.anatloc2embedding = anatloc2embedding

    def rank(self, query, fact1, fact2, return_metrics=False):
        q_vec = self.anatloc2embedding[query].reshape(1, -1)
        f1_vec = self.anatloc2embedding[fact1].reshape(1, -1)
        f2_vec = self.anatloc2embedding[fact2].reshape(1, -1)
        q1_cos_sim = cosine_similarity(q_vec, f1_vec).item()
        q2_cos_sim = cosine_similarity(q_vec, f2_vec).item()
        q1_lev_dist = levenshtein_distance(query, fact1)
        q2_lev_dist = levenshtein_distance(query, fact2)
        if q1_lev_dist < q2_lev_dist and q1_cos_sim > q2_cos_sim:
            pred = 1
        elif q1_lev_dist > q2_lev_dist and q1_cos_sim < q2_cos_sim:
            pred = -1
        else:
            pred = 0
        if return_metrics:
            return pred, q1_cos_sim, q2_cos_sim, q1_lev_dist, q2_lev_dist
        else:
            return pred
        
    def run_tests(self, tests_filepath):
        tests = load_json(tests_filepath)
        total = 0
        correct = 0
        incorrect = 0
        unsure = 0
        for test in tests:
            query = test['query']
            assert 'fact1' in test or 'facts1' in test # single or multiple facts
            assert 'fact2' in test or 'facts2' in test # single or multiple facts
            if 'fact1' in test:
                facts1 = [test['fact1']]
            else:
                facts1 = test['facts1']
            if 'fact2' in test:
                facts2 = [test['fact2']]
            else:
                facts2 = test['facts2']
            label = test['label']
            for fact1 in facts1:
                for fact2 in facts2:
                    pred, sim1, sim2, levd1, levd2 = self.rank(query, fact1, fact2, return_metrics=True)
                    total += 1
                    if abs(pred - label) == 2: # 1 vs. -1 or -1 vs. 1
                        incorrect += 1
                        print_red(f'Incorrect: wrong prediction for\n\tquery={query}\n\tfact1={fact1}\n\tfact2={fact2}'
                                     f'\n\tsim1={sim1}\n\tsim2={sim2}\n\tlevd1={levd1}\n\tlevd2={levd2}'
                                     f'\n\tlabel={label}\n\tpred={pred}')
                    elif pred == label:
                        correct += 1
                    else:
                        unsure += 1
                        print_orange(f'Unsure: unsure prediction for\n\tquery={query}\n\tfact1={fact1}\n\tfact2={fact2}'
                                     f'\n\tsim1={sim1}\n\tsim2={sim2}\n\tlevd1={levd1}\n\tlevd2={levd2}'
                                     f'\n\tlabel={label}\n\tpred={pred}')

        print(f'Accuracy: {correct / total:.2f}. Correct: {correct}, Incorrect: {incorrect}, Unsure: {unsure}')
