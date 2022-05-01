import torch
import torch.nn as nn
from ignite.engine import Engine

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device):
    
    def step_fn(unused_engine, batch):

        # Extract elements from batch
        idxs = batch['idx']
        questions = batch['q'].to(device)
        question_lengths = batch['ql']
        answers = batch['a'].to(device)

        # Zero grad if training
        if training:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'questions': questions,
                'question_lengths': question_lengths,
            }

            if training:
                model_kwargs['answers'] = answers
                model_kwargs['mode'] = 'train'
            else:
                model_kwargs['max_answer_length'] = answers.size(1)
                model_kwargs['mode'] = 'eval'

            # Forward pass
            model_output = model(**model_kwargs)

            pred_answer_logits = model_output['pred_answers']
            pred_question_logits = model_output['pred_questions']            
            
            # Compute losses
            answer_loss = nlg_criterion(pred_answer_logits.view(-1, pred_answer_logits.shape[-1]), answers.view(-1))
            question_loss = nlg_criterion(pred_question_logits.view(-1, pred_question_logits.shape[-1]), questions.view(-1))            
            batch_loss = answer_loss + question_loss            
            
            # Backward pass + optimizer step if training
            if training:
                batch_loss.backward()
                optimizer.step()

        # Compute predicted Q & A
        pred_questions = pred_question_logits.argmax(-1)
        pred_answers = pred_answer_logits.argmax(-1)        
        
        output = {
            'idxs': idxs,
            'loss': batch_loss.detach(),
            'question_loss': question_loss.detach(),
            'answer_loss': answer_loss.detach(),
            'answers': tokenizer.clean_batch(answers.detach()),
            'questions': tokenizer.clean_batch(questions.detach()),
            'pred_answers': tokenizer.clean_batch(pred_answers.detach()),
            'pred_questions': tokenizer.clean_batch(pred_questions.detach()),
        }

        return output
    
    return step_fn

def get_engine(model, tokenizer, device, training=False, optimizer=None):
    nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss
    step_fn = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                          training=training, device=device)
    engine = Engine(step_fn)
    return engine