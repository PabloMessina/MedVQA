import torch
import torch.nn as nn
from ignite.engine import Engine
from medvqa.utils.constants import MIMICCXR_DATASET_ID

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device,
            # tags aux task
            use_tags=False,
            tags_criterion=None,
            # orientation aux task
            use_orientation=False,
            iuxray_orientation_criterion=None,
            mimiccxr_orientation_criterion=None,
            # chexpert aux task
            use_chexpert=False,
            chexpert_criterion=None,
    ):
    
    def step_fn(unused_engine, batch):

        # Extract elements from batch
        idxs = batch['idx']
        images = batch['i'].to(device)
        questions = batch['q'].to(device)
        question_lengths = batch['ql']
        answers = batch['a'].to(device)        
        
        if use_tags:
            tags = batch['tags'].to(device)        
        if use_orientation:
            dataset_id = batch['dataset_id']
            orientation = batch['orientation'].to(device)
        if use_chexpert:
            chexpert = batch['chexpert'].to(device)

        # Zero grad if training
        if training:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'images': images,
                'questions': questions,
                'question_lengths': question_lengths,
            }

            if training:
                model_kwargs['answers'] = answers
                model_kwargs['mode'] = 'train'
            else:
                model_kwargs['max_answer_length'] = answers.size(1)
                model_kwargs['mode'] = 'eval'

            if use_orientation:
                if dataset_id == MIMICCXR_DATASET_ID:
                    model_kwargs['mimiccxr_foward'] = True
                else:
                    model_kwargs['iuxray_foward'] = True

            # Forward pass
            model_output = model(**model_kwargs)

            pred_answer_logits = model_output['pred_answers']
            pred_question_logits = model_output['pred_questions']
            
            if use_tags:
                pred_tags_logits = model_output['pred_tags']            
            if use_orientation:
                if dataset_id == MIMICCXR_DATASET_ID:
                    pred_orientation_logits = model_output['mimiccxr_pred_orientation']
                else:
                    pred_orientation_logits = model_output['iuxray_pred_orientation']
            if use_chexpert:
                pred_chexpert_logits = model_output['pred_chexpert']

            # Compute losses
            answer_loss = nlg_criterion(pred_answer_logits.view(-1, pred_answer_logits.shape[-1]), answers.view(-1))
            question_loss = nlg_criterion(pred_question_logits.view(-1, pred_question_logits.shape[-1]), questions.view(-1))            
            batch_loss = answer_loss + question_loss
            
            if use_tags:
                tags_loss = tags_criterion(pred_tags_logits, tags.float())
                batch_loss += tags_loss
            if use_orientation:
                if dataset_id == MIMICCXR_DATASET_ID:
                    orientation_loss = mimiccxr_orientation_criterion(pred_orientation_logits, orientation)
                else:
                    orientation_loss = iuxray_orientation_criterion(pred_orientation_logits, orientation)
                batch_loss += orientation_loss            
            if use_chexpert:
                batch_loss += chexpert_criterion(pred_chexpert_logits, chexpert.float())

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
            'answers': tokenizer.clean_batch(answers.detach()),
            'questions': tokenizer.clean_batch(questions.detach()),
            'pred_answers': tokenizer.clean_batch(pred_answers.detach()),
            'pred_questions': tokenizer.clean_batch(pred_questions.detach()),
        }
        if use_tags:
            output['tags'] = tags.detach().cpu()
            output['pred_tags'] = (pred_tags_logits.detach() > 0).cpu()
        if use_orientation:
            output['orientation'] = orientation.detach()
            output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()
            output['dataset_id'] = dataset_id
        if use_chexpert:
            output['chexpert'] = chexpert.detach().cpu()
            output['pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()

        return output
    
    return step_fn

def get_engine(model, tokenizer, use_tags, use_orientation, use_chexpert,
               device, training=False, optimizer=None):
    # Criterion
    nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss
    
    if use_tags: # auxiliary task
        tags_criterion = nn.BCEWithLogitsLoss()
    else:
        tags_criterion = None
    
    if use_orientation: # auxiliary task
        iuxray_orientation_criterion = nn.CrossEntropyLoss()
        mimiccxr_orientation_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore unknown
    else:
        iuxray_orientation_criterion = None
        mimiccxr_orientation_criterion = None

    if use_chexpert: # auxiliary task
        chexpert_criterion = nn.BCEWithLogitsLoss()
    else:
        chexpert_criterion = None

    # Create engine
    step_fn = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                          training=training, device=device,
                          # tags auxiliary task
                          use_tags=use_tags,
                          tags_criterion=tags_criterion,
                          # orientation auxiliary task
                          use_orientation=use_orientation,
                          iuxray_orientation_criterion=iuxray_orientation_criterion,
                          mimiccxr_orientation_criterion=mimiccxr_orientation_criterion,
                          # chexpert auxiliary task
                          use_chexpert=use_chexpert,
                          chexpert_criterion=chexpert_criterion)
    engine = Engine(step_fn)
    return engine