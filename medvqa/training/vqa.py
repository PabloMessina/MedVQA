import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.utils.constants import MIMICCXR_DATASET_ID

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device,
            include_answer=True,
            max_answer_length=None,
            # automatic mixed precision
            use_amp=False,
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
            # question auxiliary task
            classify_questions=False,
            question_criterion=None,
    ):

    scaler = GradScaler(enabled=use_amp)

    if not include_answer:
        assert max_answer_length is not None
    
    def step_fn(unused_engine, batch):

        # Extract elements from batch
        idxs = batch['idx']
        images = batch['i'].to(device)
        questions = batch['q'].to(device)
        question_lengths = batch['ql']
        if include_answer:
            answers = batch['a'].to(device)
        
        if use_tags:
            tags = batch['tags'].to(device)
        if use_orientation:
            dataset_id = batch['dataset_id']
            orientation = batch['orientation'].to(device)
        if use_chexpert:
            chexpert = batch['chexpert'].to(device)
        if classify_questions:
            question_labels = batch['qlabels'].to(device)
        
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
                if include_answer:
                    model_kwargs['max_answer_length'] = answers.size(1)
                else:                    
                    model_kwargs['max_answer_length'] = max_answer_length
                model_kwargs['mode'] = 'eval'

            if use_orientation:
                if dataset_id == MIMICCXR_DATASET_ID:
                    model_kwargs['mimiccxr_foward'] = True
                else:
                    model_kwargs['iuxray_foward'] = True

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

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
                if classify_questions:
                    pred_qlabels_logits = model_output['pred_qlabels']

                # Compute losses
                question_loss = nlg_criterion(pred_question_logits.view(-1, pred_question_logits.shape[-1]), questions.view(-1))            
                batch_loss = question_loss
                if include_answer:
                    answer_loss = nlg_criterion(pred_answer_logits.view(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                    batch_loss += answer_loss
                
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
                    chexpert_loss = chexpert_criterion(pred_chexpert_logits, chexpert.float())
                    batch_loss += chexpert_loss
                if classify_questions:
                    qlabels_loss = question_criterion(pred_qlabels_logits, question_labels.float())
                    batch_loss += qlabels_loss

            # Backward pass + optimizer step if training
            if training:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # batch_loss.backward()
                # optimizer.step()
                optimizer.zero_grad()

        # Compute predicted Q & A
        pred_questions = pred_question_logits.argmax(-1)
        pred_answers = pred_answer_logits.argmax(-1)        
        
        output = {
            'idxs': idxs,
            'loss': batch_loss.detach(),
            'question_loss': question_loss.detach(),
            'questions': tokenizer.clean_batch(questions.detach()),
            'pred_answers': tokenizer.clean_batch(pred_answers.detach()),
            'pred_questions': tokenizer.clean_batch(pred_questions.detach()),
        }
        if include_answer:
            output['answer_loss'] = answer_loss.detach()
            output['answers'] = tokenizer.clean_batch(answers.detach())
        if use_tags:
            output['tags'] = tags.detach().cpu()
            output['pred_tags'] = (pred_tags_logits.detach() > 0).cpu()
            output['tags_loss'] = tags_loss.detach()
        if use_orientation:
            output['orientation'] = orientation.detach()
            output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()
            output['dataset_id'] = dataset_id
            output['orientation_loss'] = orientation_loss.detach()
        if use_chexpert:
            output['chexpert'] = chexpert.detach().cpu()
            output['pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
            output['chexpert_loss'] = chexpert_loss.detach()
        if classify_questions:
            output['qlabels'] = question_labels.detach().cpu()
            output['pred_qlabels'] = (pred_qlabels_logits.detach() > 0).cpu()
            output['qlabels_loss'] = qlabels_loss.detach()

        return output
    
    return step_fn

def get_engine(model, tokenizer, use_tags, use_orientation, use_chexpert, classify_questions,
               device, include_answer=True, max_answer_length = None,
               use_amp=False, training=False, optimizer=None):
    # Criterion
    nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss
    
    # Auxiliary tasks
    if use_tags:
        tags_criterion = nn.BCEWithLogitsLoss()
    else:
        tags_criterion = None
    
    if use_orientation:
        iuxray_orientation_criterion = nn.CrossEntropyLoss()
        mimiccxr_orientation_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore unknown
    else:
        iuxray_orientation_criterion = None
        mimiccxr_orientation_criterion = None

    if use_chexpert:
        chexpert_criterion = nn.BCEWithLogitsLoss()
    else:
        chexpert_criterion = None
    
    if classify_questions:
        question_criterion = nn.BCEWithLogitsLoss()
    else:
        question_criterion = None

    # Create engine
    step_fn = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                          include_answer=include_answer, max_answer_length=max_answer_length,
                          training=training,
                          device=device, use_amp=use_amp,
                          # tags auxiliary task
                          use_tags=use_tags,
                          tags_criterion=tags_criterion,
                          # orientation auxiliary task
                          use_orientation=use_orientation,
                          iuxray_orientation_criterion=iuxray_orientation_criterion,
                          mimiccxr_orientation_criterion=mimiccxr_orientation_criterion,
                          # chexpert auxiliary task
                          use_chexpert=use_chexpert,
                          chexpert_criterion=chexpert_criterion,
                          # question auxiliary task
                          classify_questions=classify_questions,
                          question_criterion=question_criterion,
                          )
    engine = Engine(step_fn)
    return engine