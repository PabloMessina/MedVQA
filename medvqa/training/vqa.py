import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.models.vqa.open_ended_vqa import QuestionEncoding
from medvqa.models.vqa.answer_decoder import AnswerDecoding
from medvqa.losses import get_binary_multilabel_loss
from medvqa.utils.constants import IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID, CHEXPERT_DATASET_ID

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device,
        question_encoding = QuestionEncoding.BILSTM,
        answer_decoding = AnswerDecoding.TEACHER_FORCING,
        beam_search_k = None,
        include_answer=True,
        max_answer_length=None,
        # automatic mixed precision
        use_amp=False,
        # tags aux task
        classify_tags=False,
        tags_criterion=None,
        # orientation aux task
        classify_orientation=False,
        iuxray_orientation_criterion=None,
        mimiccxr_orientation_criterion=None,
        # chexpert aux task
        classify_chexpert=False,
        chexpert_criterion=None,
        # question auxiliary task
        classify_questions=False,
        question_criterion=None,
        # chexpert dataset
        chexpert_aux_criterion=None,
    ):

    scaler = GradScaler(enabled=use_amp)
    verbose_question = question_encoding != QuestionEncoding.ONE_HOT   

    assert training == (answer_decoding == AnswerDecoding.TEACHER_FORCING)

    use_beam_search = answer_decoding == AnswerDecoding.BEAM_SEARCH
    
    if use_beam_search:
        assert beam_search_k is not None

    if not include_answer:
        assert max_answer_length is not None
    
    def step_fn__mimiccxr_iuxray(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        images = batch['i'].to(device)
        questions = batch['q'].to(device)
        if verbose_question:
            question_lengths = batch['ql']
        if include_answer:
            answers = batch['a'].to(device)
        
        if classify_tags:
            tags = batch['tags'].to(device)
        if classify_orientation:
            orientation = batch['orientation'].to(device)
        if classify_chexpert:
            chexpert = batch['chexpert'].to(device)
        if classify_questions:
            question_labels = batch['qlabels'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'images': images,
                'questions': questions,
                'device': device,
            }
            if verbose_question:
                model_kwargs['question_lengths'] = question_lengths

            if training:
                model_kwargs['mode'] = 'train'
                model_kwargs['answers'] = answers
            else:
                model_kwargs['mode'] = 'eval'
                if use_beam_search:
                    model_kwargs['beam_search_k'] = beam_search_k
                if include_answer:
                    model_kwargs['max_answer_length'] = answers.size(1)
                else:                    
                    model_kwargs['max_answer_length'] = max_answer_length

            if classify_orientation:
                if dataset_id == MIMICCXR_DATASET_ID:
                    model_kwargs['mimiccxr_foward'] = True
                else:
                    model_kwargs['iuxray_foward'] = True

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)

                if training:
                    pred_answer_logits = model_output['pred_answers']
                else:
                    pred_answers = model_output['pred_answers']
                
                if verbose_question:
                    pred_question_logits = model_output['pred_questions']
                
                if classify_tags:
                    pred_tags_logits = model_output['pred_tags']            
                if classify_orientation:
                    if dataset_id == MIMICCXR_DATASET_ID:
                        pred_orientation_logits = model_output['mimiccxr_pred_orientation']
                    else:
                        pred_orientation_logits = model_output['iuxray_pred_orientation']
                if classify_chexpert:
                    pred_chexpert_logits = model_output['pred_chexpert']
                    pred_chexpert_probs = model_output['pred_chexpert_probs']
                if classify_questions:
                    pred_qlabels_logits = model_output['pred_qlabels']

                if training:                    
                    # Compute losses
                    losses = []
                    if verbose_question:
                        question_loss = nlg_criterion(pred_question_logits.view(-1, pred_question_logits.shape[-1]), questions.view(-1))            
                        losses.append(question_loss)
                    if include_answer:
                        answer_loss = nlg_criterion(pred_answer_logits.view(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        losses.append(answer_loss)                    
                    if classify_tags:
                        tags_loss = tags_criterion(pred_tags_logits, tags.float())
                        losses.append(tags_loss)
                    if classify_orientation:
                        if dataset_id == MIMICCXR_DATASET_ID:
                            orientation_loss = mimiccxr_orientation_criterion(pred_orientation_logits, orientation)
                        else:
                            orientation_loss = iuxray_orientation_criterion(pred_orientation_logits, orientation)
                        losses.append(orientation_loss)
                    if classify_chexpert:
                        chexpert_loss = chexpert_criterion(pred_chexpert_logits, chexpert.float())
                        losses.append(chexpert_loss)
                    if classify_questions:
                        qlabels_loss = question_criterion(pred_qlabels_logits, question_labels.float())
                        losses.append(qlabels_loss)

                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None

                    # Backward pass + optimizer step if training
                    assert batch_loss is not None
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # batch_loss.backward()
                    # optimizer.step()
                    optimizer.zero_grad()

        # Compute predicted Q & A
        if training:
            pred_answers = pred_answer_logits.argmax(-1)
        if verbose_question:
            pred_questions = pred_question_logits.argmax(-1)
        
        output = {
            'idxs': idxs,
            'pred_answers': tokenizer.clean_batch(pred_answers.detach()),
            'dataset_id': dataset_id,
        }            
        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if verbose_question:
            output['questions'] = tokenizer.clean_batch(questions.detach())
            output['pred_questions'] = tokenizer.clean_batch(pred_questions.detach())
            if training:
                output['question_loss'] = question_loss.detach()
        else:
            output['questions'] = questions.detach()
        if include_answer:
            output['answers'] = tokenizer.clean_batch(answers.detach())
            if training:
                output['answer_loss'] = answer_loss.detach()
        if classify_tags:
            output['tags'] = tags.detach().cpu()
            output['pred_tags'] = (pred_tags_logits.detach() > 0).cpu()
            if training:
                output['tags_loss'] = tags_loss.detach()
        if classify_orientation:
            output['orientation'] = orientation.detach()
            output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()            
            if training:
                output['orientation_loss'] = orientation_loss.detach()
        if classify_chexpert:
            output['chexpert'] = chexpert.detach().cpu()
            output['pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
            output['pred_chexpert_probs'] = pred_chexpert_probs.detach().cpu()
            if training:
                output['chexpert_loss'] = chexpert_loss.detach()
        if classify_questions:
            output['qlabels'] = question_labels.detach().cpu()
            output['pred_qlabels'] = (pred_qlabels_logits.detach() > 0).cpu()
            if training:
                output['qlabels_loss'] = qlabels_loss.detach()

        return output
    
    def step_fn__chexpert(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        images = batch['i'].to(device)
        orientations = batch['o'].to(device)
        genders = batch['g'].to(device)
        chexpert = batch['l'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'images': images,
                'chexpert_forward': True,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)
                
                pred_chexpert_logits = model_output['pred_chexpert']
                pred_chexpert_probs = model_output['pred_chexpert_probs']
                pred_orientation_logits = model_output['pred_orientation']
                pred_gender_logits = model_output['pred_gender']

                if training:                    
                    # Compute losses
                    chexpert_loss = chexpert_criterion(pred_chexpert_logits, chexpert.float())
                    orientation_loss = chexpert_aux_criterion(pred_orientation_logits, orientations)
                    gender_loss = chexpert_aux_criterion(pred_gender_logits, genders)                    
                    batch_loss = chexpert_loss + orientation_loss + gender_loss

                    # Backward pass + optimizer step if training
                    assert batch_loss is not None
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # batch_loss.backward()
                    # optimizer.step()
                    optimizer.zero_grad()
        
        output = {
            'idxs': idxs,
        }            
        if training:
            output['loss'] = batch_loss.detach()
            
        output['chexpert'] = chexpert.detach().cpu()
        output['pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
        output['pred_chexpert_probs'] = pred_chexpert_probs.detach().cpu()
        if training:
            output['chexpert_loss'] = chexpert_loss.detach()

        output['orientation'] = orientations.detach()
        output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()
        output['dataset_id'] = dataset_id
        if training:
            output['orientation_loss'] = orientation_loss.detach()

        output['gender'] = genders.detach()
        output['pred_gender'] = pred_gender_logits.argmax(-1).detach()
        output['dataset_id'] = dataset_id
        if training:
            output['gender_loss'] = gender_loss.detach()

        return output

    def step_fn(unused_engine, batch):        
        dataset_id = batch['dataset_id']
        # print(f"step_fn(dataset_id={dataset_id})")
        if dataset_id == MIMICCXR_DATASET_ID or dataset_id == IUXRAY_DATASET_ID:
            return step_fn__mimiccxr_iuxray(batch)
        if dataset_id == CHEXPERT_DATASET_ID:
            return step_fn__chexpert(batch)
        assert False, f'Unknown dataset_id {dataset_id}'
    
    return step_fn

def get_engine(model, tokenizer, classify_tags, classify_orientation, classify_chexpert, classify_questions,
               question_encoding, answer_decoding, device,
               binary_loss_name='bce',
               beam_search_k=None, include_answer=True, max_answer_length=None,
               use_amp=False,
               training=False,
               train_with_chexpert_dataset=False,
               optimizer=None):
    # Criterion
    nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss
    
    # Auxiliary tasks
    if classify_tags:
        tags_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        tags_criterion = None
    
    if classify_orientation:
        iuxray_orientation_criterion = nn.CrossEntropyLoss()
        mimiccxr_orientation_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore unknown
    else:
        iuxray_orientation_criterion = None
        mimiccxr_orientation_criterion = None

    if classify_chexpert or train_with_chexpert_dataset:
        chexpert_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        chexpert_criterion = None
    
    if classify_questions:
        question_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        question_criterion = None

    if train_with_chexpert_dataset:
        chexpert_aux_criterion = nn.CrossEntropyLoss()
    else:
        chexpert_aux_criterion = None

    # Create engine
    step_fn = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                          include_answer=include_answer, max_answer_length=max_answer_length,
                          training=training,
                          device=device, use_amp=use_amp,
                          question_encoding=question_encoding,
                          answer_decoding=answer_decoding,
                          beam_search_k=beam_search_k,
                          # tags auxiliary task
                          classify_tags=classify_tags,
                          tags_criterion=tags_criterion,
                          # orientation auxiliary task
                          classify_orientation=classify_orientation,
                          iuxray_orientation_criterion=iuxray_orientation_criterion,
                          mimiccxr_orientation_criterion=mimiccxr_orientation_criterion,
                          # chexpert auxiliary task
                          classify_chexpert=classify_chexpert,
                          chexpert_criterion=chexpert_criterion,
                          # question auxiliary task
                          classify_questions=classify_questions,
                          question_criterion=question_criterion,
                          # chexpert dataset
                          chexpert_aux_criterion=chexpert_aux_criterion,
                          )
    engine = Engine(step_fn)
    return engine