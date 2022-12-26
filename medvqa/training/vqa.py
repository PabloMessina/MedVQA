import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.models.vqa.open_ended_vqa import QuestionEncoding
from medvqa.models.common import AnswerDecoding
from medvqa.losses import get_binary_multilabel_loss
from medvqa.utils.constants import (
    CXR14_DATASET_ID,
    CHEXPERT_TASKS,
    IUXRAY_DATASET_ID,
    IUXRAY_DATASET_ID__CHEXPERT_MODE,
    MIMICCXR_DATASET_ID,
    CHEXPERT_DATASET_ID,
    MIMICCXR_DATASET_ID__CHEXPERT_MODE,
    VINBIG_DATASET_ID,
)

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device,        
        question_encoding = QuestionEncoding.BILSTM,
        answer_decoding = AnswerDecoding.TEACHER_FORCING,
        shift_answer = False,
        beam_search_k = None,
        include_answer=True,
        include_image=True,
        include_visual_features=False,
        max_answer_length=None,
        use_merged_findings=False,
        iters_to_accumulate=1, # for gradient accumulation
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
        chexpert_mode=None,
        # cxr14 dataset
        cxr14_criterion=None,
        # vinbig dataset
        vinbig_criterion=None,
        # batchwise learning rate updatse
        update_lr_batchwise=False,
        lr_scheduler=None,
    ):

    scaler = GradScaler(enabled=use_amp)
    verbose_question = question_encoding != QuestionEncoding.ONE_HOT   

    assert training == (answer_decoding == AnswerDecoding.TEACHER_FORCING)

    use_beam_search = answer_decoding == AnswerDecoding.BEAM_SEARCH
    
    if use_beam_search:
        assert beam_search_k is not None

    if not include_answer:
        assert max_answer_length is not None

    if update_lr_batchwise:
        assert lr_scheduler is not None

    use_chexpert_vqa = chexpert_mode == CHEXPERT_TASKS.VQA

    if training:
        iters_count = 0
        def backward_and_optimizer_step(batch_loss):
            nonlocal iters_count
            assert batch_loss is not None
            batch_loss = batch_loss / iters_to_accumulate
            scaler.scale(batch_loss).backward()
            if (iters_count + 1) % iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                # batch_loss.backward()
                # optimizer.step()
                optimizer.zero_grad()                
            iters_count += 1
    
    def step_fn__mimiccxr_iuxray(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        questions = batch['q'].to(device)
        if include_image:
            images = batch['i'].to(device)
        if include_visual_features:
            visual_features = batch['vf'].to(device)
        if verbose_question:
            question_lengths = batch['ql']
        if include_answer:
            answers = batch['a'].to(device)
            if shift_answer:
                answers_start = answers[:, :-1]
                answers_end = answers[:, 1:]

        is_mimiccxr = (dataset_id == MIMICCXR_DATASET_ID or dataset_id == MIMICCXR_DATASET_ID__CHEXPERT_MODE)

        findings_name = 'findings' if use_merged_findings else 'chexpert'
        
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
                'questions': questions,
                'device': device,
            }
            
            if include_image:
                model_kwargs['raw_images'] = images
            if include_visual_features:
                model_kwargs['visual_features'] = visual_features

            if verbose_question:
                model_kwargs['question_lengths'] = question_lengths

            if training:
                model_kwargs['mode'] = 'train'
                if shift_answer:
                    model_kwargs['answers'] = answers_start
                else:
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
                if is_mimiccxr:
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
                    if is_mimiccxr:
                        pred_orientation_logits = model_output['mimiccxr_pred_orientation']
                    else:
                        pred_orientation_logits = model_output['iuxray_pred_orientation']
                if classify_chexpert:
                    pred_chexpert_logits = model_output[f'pred_{findings_name}']
                    pred_chexpert_probs = model_output[f'pred_{findings_name}_probs']
                if classify_questions:
                    pred_qlabels_logits = model_output['pred_qlabels']

                if training:                    
                    # Compute losses
                    losses = []
                    if verbose_question:
                        question_loss = nlg_criterion(pred_question_logits.view(-1, pred_question_logits.shape[-1]), questions.view(-1))            
                        losses.append(question_loss)
                    if include_answer:
                        # print('pred_answer_logits.shape =', pred_answer_logits.shape)
                        # print('answers.shape =', answers.shape)
                        # answer_loss = nlg_criterion(pred_answer_logits.view(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        if shift_answer:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers_end.reshape(-1))
                        else:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        losses.append(answer_loss)
                    if classify_tags:
                        tags_loss = tags_criterion(pred_tags_logits, tags.float())
                        losses.append(tags_loss)
                    if classify_orientation:
                        if is_mimiccxr:
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
                    backward_and_optimizer_step(batch_loss)

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
            output[f'pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
            output[f'pred_chexpert_probs'] = pred_chexpert_probs.detach().cpu()
            if training:
                output[f'chexpert_loss'] = chexpert_loss.detach()
        if classify_questions:
            output['qlabels'] = question_labels.detach().cpu()
            output['pred_qlabels'] = (pred_qlabels_logits.detach() > 0).cpu()
            if training:
                output['qlabels_loss'] = qlabels_loss.detach()

        return output
    
    def step_fn__chexpert_cxr14(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        orientations = batch['o'].to(device)
        genders = batch['g'].to(device)
        labels = batch['l'].to(device)
        use_vqa_mode = use_chexpert_vqa or dataset_id == CXR14_DATASET_ID
        dataset_name = 'chexpert' if dataset_id == CHEXPERT_DATASET_ID else 'cxr14'
        findings_name = 'findings' if use_merged_findings else dataset_name
        findings_criterion = chexpert_criterion if dataset_id == CHEXPERT_DATASET_ID else cxr14_criterion

        if include_image:
            images = batch['i'].to(device)
        if include_visual_features:
            visual_features = batch['vf'].to(device)
        if use_vqa_mode:
            questions = batch['q'].to(device)
            answers = batch['a'].to(device)
            if shift_answer:
                answers_start = answers[:, :-1]
                answers_end = answers[:, 1:]
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                f'{dataset_name}_forward': True,
                'device': device,
            }

            if include_image:
                model_kwargs['raw_images'] = images
            if include_visual_features:
                model_kwargs['visual_features'] = visual_features

            if use_vqa_mode:
                model_kwargs['questions'] = questions
                if training:
                    model_kwargs['mode'] = 'train'
                    if shift_answer:
                        model_kwargs['answers'] = answers_start
                    else:
                        model_kwargs['answers'] = answers
                else:
                    model_kwargs['mode'] = 'eval'
                    if use_beam_search:
                        model_kwargs['beam_search_k'] = beam_search_k
                    if include_answer:
                        model_kwargs['max_answer_length'] = answers.size(1)
                    else:                    
                        model_kwargs['max_answer_length'] = max_answer_length

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)
                
                pred_labels_logits = model_output[f'pred_{findings_name}']
                pred_labels_probs = model_output[f'pred_{findings_name}_probs']
                pred_orientation_logits = model_output['pred_orientation']
                pred_gender_logits = model_output['pred_gender']
                if use_vqa_mode:
                    if training:
                        pred_answer_logits = model_output['pred_answers']
                    else:
                        pred_answers = model_output['pred_answers']

                if training:                    
                    # Compute losses                    
                    labels_loss = findings_criterion(pred_labels_logits, labels.float())
                    orientation_loss = chexpert_aux_criterion(pred_orientation_logits, orientations)
                    gender_loss = chexpert_aux_criterion(pred_gender_logits, genders)                    
                    batch_loss = labels_loss + orientation_loss + gender_loss
                    if use_vqa_mode:
                        # answer_loss = nlg_criterion(pred_answer_logits.view(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        if shift_answer:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers_end.reshape(-1))
                        else:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        batch_loss += answer_loss

                    # Backward pass + optimizer step if training
                    backward_and_optimizer_step(batch_loss)
        
        output = {
            'idxs': idxs,
            'dataset_id': dataset_id,
        }            
        if training:
            output['loss'] = batch_loss.detach()
            
        # dataset-specific labels
        output[dataset_name] = labels.detach().cpu()
        output[f'pred_{dataset_name}'] = (pred_labels_logits.detach() > 0).cpu()
        output[f'pred_{dataset_name}_probs'] = pred_labels_probs.detach().cpu()
        if training:
            output[f'{dataset_name}_loss'] = labels_loss.detach()

        # orientation
        output['orientation'] = orientations.detach()
        output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()        
        if training:
            output['orientation_loss'] = orientation_loss.detach()

        # gender
        output['gender'] = genders.detach()
        output['pred_gender'] = pred_gender_logits.argmax(-1).detach()        
        if training:
            output['gender_loss'] = gender_loss.detach()

        # answers (vqa)
        if use_vqa_mode:
            if training:
                pred_answers = pred_answer_logits.argmax(-1)        
                output['answer_loss'] = answer_loss.detach()            
            output['pred_answers'] = tokenizer.clean_batch(pred_answers.detach())
            output['answers'] = tokenizer.clean_batch(answers.detach())                

        return output

    def step_fn__vinbig(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        vinbig_labels = batch['l'].to(device)
        questions = batch['q'].to(device)
        answers = batch['a'].to(device)
        if shift_answer:
            answers_start = answers[:, :-1]
            answers_end = answers[:, 1:]
        if include_image:
            images = batch['i'].to(device)
        if include_visual_features:
            visual_features = batch['vf'].to(device)

        findings_name = 'findings' if use_merged_findings else 'vinbig'
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'vinbig_forward': True,
                'device': device,
            }

            if include_image:
                model_kwargs['raw_images'] = images
            if include_visual_features:
                model_kwargs['visual_features'] = visual_features
            
            model_kwargs['questions'] = questions
            if training:
                model_kwargs['mode'] = 'train'
                if shift_answer:
                    model_kwargs['answers'] = answers_start
                else:
                    model_kwargs['answers'] = answers
            else:
                model_kwargs['mode'] = 'eval'
                if use_beam_search:
                    model_kwargs['beam_search_k'] = beam_search_k
                if include_answer:
                    model_kwargs['max_answer_length'] = answers.size(1)
                else:                    
                    model_kwargs['max_answer_length'] = max_answer_length

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)
                
                pred_vinbig_logits = model_output[f'pred_{findings_name}']
                pred_vinbig_probs = model_output[f'pred_{findings_name}_probs']
                if training:
                    pred_answer_logits = model_output['pred_answers']
                else:
                    pred_answers = model_output['pred_answers']

                if training:
                    # Compute losses
                    vinbig_loss = vinbig_criterion(pred_vinbig_logits, vinbig_labels.float())                    
                    batch_loss = vinbig_loss
                    if shift_answer:
                        answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers_end.reshape(-1))
                    else:
                        answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                    batch_loss += answer_loss

                    # Backward pass + optimizer step if training
                    backward_and_optimizer_step(batch_loss)
        
        output = {
            'idxs': idxs,
            'dataset_id': dataset_id,
        }            
        if training:
            output['loss'] = batch_loss.detach()
            
        # vinbig labels
        output[f'vinbig_labels'] = vinbig_labels.detach().cpu()
        output[f'pred_vinbig_labels'] = (pred_vinbig_logits.detach() > 0).cpu()
        output[f'pred_vinbig_probs'] = pred_vinbig_probs.detach().cpu()
        if training:
            output[f'vinbig_loss'] = vinbig_loss.detach()

        # answers (vqa)
        if use_chexpert_vqa:
            if training:
                pred_answers = pred_answer_logits.argmax(-1)        
                output['answer_loss'] = answer_loss.detach()            
            output['pred_answers'] = tokenizer.clean_batch(pred_answers.detach())
            output['answers'] = tokenizer.clean_batch(answers.detach())                

        return output

    _mim_iu_datasets = [MIMICCXR_DATASET_ID, IUXRAY_DATASET_ID,
             MIMICCXR_DATASET_ID__CHEXPERT_MODE, IUXRAY_DATASET_ID__CHEXPERT_MODE]

    _chexpert_cxr14_datsets = [CHEXPERT_DATASET_ID, CXR14_DATASET_ID]
    
    def step_fn(unused_engine, batch):
        dataset_id = batch['dataset_id']
        # print(f"step_fn(dataset_id={dataset_id})")
        if dataset_id in _mim_iu_datasets:
            output = step_fn__mimiccxr_iuxray(batch)
        elif dataset_id in _chexpert_cxr14_datsets:
            output = step_fn__chexpert_cxr14(batch)
        elif dataset_id == VINBIG_DATASET_ID:
            output = step_fn__vinbig(batch)
        else: assert False, f'Unknown dataset_id {dataset_id}'
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn

def _get_dataset_masks(dataset_id, labels_remapper, n_labels, device):
    mask = [0] * n_labels
    new_labels = labels_remapper[dataset_id]
    for i in range(len(new_labels)):
        mask[new_labels[i]] = 1
    mask = torch.tensor(mask).to(device)
    return mask

def get_engine(model, tokenizer, classify_tags, classify_orientation, classify_chexpert, classify_questions,
               question_encoding, answer_decoding, device,
               iters_to_accumulate=1,
               binary_loss_name='bce',
               include_image=True, include_visual_features=False,
               shift_answer=False, include_answer=True,
               beam_search_k=None, max_answer_length=None,
               use_amp=False,
               training=False,
               train_with_chexpert_dataset=False,
               train_with_cxr14=False,
               chexpert_mode=None,
               use_vinbig_dataset=False,
               optimizer=None,
               update_lr_batchwise=False, lr_scheduler=None,
               use_merged_findings=False, findings_remapper=None, n_findings=None):
    
    print(f'get_engine(): shift_answer={shift_answer}')
    
    # Criterion
    nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss

    if use_merged_findings and training:
        assert binary_loss_name == 'wbce-c'
        assert findings_remapper is not None
        assert n_findings is not None
    
    # Auxiliary tasks
    if training and classify_tags:
        tags_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        tags_criterion = None
    
    if training and classify_orientation:
        iuxray_orientation_criterion = nn.CrossEntropyLoss()
        mimiccxr_orientation_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore unknown
    else:
        iuxray_orientation_criterion = None
        mimiccxr_orientation_criterion = None
    
    if training and classify_questions:
        question_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        question_criterion = None

    if training and train_with_chexpert_dataset or train_with_cxr14 or train_with_cxr14:
        chexpert_aux_criterion = nn.CrossEntropyLoss()
        assert chexpert_mode is not None
    else:
        chexpert_aux_criterion = None

    if training and classify_chexpert or train_with_chexpert_dataset:
        if use_merged_findings:
            chexpert_mask = _get_dataset_masks(CHEXPERT_DATASET_ID, findings_remapper, n_findings, device)
            chexpert_criterion = get_binary_multilabel_loss(binary_loss_name, classes_mask=chexpert_mask)
        else:
            chexpert_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        chexpert_criterion = None

    if training and train_with_cxr14:
        if use_merged_findings:
            cxr14_mask = _get_dataset_masks(CXR14_DATASET_ID, findings_remapper, n_findings, device)
            cxr14_criterion = get_binary_multilabel_loss(binary_loss_name, classes_mask=cxr14_mask)
        else:
            cxr14_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        cxr14_criterion = None
    
    if training and use_vinbig_dataset:
        if use_merged_findings:
            vinbig_mask = _get_dataset_masks(VINBIG_DATASET_ID, findings_remapper, n_findings, device)
            vinbig_criterion = get_binary_multilabel_loss(binary_loss_name, classes_mask=vinbig_mask)
        else:
            vinbig_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        vinbig_criterion = None

    # Create engine
    step_fn = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                          include_visual_features=include_visual_features,
                          include_image=include_image, include_answer=include_answer,
                          max_answer_length=max_answer_length,
                          training=training,
                          device=device, use_amp=use_amp,
                          question_encoding=question_encoding,
                          answer_decoding=answer_decoding,
                          beam_search_k=beam_search_k,
                          shift_answer=shift_answer,
                          use_merged_findings=use_merged_findings,
                          iters_to_accumulate=iters_to_accumulate,
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
                          chexpert_mode=chexpert_mode,
                          # cxr14 dataset
                          cxr14_criterion=cxr14_criterion,
                          # vinbig dataset
                          vinbig_criterion=vinbig_criterion,
                          # batchwise learning rate updates
                          update_lr_batchwise=update_lr_batchwise,
                          lr_scheduler=lr_scheduler,
                          )
    engine = Engine(step_fn)
    return engine