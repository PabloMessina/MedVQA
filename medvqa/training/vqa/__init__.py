import torch

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device):
    
    def step_fn(engine, batch):

        images = batch['i'].to(device)
        questions = batch['q'].to(device)
        question_lengths = batch['ql']
        answers = batch['a'].to(device)

        # zero grad if training
        if training:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # forward pass
            if training:
                output = model(images, questions, question_lengths, answers=answers, mode='train')
            else:
                output = model(images, questions, question_lengths, max_answer_length=answers.size(1), mode='eval')                    
            pred_answer_logits, pred_question_logits = output

            # compute losses
            answer_loss = nlg_criterion(pred_answer_logits.view(-1, pred_answer_logits.shape[-1]), answers.view(-1))
            question_loss = nlg_criterion(pred_question_logits.view(-1, pred_question_logits.shape[-1]), questions.view(-1))                    
            batch_loss = answer_loss + question_loss

            # backward pass + optimizer step if training
            if training:
                batch_loss.backward()
                optimizer.step()

        # computed predicted Q & A
        pred_questions = pred_question_logits.argmax(-1)
        pred_answers = pred_answer_logits.argmax(-1)        
        
        return {
            'loss': batch_loss.detach(),
            'answers': tokenizer.clean_batch(answers.detach()),
            'questions': tokenizer.clean_batch(questions.detach()),
            'pred_answers': tokenizer.clean_batch(pred_answers.detach()),
            'pred_questions': tokenizer.clean_batch(pred_questions.detach()),
        }
    
    return step_fn

# def run_training_from_scratch(model, optimizer, lr_scheduler, nlg_criterion, tokenizer, device,
#                               dataloaders, max_epochs, epoch_length_train, epoch_length_val):

#     print('training model from scratch ...')

#     train_step = _get_step_fn(model, optimizer, nlg_criterion, tokenizer,
#                             training=True, device=device)
#     trainer = Engine(train_step)

#     val_step = _get_step_fn(model, optimizer, nlg_criterion, tokenizer,
#                             training=False, device=device)
#     validator = Engine(val_step)

#     train_loader = dataloaders['train']
#     val_loader = dataloaders['val']

#     attach_bleu_question(trainer, device)
#     attach_bleu_question(validator, device)

#     attach_bleu_answer(trainer, device)
#     attach_bleu_answer(validator, device)

#     attach_loss('loss', trainer, device)
#     attach_loss('loss', validator, device)

#     timer = Timer()
#     timer.attach(trainer, start=Events.EPOCH_STARTED)
#     timer.attach(validator, start=Events.EPOCH_STARTED)
    
#     log_metrics_handler = get_log_metrics_handlers(timer, metrics_to_print=[
#         'loss', 'bleu_question', 'bleu_answer'
#     ])
#     log_iteration_handler = get_log_iteration_handler()
#     lr_sch_handler = get_lr_sch_handler(trainer, validator, lr_scheduler, _merge_metrics)

#     trainer.add_event_handler(Events.EPOCH_STARTED, get_log_epoch_started_handler())
#     trainer.add_event_handler(Events.EPOCH_STARTED, lambda : print('(1) Training stage ...'))
#     trainer.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
#     trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
#     trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda : validator.run(val_loader,
#                                      max_epochs=1, epoch_length=epoch_length_val))
#     validator.add_event_handler(Events.EPOCH_STARTED, lambda : print('(2) Validation stage ...'))
#     validator.add_event_handler(Events.ITERATION_STARTED, log_iteration_handler)
#     validator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics_handler)
#     validator.add_event_handler(Events.EPOCH_COMPLETED, lr_sch_handler)

#     trainer.run(train_loader, max_epochs=max_epochs, epoch_length=epoch_length_train)

# def train_model(model, optimizer, nlg_criterion, aux_criterion, scheduler,
#                 dataloaders, epochs, n_batches_per_epoch,
#                 device):

    
#     since = time.time()
#     # smoothing_function = SmoothingFunction().method1  
#     smoothing_function = None

#     if checkpoint:
#         print('resuming training ....')
#         history = checkpoint['history']
#         epoch_one = 1 + len(history['train_loss'])
#         epochs += len(history['train_loss'])
#         best_wts = checkpoint['best_wts']
#         best_val_exact_matches = checkpoint['best_val_exact_matches']
#         best_train_exact_matches = checkpoint['best_train_exact_matches']
#         model.load_state_dict(checkpoint['last_wts'])
#     else:
#         print('training from scratch ....')
#         epoch_one = 1
#         best_wts = copy.deepcopy(model.state_dict())
#         best_val_exact_matches = -1e9
#         best_train_exact_matches = -1e9
#         history = dict()
#         for met in ('loss', 'bleu', 'exact_matches'):
#             for phase in ('train', 'val'):
#                 history['%s_%s' % (phase,met)] = []

#     last_train_epoch_score = 0
#     last_val_epoch_score = 0
    
#     cyclic_train_dataloader = cyclic_dataloader_gen(dataloaders['train'])

#     for epoch in range(epoch_one, epochs+1):
#         print('\n====== [Epoch {}/{}]'.format(epoch, epochs))

#         for phase in ('train', 'val'):

#             print('--------------------- %s ---------------------' % phase)

#             if phase == 'train':
#                 model.train()
#                 dataloader = next_k_iterations_gen(cyclic_train_dataloader, n_batches_per_epoch)
#                 n_steps = n_batches_per_epoch
#             else:
#                 model.eval()
#                 dataloader = dataloaders['val']
#                 n_steps = len(dataloader)
            
#             running_loss = 0.0
#             running_exact_matches = 0
#             running_bleu = 0
#             running_quest_exact_matches = 0
#             running_quest_bleu = 0
#             running_count = 0
#             running_answer_count = 0
#             epoch_since = time.time()

#             with torch.set_grad_enabled(phase == 'train'):

#                 for batch_idx, batch_dict in enumerate(dataloader):

#                     images = batch_dict['i'].to(DEVICE)
#                     questions = batch_dict['q'].to(DEVICE)
#                     question_lengths = batch_dict['ql']
#                     answers = batch_dict['a'].to(DEVICE)
#                     batch_size = questions.size(0)

#                     optimizer.zero_grad()

#                     # forward pass
#                     if phase == 'train':
#                         output = model(images, questions, question_lengths, answers=answers, mode='train')
#                     else:
#                         output = model(images, questions, question_lengths, max_answer_length=answers.size(1), mode='eval')
                    
#                     pred_answers, pred_questions = output

#                     answer_loss = nlp_criterion(pred_answers.view(-1, pred_answers.shape[-1]), answers.view(-1))
#                     question_loss = nlp_criterion(pred_questions.view(-1, pred_questions.shape[-1]), questions.view(-1))
                    
#                     final_loss = answer_loss + question_loss

#                     total_bleu, exact_matches, acount = compute_nlp_metrics(answers.cpu().numpy(), pred_answers.argmax(-1).cpu().numpy(), smoothing_function)
#                     quest_bleu, quest_exact_matches, _ = compute_nlp_metrics(questions.cpu().numpy(), pred_questions.argmax(-1).cpu().numpy(), smoothing_function)

#                     # backward pass + optimization only if in training phase
#                     if phase == 'train':
#                         final_loss.backward()
#                         # clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
#                         optimizer.step()

#                     # statistics
#                     running_loss += final_loss.item() * batch_size
                    
#                     running_answer_count += acount
#                     running_exact_matches += exact_matches
#                     running_bleu += total_bleu
                    
#                     running_quest_exact_matches += quest_exact_matches
#                     running_quest_bleu += quest_bleu
                    
#                     running_count += batch_size

#                     if ((batch_idx + 1) % 20 == 0 or batch_idx + 1 == n_steps):
#                         elapsed_time = time.time() - epoch_since
#                         print("Batch: %d/%d, running_loss=%.5f, bleu=%.5f, exact=%.5f, qbleu=%.5f, qexact=%.5f, elapsed_time=%.0fm %.0fs" % (
#                             batch_idx+1, n_steps,
#                             running_loss/running_count,
#                             running_bleu/running_answer_count,
#                             running_exact_matches/running_answer_count,
#                             running_quest_bleu/running_count,
#                             running_quest_exact_matches/running_count,
#                             elapsed_time // 60, elapsed_time % 60,
#                         ), end="\r")
#                         # ))
#             print()

#             epoch_loss = running_loss / running_count
#             epoch_bleu = running_bleu / running_answer_count
#             epoch_exact_matches = running_exact_matches / running_answer_count
#             epoch_quest_bleu = running_quest_bleu / running_count
#             epoch_quest_exact_matches = running_quest_exact_matches / running_count

#             epoch_score = epoch_bleu + epoch_quest_bleu
#             if phase == 'train':
#                 last_train_epoch_score = epoch_score
#             else:
#                 last_val_epoch_score = epoch_score

#             history['%s_loss' % phase].append(epoch_loss)
#             history['%s_bleu' % phase].append(epoch_bleu)
#             history['%s_exact_matches' % phase].append(epoch_exact_matches)

#             if phase == 'val':
#                 scheduler.step(last_train_epoch_score * 0.8 + last_val_epoch_score * 0.2) # decay learning rate if necessary
#                 if epoch_exact_matches > best_val_exact_matches or (
#                       epoch_exact_matches == best_val_exact_matches and\
#                       last_train_exact_matches > best_train_exact_matches): # improvement detected!
#                     # update best val and train exact_matches
#                     best_val_exact_matches = epoch_exact_matches
#                     best_train_exact_matches = last_train_exact_matches
#                     # update best weights
#                     best_wts = copy.deepcopy(model.state_dict())
#                     print('\t*** improvement detected! best_val_exact_matches=%f' % best_val_exact_matches)
#             else:
#                 last_train_exact_matches = epoch_exact_matches

#     print()
#     elapsed_time = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#       elapsed_time // 60, elapsed_time % 60))
#     print('Best val exact_matches: {:4f}'.format(best_val_exact_matches))

#     return dict(
#         best_wts=best_wts,
#         last_wts=copy.deepcopy(model.state_dict()),
#         history=history,
#         best_val_exact_matches=best_val_exact_matches,
#         best_train_exact_matches=best_train_exact_matches,
#     )