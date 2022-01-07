import torch

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device):
    
    def step_fn(engine, batch):

        idxs = batch['idx']
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
            'idxs': idxs,
            'loss': batch_loss.detach(),
            'answers': tokenizer.clean_batch(answers.detach()),
            'questions': tokenizer.clean_batch(questions.detach()),
            'pred_answers': tokenizer.clean_batch(pred_answers.detach()),
            'pred_questions': tokenizer.clean_batch(pred_questions.detach()),
        }
    
    return step_fn