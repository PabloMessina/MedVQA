from abc import ABC

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):

        # print("entity_logits.shape: ", entity_logits.shape)
        # print("rel_logits.shape: ", rel_logits.shape)
        # print("entity_types.shape: ", entity_types.shape)
        # print("rel_types.shape: ", rel_types.shape)
        # print("entity_sample_masks.shape: ", entity_sample_masks.shape)
        # print("rel_sample_masks.shape: ", rel_sample_masks.shape)

        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss
        
        return train_loss