import  os
import argparse
import torch
import ast
import logging

from medvqa.datasets.fact_embedding.fact_embedding_dataset_management import (
    _LABEL_TO_CATEGORY,
    _LABEL_TO_COMPARISON_STATUS,
    _LABEL_TO_HEALTH_STATUS,
    FactEmbeddingTrainer,
)
from medvqa.datasets.radgraph import RADGRAPH_CONLLFORMAT_ENTITY_TYPE_COUNT, RADGRAPH_CONLLFORMAT_RELATION_TYPE_COUNT
from medvqa.losses.optimizers import create_optimizer
from medvqa.losses.schedulers import create_lr_scheduler
from medvqa.models.nlp.fact_encoder import FactEncoder, HuggingfaceModels

from medvqa.training.utils import append_metric_name, run_common_boilerplate_code_and_start_training
from medvqa.utils.common import WORKSPACE_DIR, DictWithDefault
from medvqa.metrics import (
    attach_condition_aware_logits_above_threshold_accuracy,
    attach_condition_aware_loss,
    attach_condition_aware_multilabel_f1score,
    attach_condition_aware_triplet_accuracy,
    attach_condition_aware_accuracy,
)
from medvqa.models.checkpoint import load_metadata

from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.files_utils import get_checkpoint_folder_path
from medvqa.training.fact_embedding import get_engine
from medvqa.datasets.dataloading_utils import (
    balanced_dataloaders_generator,
    get_fact_embedding_collate_batch_fn,
    multi_cyclic_dataloaders_generator,
)
from medvqa.metrics.utils import get_merge_metrics_fn
from medvqa.utils.logging_utils import log_title, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class _TripletRuleWeightsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            parsed_dict = ast.literal_eval(values)
            try:
                # Check that parsed_dict is a dictionary mapping string keys to non-empty lists of floats
                assert isinstance(parsed_dict, dict)
                for key, value in parsed_dict.items():
                    assert isinstance(key, str)
                    assert isinstance(value, list)
                    assert len(value) > 0
                    for v in value:
                        assert isinstance(v, float)
                # If all checks pass, set the attribute
                setattr(namespace, self.dest, parsed_dict)
            except AssertionError:
                # If any of the checks fail, raise an error
                raise ValueError("Invalid input format. Must be a dictionary mapping string keys to non-empty lists of floats")
        except (ValueError, SyntaxError) as e:
            raise ValueError("Invalid input format. Error: {}".format(str(e)))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # --- Required arguments

    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs the model will be trained')
    parser.add_argument('--batches_per_epoch', type=int, required=True, help='Number of batches per epoch')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--train_metrics_weight', type=float, default=0.1, help='Weight for training metrics') # 0.1 for training metrics, 0.9 for validation metrics

    # --- Other arguments

    parser.add_argument('--val_batch_size', type=int, default=None, help='Batch size for validation')
    parser.add_argument('--radgraph_spert_batch_size', type=int, default=None, help='Batch size for RadGraph NER/RE')
    parser.add_argument('--checkpoint_folder', type=str, default=None,
                        help='Relative path to folder with checkpoint to resume training from')

    # Model arguments
    parser.add_argument('--huggingface_model_name', type=str, default=None, choices=HuggingfaceModels.get_all())
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--n_chest_imagenome_observations', type=int, default=None)
    parser.add_argument('--n_chest_imagenome_anatomical_locations', type=int, default=None)
    parser.add_argument('--use_aux_task_hidden_layer', action='store_true', default=False)
    parser.add_argument('--aux_task_hidden_layer_size', type=int, default=None)
    parser.add_argument('--nli_hidden_layer_size', type=int, default=None)
    parser.add_argument('--pretrained_checkpoint_folder_path', type=str, default=None)
    parser.add_argument('--pretrained_checkpoint_folder_paths', type=str, nargs='+', default=None)
    parser.add_argument('--freeze_huggingface_model', action='store_true', default=False)
    parser.add_argument('--spert_size_embedding', type=int, default=25)
    parser.add_argument('--spert_max_pairs', type=int, default=1000)
    parser.add_argument('--spert_prop_drop', type=float, default=0.1)
    parser.add_argument('--fact_decoder_embed_size', type=int, default=256)
    parser.add_argument('--fact_decoder_hidden_size', type=int, default=256)
    parser.add_argument('--fact_decoder_nhead', type=int, default=1)
    parser.add_argument('--fact_decoder_dim_feedforward', type=int, default=256)
    parser.add_argument('--fact_decoder_num_layers', type=int, default=1)
    
    # Optimization arguments
    parser.add_argument('--optimizer_name', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='reduce-lr-on-plateau')
    parser.add_argument('--lr_decay', type=float, default=0.76, help='Learning rate decay')
    parser.add_argument('--lr_decay_patience', type=int, default=2, help='Learning rate decay patience')
    parser.add_argument('--warmup_and_decay_args', type=str, default=None)
    parser.add_argument('--warmup_and_cosine_args', type=str, default=None)
    parser.add_argument('--warmup_decay_and_cyclic_decay_args', type=str, default=None)
    parser.add_argument('--iters_to_accumulate', type=int, default=1, help='For gradient accumulation')
    parser.add_argument('--override_lr', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', type=float, default=None, help='Max gradient norm')
    # Loss weights
    parser.add_argument('--triplet_loss_weight', type=float, default=1.0)
    parser.add_argument('--category_classif_loss_weight', type=float, default=1.0)
    parser.add_argument('--health_status_classif_loss_weight', type=float, default=1.0)
    parser.add_argument('--comparison_status_classif_loss_weight', type=float, default=1.0)
    parser.add_argument('--chest_imagenome_obs_classif_loss_weight', type=float, default=1.0)
    parser.add_argument('--chest_imagenome_anatloc_classif_loss_weight', type=float, default=1.0)
    parser.add_argument('--nli_loss_weight', type=float, default=1.0)
    parser.add_argument('--entcon_loss_weight', type=float, default=1.0)
    parser.add_argument('--sentence_autoencoder_loss_weight', type=float, default=1.0)

    # Dataset and dataloading arguments
    parser.add_argument('--triplets_filepath', type=str, default=None, help='Path to triplets file')
    parser.add_argument('--triplet_rule_weights', type=str, action=_TripletRuleWeightsAction, help='Weights for triplet rules')
    parser.add_argument('--integrated_facts_metadata_jsonl_filepath', type=str, default=None, help='Path to integrated facts metadata file')
    parser.add_argument('--paraphrases_jsonl_filepaths', type=str, nargs='+', default=None, help='Path to paraphrases files')
    parser.add_argument('--integrated_chest_imagenome_observations_filepath', type=str, default=None)
    parser.add_argument('--integrated_chest_imagenome_anatomical_locations_filepath', type=str, default=None)
    parser.add_argument('--integrated_nli_jsonl_filepath', type=str, default=None)
    parser.add_argument('--integrated_sentence_facts_jsonl_filepath', type=str, default=None)
    parser.add_argument('--sentences_and_cluster_ids_filepath', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset to use')
    parser.add_argument('--triplets_weight', type=float, default=0, help='Weight for triplet sampling during training')
    parser.add_argument('--metadata_classification_weight', type=float, default=0,
                        help='Weight for metadata classification sampling during training')
    parser.add_argument('--chest_imagenome_observations_classification_weight', type=float, default=0,
                        help='Weight for chest imagenome observations classification sampling during training')
    parser.add_argument('--chest_imagenome_anatomical_locations_classification_weight', type=float, default=0,
                        help='Weight for chest imagenome anatomical locations classification sampling during training')
    parser.add_argument('--nli_weight', type=float, default=0, help='Weight for NLI sampling during training')
    parser.add_argument('--entcon_weight', type=float, default=0, help='Weight for entailment/contradiction sampling during training')
    parser.add_argument('--radgraph_ner_re_weight', type=float, default=0, help='Weight for RadGraph NER/RE sampling during training')
    parser.add_argument('--sentence_autoencoder_weight', type=float, default=0, help='Weight for sentence autoencoder sampling during training')
    parser.add_argument('--use_nli_val_in_train', action='store_true', default=False, help='Use NLI validation set in training')
    parser.add_argument('--use_anli', action='store_true', default=False)
    parser.add_argument('--use_multinli', action='store_true', default=False)
    parser.add_argument('--use_snli', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for parallel dataloading')    
    parser.add_argument('--device', type=str, default='GPU', help='Device to use (GPU or CPU)')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # Checkpoint saving arguments
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no_save', dest='save', action='store_false')
    parser.set_defaults(save=True)
    
    return parser.parse_args(args=args)

_METRIC_WEIGHTS = DictWithDefault(default=1.0) # Default weight is 1.0
_METRIC_WEIGHTS['nli_acc'] = 1.5 # NLI accuracy is more important than other metrics
_METRIC_WEIGHTS['entcon_acc'] = 1.5 # Entailment/contradiction accuracy is more important than other metrics

def _metric_getter(metrics_dict, key):
    if '_loss' in key:
        return 1 / (1 + metrics_dict[key]) # convert loss to score
    return metrics_dict[key]
   
def train_model(
        model_kwargs,
        optimizer_kwargs,
        lr_scheduler_kwargs,
        fact_embedding_trainer_kwargs,
        dataloading_kwargs,
        collate_batch_fn_kwargs,
        training_kwargs,
        trainer_engine_kwargs,
        validator_engine_kwargs,
        epochs,
        batches_per_epoch,
        num_workers,
        device='GPU',
        checkpoint_folder_path=None,
        save=True,
        override_lr=False,
        ):
    
    # Pull out some args from kwargs
    batch_size = dataloading_kwargs['batch_size']
    val_batch_size = dataloading_kwargs['val_batch_size']
    if val_batch_size is None:
        val_batch_size = batch_size
    radgraph_spert_batch_size = dataloading_kwargs['radgraph_spert_batch_size']
    if radgraph_spert_batch_size is None:
        radgraph_spert_batch_size = batch_size
    use_triplets = dataloading_kwargs['triplets_weight'] > 0
    use_metadata = dataloading_kwargs['metadata_classification_weight'] > 0
    use_chest_imagenome_observations = dataloading_kwargs['chest_imagenome_observations_classification_weight'] > 0
    use_chest_imagenome_anatlocs = dataloading_kwargs['chest_imagenome_anatomical_locations_classification_weight'] > 0
    use_nli = dataloading_kwargs['nli_weight'] > 0
    use_entcon = dataloading_kwargs['entcon_weight'] > 0
    use_radgraph_ner_re = dataloading_kwargs['radgraph_ner_re_weight'] > 0
    use_sentence_autoencoder = dataloading_kwargs['sentence_autoencoder_weight'] > 0
    assert sum([use_triplets, use_metadata, use_chest_imagenome_observations,
                use_chest_imagenome_anatlocs, use_nli, use_entcon, use_radgraph_ner_re,
                use_sentence_autoencoder]) > 0
    train_metrics_weight = training_kwargs.get('train_metrics_weight', 0.1)
    assert 0 <= train_metrics_weight <= 1
    logger.info(f'train_metrics_weight = {train_metrics_weight}')
    
    # Define collate batch functions
    log_title(logger, 'Defining collate batch functions')
    triplet_collate_batch_fn = get_fact_embedding_collate_batch_fn(**collate_batch_fn_kwargs['triplets']) \
        if use_triplets else None
    metadata_classification_collate_batch_fn = get_fact_embedding_collate_batch_fn(
        **collate_batch_fn_kwargs['metadata_classification']) if use_metadata else None
    chest_imagenome_observation_collate_batch_fn = get_fact_embedding_collate_batch_fn(
        **collate_batch_fn_kwargs['chest_imagenome_observation_classification']) \
        if use_chest_imagenome_observations else None
    chest_imagenome_anatomical_location_collate_batch_fn = get_fact_embedding_collate_batch_fn(
        **collate_batch_fn_kwargs['chest_imagenome_anatomical_location_classification']) \
        if use_chest_imagenome_anatlocs else None
    nli_collate_batch_fn = get_fact_embedding_collate_batch_fn(
        **collate_batch_fn_kwargs['nli']) if use_nli else None
    entcon_collate_batch_fn = get_fact_embedding_collate_batch_fn(
        **collate_batch_fn_kwargs['entcon']) if use_entcon else None
    sentence_autoencoder_collate_batch_fn = get_fact_embedding_collate_batch_fn(
        **collate_batch_fn_kwargs['sentence_autoencoder']) if use_sentence_autoencoder else None
    
    # Create dataloaders
    if use_radgraph_ner_re:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_kwargs['huggingface_model_name'], trust_remote_code=True)
    else:
        tokenizer = None
    fact_embedding_trainer = FactEmbeddingTrainer(
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        radgraph_spert_batch_size=radgraph_spert_batch_size,
        triplet_collate_batch_fn=triplet_collate_batch_fn,
        metadata_classification_collate_batch_fn=metadata_classification_collate_batch_fn,
        chest_imagenome_observation_collate_batch_fn=chest_imagenome_observation_collate_batch_fn,
        chest_imagenome_anatomical_location_collate_batch_fn=chest_imagenome_anatomical_location_collate_batch_fn,
        nli_collate_batch_fn=nli_collate_batch_fn,
        entcon_collate_batch_fn=entcon_collate_batch_fn,
        sentence_autoencoder_collate_batch_fn=sentence_autoencoder_collate_batch_fn,
        num_workers=num_workers,
        tokenizer=tokenizer,
        **fact_embedding_trainer_kwargs,
    )
    
    _train_weights = []
    _train_dataloaders = []
    _val_dataloaders = []
    if use_triplets:
        _train_dataloaders.append(fact_embedding_trainer.train_triplet_dataloader)
        _train_weights.append(dataloading_kwargs['triplets_weight'])
        _val_dataloaders.append(fact_embedding_trainer.val_triplets_dataloader)
    if use_metadata:
        _train_dataloaders.append(fact_embedding_trainer.train_metadata_classification_dataloader)
        _train_weights.append(dataloading_kwargs['metadata_classification_weight'])
    if use_chest_imagenome_observations:
        _train_dataloaders.append(fact_embedding_trainer.train_chest_imagenome_observations_dataloader)
        _train_weights.append(dataloading_kwargs['chest_imagenome_observations_classification_weight'])
    if use_chest_imagenome_anatlocs:
        _train_dataloaders.append(fact_embedding_trainer.train_chest_imagenome_anatomical_locations_dataloader)
        _train_weights.append(dataloading_kwargs['chest_imagenome_anatomical_locations_classification_weight'])
    if use_nli:
        _train_dataloaders.append(fact_embedding_trainer.train_nli_dataloader)
        _train_weights.append(dataloading_kwargs['nli_weight'])
        _val_dataloaders.append(fact_embedding_trainer.val_nli_dataloader)
    if use_entcon:
        _train_dataloaders.append(fact_embedding_trainer.train_entcon_dataloader)
        _train_weights.append(dataloading_kwargs['entcon_weight'])
        _val_dataloaders.append(fact_embedding_trainer.val_entcon_dataloader)
    if use_radgraph_ner_re:
        _train_dataloaders.append(fact_embedding_trainer.radgraph_spert_train_dataloader)
        _train_weights.append(dataloading_kwargs['radgraph_ner_re_weight'])
    if use_sentence_autoencoder:
        _train_dataloaders.append(fact_embedding_trainer.train_sentence_autoencoder_dataloader)
        _train_weights.append(dataloading_kwargs['sentence_autoencoder_weight'])
        _val_dataloaders.append(fact_embedding_trainer.val_sentence_autoencoder_dataloader)
    
    assert len(_train_dataloaders) > 0
    assert len(_train_weights) == len(_train_dataloaders)
    assert len(_val_dataloaders) > 0
    logger.info(f'len(_train_dataloaders) = {len(_train_dataloaders)}')
    logger.info(f'len(_train_weights) = {len(_train_weights)}')
    logger.info(f'_train_weights = {_train_weights}')
    logger.info(f'len(_val_dataloaders) = {len(_val_dataloaders)}')
    
    train_dataloader = balanced_dataloaders_generator(_train_dataloaders, _train_weights)

    val_dataloader = multi_cyclic_dataloaders_generator(_val_dataloaders)
    val_dataloader_size = sum(len(d) for d in _val_dataloaders)

    trainer_name = fact_embedding_trainer.name
    logger.info(f'fact_embedding_trainer.name = {trainer_name}')

    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    logger.info(f'device = {device}')

    # Create model
    log_title(logger, 'Creating instance of FactEncoder')
    if use_sentence_autoencoder: # if using sentence autoencoder, we need to pass special arguments to FactEncoder
        _tokenizer = fact_embedding_trainer.sentence_decoder_tokenizer
        model_kwargs['fact_decoder_start_idx'] = _tokenizer.token2id[_tokenizer.START_TOKEN]
        model_kwargs['fact_decoder_vocab_size']  = _tokenizer.vocab_size
    model = FactEncoder(**model_kwargs)
    model = model.to(device)

    # Optimizer
    log_title(logger, 'Creating optimizer')
    optimizer = create_optimizer(params=model.parameters(), **optimizer_kwargs)

    # Learning rate scheduler
    log_title(logger, 'Creating learning rate scheduler')
    lr_scheduler, update_lr_batchwise = create_lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)

    # Create trainer and validator engines
    log_title(logger, 'Creating trainer and validator engines')
    trainer_engine = get_engine(model=model, optimizer=optimizer, device=device, 
                                update_lr_batchwise=update_lr_batchwise, lr_scheduler=lr_scheduler, **trainer_engine_kwargs)
    validator_engine = get_engine(model=model, device=device, **validator_engine_kwargs)
    
    # Attach metrics, losses, timer and events to engines
    log_title(logger, 'Attaching metrics, losses, timer and events to engines')

    train_metrics_to_merge = []
    val_metrics_to_merge = []
    metrics_to_print = []
    
    attach_condition_aware_loss(trainer_engine, 'loss')
    metrics_to_print.append('loss')

    # Attach metrics for triplet ranking
    if use_triplets:
        attach_condition_aware_loss(trainer_engine, 'triplet_loss', condition_function=lambda x: x['flag'] == 't')
        metrics_to_print.append('triplet_loss')
        for rule_id in fact_embedding_trainer.rule_ids:
            metric_name = f'tacc({rule_id})'
            attach_condition_aware_triplet_accuracy(validator_engine, rule_id, metric_name, condition_function=lambda x: x['flag'] == 't')
            append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, metric_name, train=False)

    # Attach metrics for metadata classification
    if use_metadata:
        _cond_func = lambda x: x['flag'] == 'mc' # only consider metadata classification samples
        
        attach_condition_aware_loss(trainer_engine, 'c_loss', condition_function=_cond_func)
        metrics_to_print.append('c_loss') # category classification loss
        attach_condition_aware_loss(trainer_engine, 'hs_loss', condition_function=_cond_func)
        metrics_to_print.append('hs_loss') # health status classification loss
        attach_condition_aware_loss(trainer_engine, 'cs_loss', condition_function=_cond_func)
        metrics_to_print.append('cs_loss') # comparison status classification loss

        attach_condition_aware_accuracy(trainer_engine, pred_field_name='pred_category', gt_field_name='gt_category',
                                        condition_function=_cond_func, metric_name='cacc')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'cacc', val=False)

        attach_condition_aware_accuracy(trainer_engine, pred_field_name='pred_health_status', gt_field_name='gt_health_status',
                                        condition_function=_cond_func, metric_name='hsacc')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'hsacc', val=False)

        attach_condition_aware_accuracy(trainer_engine, pred_field_name='pred_comparison_status', gt_field_name='gt_comparison_status',
                                        condition_function=_cond_func, metric_name='csacc')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'csacc', val=False)

    # Attach metrics for chest imagenome observation classification
    if use_chest_imagenome_observations:
        _cond_func = lambda x: x['flag'] == 'cioc' # only consider chest imagenome observation classification samples
        attach_condition_aware_loss(trainer_engine, 'chstimgn_loss', condition_function=_cond_func, metric_name='chstimgn_obs_loss')
        metrics_to_print.append('chstimgn_obs_loss') # chest imagenome observation classification loss
        attach_condition_aware_multilabel_f1score(trainer_engine, pred_field_name='pred_labels', gt_field_name='gt_labels',
                                                condition_function=_cond_func, metric_name='chestimg_obs_macf1')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chestimg_obs_macf1', val=False)

    # Attach metrics for chest imagenome anatomical location classification
    if use_chest_imagenome_anatlocs:
        _cond_func = lambda x: x['flag'] == 'cialc' # only consider chest imagenome anatomical location classification samples
        attach_condition_aware_loss(trainer_engine, 'chstimgn_loss', condition_function=_cond_func, metric_name='chstimgn_anatloc_loss')
        metrics_to_print.append('chstimgn_anatloc_loss') # chest imagenome anatomical location classification loss
        attach_condition_aware_multilabel_f1score(trainer_engine, pred_field_name='pred_labels', gt_field_name='gt_labels',
                                                condition_function=_cond_func, metric_name='chestimg_anatloc_macf1')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'chestimg_anatloc_macf1', val=False)

    # Attach metrics for NLI
    if use_nli:
        _cond_func = lambda x: x['flag'] == 'nli' # only consider NLI samples
        attach_condition_aware_loss(trainer_engine, 'nli_loss', condition_function=_cond_func)
        metrics_to_print.append('nli_loss') # NLI loss
        attach_condition_aware_accuracy(trainer_engine, pred_field_name='pred_labels', gt_field_name='gt_labels',
                                        condition_function=_cond_func, metric_name='nli_acc')
        attach_condition_aware_accuracy(validator_engine, pred_field_name='pred_labels', gt_field_name='gt_labels',
                                        condition_function=_cond_func, metric_name='nli_acc')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'nli_acc')
        
    # Attach metrics for entailment/contradiction
    if use_entcon:
        _cond_func = lambda x: x['flag'] == 'entcon' # only consider entailment/contradiction samples
        attach_condition_aware_loss(trainer_engine, 'entcon_loss', condition_function=_cond_func)
        metrics_to_print.append('entcon_loss') # entailment/contradiction loss
        attach_condition_aware_logits_above_threshold_accuracy(trainer_engine, logits_field_name='logits', 
                                                               condition_function=_cond_func, metric_name='entcon_acc')
        attach_condition_aware_logits_above_threshold_accuracy(validator_engine, logits_field_name='logits', 
                                                               condition_function=_cond_func, metric_name='entcon_acc')
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'entcon_acc')

    # Attach metrics for RadGraph NER/RE
    if use_radgraph_ner_re:
        _cond_func = lambda x: x['flag'] == 'spert'
        attach_condition_aware_loss(trainer_engine, 'spert_loss', condition_function=_cond_func)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'spert_loss', val=False)

    # Attach metrics for sentence autoencoder
    if use_sentence_autoencoder:
        _cond_func = lambda x: x['flag'] == 'sae'
        attach_condition_aware_loss(trainer_engine, 'sae_loss', condition_function=_cond_func)
        attach_condition_aware_loss(validator_engine, 'sae_loss', condition_function=_cond_func)
        append_metric_name(train_metrics_to_merge, val_metrics_to_merge, metrics_to_print, 'sae_loss')

    # Score function
    if use_triplets:
        triplet_rule_weights = fact_embedding_trainer.triplet_rule_weights
        for i, w in enumerate(triplet_rule_weights['anatomical_locations']):
            rule_id = f'al{i}'
            assert rule_id in fact_embedding_trainer.rule_ids, f'{rule_id} not in {fact_embedding_trainer.rule_ids}'
            _METRIC_WEIGHTS[f'tacc({rule_id})'] = w
        for i, w in enumerate(triplet_rule_weights['observations']):
            rule_id = f'ob{i}'
            assert rule_id in fact_embedding_trainer.rule_ids, f'{rule_id} not in {fact_embedding_trainer.rule_ids}'
            _METRIC_WEIGHTS[f'tacc({rule_id})'] = w
    assert len(val_metrics_to_merge) > 0
    if len(train_metrics_to_merge) > 0:
        val_metrics_weight = 1 - train_metrics_weight
        assert 0 <= val_metrics_weight <= 1
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS,
                                                train_metrics_weight, val_metrics_weight, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(trainer_engine.state.metrics, validator_engine.state.metrics)
    else:
        merge_metrics_fn = get_merge_metrics_fn(train_metrics_to_merge, val_metrics_to_merge, _METRIC_WEIGHTS, 0, 1, _metric_getter)
        score_fn = lambda _ : merge_metrics_fn(validator_engine.state.metrics)

    # Run common boilerplate code and start training
    run_common_boilerplate_code_and_start_training(
        update_lr_batchwise=update_lr_batchwise,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        score_fn=score_fn,
        model=model,
        optimizer=optimizer,
        save=save,
        checkpoint_folder_path=checkpoint_folder_path,
        build_custom_checkpoint_folder_path=lambda: get_checkpoint_folder_path('fact_embedding', trainer_name, model.get_name()),
        metadata_kwargs=dict(
            model_kwargs=model_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            fact_embedding_trainer_kwargs=fact_embedding_trainer_kwargs,
            dataloading_kwargs=dataloading_kwargs,
            collate_batch_fn_kwargs=collate_batch_fn_kwargs,
            training_kwargs=training_kwargs,
            trainer_engine_kwargs=trainer_engine_kwargs,
            validator_engine_kwargs=validator_engine_kwargs,
        ),
        device=device,
        trainer_engine=trainer_engine,
        validator_engine=validator_engine,
        train_metrics_to_merge=train_metrics_to_merge,
        val_metrics_to_merge=val_metrics_to_merge,
        metrics_to_print=metrics_to_print,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        val_dataloader_size=val_dataloader_size,
        model_kwargs=model_kwargs,
        override_lr=override_lr,
    )

def train_from_scratch(
    # Model args
    huggingface_model_name,
    embedding_size,
    n_chest_imagenome_observations,
    n_chest_imagenome_anatomical_locations,
    pretrained_checkpoint_folder_path,
    pretrained_checkpoint_folder_paths,
    freeze_huggingface_model,
    use_aux_task_hidden_layer,
    aux_task_hidden_layer_size,
    nli_hidden_layer_size,
    spert_size_embedding,
    spert_max_pairs,
    spert_prop_drop,
    fact_decoder_embed_size,
    fact_decoder_hidden_size,
    fact_decoder_nhead,
    fact_decoder_dim_feedforward,
    fact_decoder_num_layers,
    # Optimizer args
    optimizer_name,
    lr,
    max_grad_norm,
    # lr_scheduler args
    scheduler,
    lr_decay,
    lr_decay_patience,
    warmup_and_decay_args,
    warmup_and_cosine_args,
    warmup_decay_and_cyclic_decay_args,
    # Dataset args
    triplets_filepath,
    triplet_rule_weights,
    integrated_facts_metadata_jsonl_filepath,
    paraphrases_jsonl_filepaths,
    integrated_chest_imagenome_observations_filepath,
    integrated_chest_imagenome_anatomical_locations_filepath,
    integrated_nli_jsonl_filepath,
    integrated_sentence_facts_jsonl_filepath,
    sentences_and_cluster_ids_filepath,
    dataset_name,
    triplets_weight,
    metadata_classification_weight,
    chest_imagenome_observations_classification_weight,
    chest_imagenome_anatomical_locations_classification_weight,
    nli_weight,
    entcon_weight,
    radgraph_ner_re_weight,
    sentence_autoencoder_weight,
    use_nli_val_in_train,
    use_anli,
    use_multinli,
    use_snli,
    # Dataloading args
    batch_size,
    val_batch_size,
    radgraph_spert_batch_size,
    num_workers,
    # Fixed traning args
    use_amp,
    iters_to_accumulate,
    train_metrics_weight,
    # Loss weights
    triplet_loss_weight,
    category_classif_loss_weight,
    health_status_classif_loss_weight,
    comparison_status_classif_loss_weight,
    chest_imagenome_obs_classif_loss_weight,
    chest_imagenome_anatloc_classif_loss_weight,
    nli_loss_weight,
    entcon_loss_weight,
    sentence_autoencoder_loss_weight,
    # Variable traning args
    epochs,
    batches_per_epoch,
    # GPU
    device,
    # Other args
    save,
):
    log_title(logger, 'Training model from scratch')

    # Define boolean flags
    use_triplets = triplets_weight > 0
    use_metadata_classification = metadata_classification_weight > 0
    use_chest_imagenome_observations_classification = chest_imagenome_observations_classification_weight > 0
    use_chest_imagenome_anatomical_locations_classification = chest_imagenome_anatomical_locations_classification_weight > 0
    use_nli = nli_weight > 0
    use_entcon = entcon_weight > 0
    use_radgraph_ner_re = radgraph_ner_re_weight > 0
    use_sentence_autoencoder = sentence_autoencoder_weight > 0
    
    model_kwargs = dict(
        huggingface_model_name=huggingface_model_name,
        embedding_size=embedding_size,
        pretrained_checkpoint_folder_path=pretrained_checkpoint_folder_path,
        pretrained_checkpoint_folder_paths=pretrained_checkpoint_folder_paths,
        freeze_huggingface_model=freeze_huggingface_model,
        classify_category=use_metadata_classification,
        n_categories=len(_LABEL_TO_CATEGORY),
        classify_health_status=use_metadata_classification,
        n_health_statuses=len(_LABEL_TO_HEALTH_STATUS),
        classify_comparison_status=use_metadata_classification,
        n_comparison_statuses=len(_LABEL_TO_COMPARISON_STATUS),
        classify_chest_imagenome_obs=use_chest_imagenome_observations_classification,
        n_chest_imagenome_observations=n_chest_imagenome_observations,
        classify_chest_imagenome_anatloc=use_chest_imagenome_anatomical_locations_classification,
        use_aux_task_hidden_layer=use_aux_task_hidden_layer,
        aux_task_hidden_layer_size=aux_task_hidden_layer_size,
        n_chest_imagenome_anatomical_locations=n_chest_imagenome_anatomical_locations,
        do_nli=use_nli,
        nli_hidden_layer_size=nli_hidden_layer_size,
        use_fact_decoder=use_sentence_autoencoder,
        fact_decoder_embed_size=fact_decoder_embed_size,
        fact_decoder_hidden_size=fact_decoder_hidden_size,
        fact_decoder_nhead=fact_decoder_nhead,
        fact_decoder_dim_feedforward=fact_decoder_dim_feedforward,
        fact_decoder_num_layers=fact_decoder_num_layers,
    )
    if use_radgraph_ner_re:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name, trust_remote_code=True)
        _spert_cls_token = _tokenizer.convert_tokens_to_ids('[CLS]')        
        logger.info(f'_spert_cls_token = {_spert_cls_token}')
        model_kwargs['use_spert'] = True
        model_kwargs['spert_size_embedding'] = spert_size_embedding
        model_kwargs['spert_relation_types'] = RADGRAPH_CONLLFORMAT_RELATION_TYPE_COUNT
        model_kwargs['spert_entity_types'] = RADGRAPH_CONLLFORMAT_ENTITY_TYPE_COUNT
        model_kwargs['spert_max_pairs'] = spert_max_pairs
        model_kwargs['spert_prop_drop'] = spert_prop_drop
        model_kwargs['spert_cls_token'] = _spert_cls_token

    optimizer_kwargs = dict(
        name=optimizer_name,
        lr=lr,
    )

    lr_scheduler_kwargs = dict(
        name=scheduler,
        factor=lr_decay,
        patience=lr_decay_patience,
        warmup_and_decay_args=warmup_and_decay_args,
        warmup_and_cosine_args=warmup_and_cosine_args,
        warmup_decay_and_cyclic_decay_args=warmup_decay_and_cyclic_decay_args,
        n_batches_per_epoch=batches_per_epoch,
    )
    
    dataloading_kwargs = dict(
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        radgraph_spert_batch_size=radgraph_spert_batch_size,
        triplets_weight=triplets_weight,
        metadata_classification_weight=metadata_classification_weight,
        chest_imagenome_observations_classification_weight=chest_imagenome_observations_classification_weight,
        chest_imagenome_anatomical_locations_classification_weight=chest_imagenome_anatomical_locations_classification_weight,
        nli_weight=nli_weight,
        entcon_weight=entcon_weight,
        radgraph_ner_re_weight=radgraph_ner_re_weight,
        sentence_autoencoder_weight=sentence_autoencoder_weight,
    )
    
    fact_embedding_trainer_kwargs = dict(
        triplets_filepath=triplets_filepath,
        triplet_rule_weights=triplet_rule_weights,
        integrated_facts_metadata_jsonl_filepath=integrated_facts_metadata_jsonl_filepath,
        paraphrases_jsonl_filepaths=paraphrases_jsonl_filepaths,
        integrated_chest_imagenome_observations_filepath=integrated_chest_imagenome_observations_filepath,
        integrated_chest_imagenome_anatomical_locations_filepath=integrated_chest_imagenome_anatomical_locations_filepath,
        integrated_nli_jsonl_filepath=integrated_nli_jsonl_filepath,
        integrated_sentence_facts_jsonl_filepath=integrated_sentence_facts_jsonl_filepath,
        sentences_and_cluster_ids_filepath=sentences_and_cluster_ids_filepath,
        dataset_name=dataset_name,
        use_triplets=use_triplets,
        use_triplets_val=use_triplets or (not use_nli and not use_entcon and not use_sentence_autoencoder),
        use_metadata_classification=use_metadata_classification,
        use_chest_imagenome_observations_classification=use_chest_imagenome_observations_classification,
        use_chest_imagenome_anatomical_locations_classification=use_chest_imagenome_anatomical_locations_classification,
        use_nli=use_nli,
        use_entcon=use_entcon,
        use_radgraph_ner_re=use_radgraph_ner_re,
        use_sentence_autoencoder=use_sentence_autoencoder,
        use_nli_val_in_train=use_nli_val_in_train,
        use_anli=use_anli,
        use_multinli=use_multinli,
        use_snli=use_snli,
    )

    collate_batch_fn_kwargs = dict(
        triplets=dict(
            huggingface_model_name=huggingface_model_name,
            for_triplet_ranking=True,
        ),
        metadata_classification=dict(
            huggingface_model_name=huggingface_model_name,
            for_metadata_classification=True,
        ),
        chest_imagenome_observation_classification=dict(
            huggingface_model_name=huggingface_model_name,
            for_chest_imagenome_observation_classification=True,
        ),
        chest_imagenome_anatomical_location_classification=dict(
            huggingface_model_name=huggingface_model_name,
            for_chest_imagenome_anatomical_location_classification=True,
        ),
        nli=dict(
            huggingface_model_name=huggingface_model_name,
            for_nli=True,
        ),
        entcon=dict(
            huggingface_model_name=huggingface_model_name,
            for_entcon=True,
        ),
        sentence_autoencoder=dict(
            huggingface_model_name=huggingface_model_name,
            for_sentence_autoencoder=True,
        ),
    )

    trainer_engine_kwargs = dict(
        iters_to_accumulate=iters_to_accumulate,
        use_amp=use_amp,
        training=True,
        # loss weights
        triplet_loss_weight=triplet_loss_weight,
        category_classif_loss_weight=category_classif_loss_weight,
        health_status_classif_loss_weight=health_status_classif_loss_weight,
        comparison_status_classif_loss_weight=comparison_status_classif_loss_weight,
        chest_imagenome_obs_classif_loss_weight=chest_imagenome_obs_classif_loss_weight,
        chest_imagenome_anatloc_classif_loss_weight=chest_imagenome_anatloc_classif_loss_weight,
        nli_loss_weight=nli_loss_weight,
        entcon_loss_weight=entcon_loss_weight,
        sentence_autoencoder_loss_weight=sentence_autoencoder_loss_weight,
        max_grad_norm=max_grad_norm,
    )

    validator_engine_kwargs = dict(
        use_amp=use_amp,
        validating=True,
    )
    
    training_kwargs = dict(
        use_amp=use_amp,
        train_metrics_weight=train_metrics_weight,
    )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                fact_embedding_trainer_kwargs=fact_embedding_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                training_kwargs=training_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                num_workers=num_workers,
                device=device,
                save=save)

def resume_training(
        checkpoint_folder,
        scheduler,
        optimizer_name,
        lr,
        lr_decay,
        lr_decay_patience,
        warmup_and_decay_args,
        warmup_and_cosine_args,
        warmup_decay_and_cyclic_decay_args,
        num_workers,    
        epochs = 1,
        batches_per_epoch = 1000,
        device = 'GPU',
        save = True,
        override_lr = False,
        **unused_kwargs,
        ):
    
    log_title(logger, 'Resuming training from checkpoint')

    checkpoint_folder = os.path.join(WORKSPACE_DIR, checkpoint_folder)
    metadata = load_metadata(checkpoint_folder)
    model_kwargs = metadata['model_kwargs']
    optimizer_kwargs = metadata['optimizer_kwargs']
    lr_scheduler_kwargs = metadata['lr_scheduler_kwargs']
    fact_embedding_trainer_kwargs = metadata['fact_embedding_trainer_kwargs']
    dataloading_kwargs = metadata['dataloading_kwargs']
    collate_batch_fn_kwargs = metadata['collate_batch_fn_kwargs']
    training_kwargs = metadata['training_kwargs']
    trainer_engine_kwargs = metadata['trainer_engine_kwargs']
    validator_engine_kwargs = metadata['validator_engine_kwargs']

    if override_lr:
        optimizer_kwargs = dict(
            name = optimizer_name,
            lr = lr,
        )
        lr_scheduler_kwargs = dict(
            name = scheduler,
            factor = lr_decay,
            patience = lr_decay_patience,
            warmup_and_decay_args = warmup_and_decay_args,
            warmup_and_cosine_args = warmup_and_cosine_args,
            warmup_decay_and_cyclic_decay_args = warmup_decay_and_cyclic_decay_args,
            n_batches_per_epoch = batches_per_epoch,
        )

    return train_model(
                model_kwargs=model_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                fact_embedding_trainer_kwargs=fact_embedding_trainer_kwargs,
                dataloading_kwargs=dataloading_kwargs,
                collate_batch_fn_kwargs=collate_batch_fn_kwargs,
                training_kwargs=training_kwargs,
                trainer_engine_kwargs=trainer_engine_kwargs,
                validator_engine_kwargs=validator_engine_kwargs,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                num_workers=num_workers,
                device=device,
                checkpoint_folder_path=checkpoint_folder,
                save=save,
                override_lr=override_lr)

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    if args['checkpoint_folder'] is not None:
        resume_training(**args)
    else:
        del args['checkpoint_folder']
        del args['override_lr']
        train_from_scratch(**args)