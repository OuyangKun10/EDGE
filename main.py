from transformers import BartTokenizer, BartModel
import gc
import torch
import datautils
import warnings
import logging
import gc
import random
import math
import re
import ast
from tqdm import tqdm
from typing import Optional
from datetime import datetime
import os
import numpy as np
import pandas as pd
# from datautils import set_up_data_loader
import hydra
from omegaconf import DictConfig, OmegaConf
from model import MultimodalBartForConditionalGeneration
from transformers import (
    BartTokenizerFast,
    AdamW
)
import yaml
import json
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")





def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#set_random_seed(random.randint(0,9999))#42  6269
set_random_seed(8840)
    
TOKENIZER = BartTokenizerFast.from_pretrained('')
MODEL = MultimodalBartForConditionalGeneration.from_pretrained('')
SOURCE_PREFIX = ''
TARGET_PREFIX = ''
# read yaml
path = './conf/config.yaml'
f = open(path, 'r', encoding='utf-8')
cfg2 = f.read()
cfg2 = yaml.load(cfg2, Loader=yaml.FullLoader)
VISUAL_DIM = cfg2['VISUAL_DIM']
ACOUSTIC_DIM = cfg2['ACOUSTIC_DIM']
SOURCE_COLUMN = 'dialogue'
SOURCE_COLUMN_1 = 'target'
SOURCE_COLUMN_2 = 'context'
TARGET_COLUMN= 'code_mixed_explanation'
SOURCE_MAX_LEN=cfg2['SOURCE_MAX_LEN']
TARGET_MAX_LEN=cfg2['TARGET_MAX_LEN']
ACOUSTIC_MAX_LEN=cfg2['ACOUSTIC_MAX_LEN']
VISUAL_MAX_LEN=cfg2['VISUAL_MAX_LEN']
def pad_seq(tensor: torch.tensor,
            dim: int,
            max_len: int):
    if max_len > tensor.shape[0]:
        return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])
    else:
        return tensor[:max_len]

    # ------------------------------ Part of data_preprocess ------------------------------ #

def read_json_data(path):
    f = open(path)
    data = json.load(f)
    f.close()
    del f
    gc.collect()
    return data

def prepare_dataset(text_path: str,
                    acosutic_path: str,
                    visual_path: str,
                    lowercase_utterances: bool = False,
                    unfolded_dialogue: bool = True):
    data = read_json_data(text_path)

    code_mixed_explanation = []#load expalantion
 #unfold the dialogue
    if unfolded_dialogue:
        dialogue = []
        for i in range(1, len(data) + 1):
            data_point = data[str(i)]

            example_target_speaker = str(data_point['target_speaker']).upper()
            example_target_utterance = str(data_point['target_utterance'])

            example_dialogue = "[CONTEXT] "
            #here is the unfold method
            for speaker, utterance in list(zip(data_point['context_speakers'], data_point['context_utterances'])):
                example_dialogue = example_dialogue + str(speaker).upper() + " : " + str(utterance) + " | "

            example_dialogue = example_dialogue + " [TARGET] " + example_target_speaker + " : " + example_target_utterance + " | "
            example_dialogue = re.sub(' +', ' ', example_dialogue)
            dialogue.append(example_dialogue)

            code_mixed_explanation.append(str(data_point['code_mixed_explanation']))

        df = pd.DataFrame(list(zip(dialogue, code_mixed_explanation)),
                          columns=['dialogue', 'code_mixed_explanation'])
        TOKENIZER.add_tokens(['[CONTEXT]', '[TARGET]'], special_tokens=True)
        MODEL.resize_token_embeddings(len(TOKENIZER))

        del dialogue
        del example_dialogue

    else:
        target = []
        context = []
        for i in range(1, len(data) + 1):
            data_point = data[str(i)]

            example_target_speaker = str(data_point['target_speaker']).upper()
            example_target_utterance = str(data_point['target_utterance'])
            example_target_utterance = example_target_speaker + " : " + example_target_utterance
            example_target_utterance = re.sub(' +', ' ', example_target_utterance)
            target.append(example_target_utterance)

            example_context_utterance = ""
            for speaker, utterance in list(zip(data_point['context_speakers'], data_point['context_utterances'])):
                example_context_utterance = example_context_utterance + str(speaker).upper() + " : " + str(
                    utterance) + " | "

            example_context_utterance = re.sub(' +', ' ', example_context_utterance)
            context.append(example_context_utterance)

            code_mixed_explanation.append(str(data_point['code_mixed_explanation']))

        df = pd.DataFrame(list(zip(context, target, code_mixed_explanation)),
                          columns=['context', 'target', 'code_mixed_explanation'])
        del target
        del context
        del example_context_utterance

    # Reading Audio Data
    acosutic_data = pd.read_pickle(acosutic_path)
    df['audio_features'] = acosutic_data['audio_feats']

    # Reading Video Data
    visaul_data = pd.read_pickle(visual_path)
    df['visual_features'] = visaul_data['video_feats']

    df = df[df['code_mixed_explanation'] != "?"]
    df = df.dropna()  #delete NaN
    if lowercase_utterances:
        df = df.apply(lambda x: x.astype(str).str.lower())

    del data
    del text_path
    del acosutic_path
    del visual_path
    del acosutic_data
    del visaul_data
    del code_mixed_explanation
    del example_target_speaker
    del example_target_utterance
    gc.collect()
    return df


def preprocess_dataset(dataset, unfolded_dialogue: bool =True):



    if unfolded_dialogue:
        source = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN].values.tolist()]
        model_inputs = TOKENIZER(source,
                                 max_length=SOURCE_MAX_LEN,
                                 padding='max_length',
                                 truncation=True)
        del source

    else:
        source_1 = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN_1].values.tolist()]
        source_2 = [SOURCE_PREFIX + s for s in dataset[SOURCE_COLUMN_2].values.tolist()]
        model_inputs = TOKENIZER(source_1,
                                 source_2,
                                 max_length=SOURCE_MAX_LEN,
                                 padding='max_length',
                                 truncation=True)

        del source_1
        del source_2

    target = [TARGET_PREFIX + t for t in dataset[TARGET_COLUMN].values.tolist()]
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(target,
                           max_length=TARGET_MAX_LEN,
                           padding='max_length',
                           truncation=True)
        # IMP:
        # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding tokens in the loss.
        labels['input_ids'] = [[(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
    model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)

    model_inputs['acoustic_input'] = torch.stack([pad_seq(torch.tensor(af, dtype=torch.float),
                                                          dim=ACOUSTIC_DIM,
                                                          max_len=ACOUSTIC_MAX_LEN)
                                                  for af in dataset['audio_features'].values.tolist()], 0).to(DEVICE)

    model_inputs['visual_input'] = torch.stack([pad_seq(torch.tensor(vf[0], dtype=torch.float),
                                                        dim=VISUAL_DIM,
                                                        max_len=VISUAL_MAX_LEN)
                                                for vf in dataset['visual_features'].values.tolist()], 0).to(DEVICE)

    model_inputs['labels'] = torch.tensor([l for l in labels['input_ids']], dtype=torch.long, device=DEVICE)

    del target
    del labels
    gc.collect()
    return model_inputs


def get_scores(reference_list: list,
               hypothesis_list: list):
    count = 0
    met = 0
    bleu_1 = 0
    bleu_2 = 0
    bleu_3 = 0
    bleu_4 = 0
    rouge1 = 0
    rouge2 = 0
    rougel = 0
    weights_1 = (1. / 1.,)
    weights_2 = (1. / 2., 1. / 2.)
    weights_3 = (1. / 3., 1. / 3., 1. / 3.)
    weights_4 = (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for reference, hypothesis in list(zip(reference_list, hypothesis_list)):
        scores = rouge_scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure

        met += meteor_score([reference], hypothesis)

        reference = reference.split()
        hypothesis = hypothesis.split()
        bleu_1 += sentence_bleu([reference], hypothesis, weights_1)
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1

    return {
        "rouge_1": rouge1 * 100 / count,
        "rouge_2": rouge2 * 100 / count,
        "rouge_L": rougel * 100 / count,
        "bleu_1": bleu_1 * 100 / count,
        "bleu_2": bleu_2 * 100 / count,
        "bleu_3": bleu_3 * 100 / count,
        "bleu_4": bleu_4 * 100 / count,
        "meteor": met * 100 / count,
    }


def _save(model,
          output_dir: str,
          tokenizer=None,
          state_dict=None):
    # If we are executing this function, we are the process zero, so we don't check for that.
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model checkpoint to {output_dir}")
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    if not isinstance(model, PreTrainedModel):
        if isinstance(unwrap_model(model), PreTrainedModel):
            if state_dict is None:
                state_dict = model.state_dict()
            unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
        else:
            print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if state_dict is None:
                state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        model.save_pretrained(output_dir, state_dict=state_dict)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model


#         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def save_model(model,
               output_dir: str,
               tokenizer=None,
               state_dict=None):
    """
    Will save the model, so you can reload it using :obj:`from_pretrained()`.

    Will only save from the main process.
    """
    _save(model, output_dir, tokenizer=tokenizer, state_dict=state_dict)


# ----------------------------------------------------- TRAINING UTILS ----------------------------------------------------- #

def train_epoch(model,
                data_loader,
                optimizer):
    model.train()
    epoch_train_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, attention_mask, acoustic_input, visual_input, labels = batch
        optimizer.zero_grad()#initialize gradient

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        acoustic_input=acoustic_input,
                        visual_input=visual_input,
                        labels=labels)
        loss = outputs['loss']
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_train_loss / step


def val_epoch(model,
              data_loader,
              optimizer):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Loss Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            acoustic_input=acoustic_input,
                            visual_input=visual_input,
                            labels=labels)
            loss = outputs['loss']
            epoch_val_loss += loss.item()

    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_val_loss / step


def test_epoch(model,
               tokenizer,
               data_loader,
               desc,
               **gen_kwargs):
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch

            generated_ids = model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           acoustic_input=acoustic_input,
                                           visual_input=visual_input,
                                           **gen_kwargs)

            generated_ids = generated_ids.detach().cpu().numpy()
            generated_ids = np.where(generated_ids != -100, generated_ids, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            labels = labels.detach().cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            gold.extend(decoded_labels)

    del batch
    del input_ids
    del attention_mask
    del acoustic_input
    del visual_input
    del labels
    del generated_ids
    del decoded_preds
    del decoded_labels
    gc.collect()
    torch.cuda.empty_cache()

    return predictions, gold


def get_val_scores(model,
                   tokenizer,
                   data_loader,
                   desc,
                   epoch,
                   **gen_kwargs):
    predictions, gold = test_epoch(model,
                                   tokenizer,
                                   data_loader,
                                   desc=desc,
                                   **gen_kwargs)
    result = get_scores(predictions, gold)

    if "Validation" in desc:
        val_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
        file_name = RESULT_OUTPUT_DIR + "val/MAF_TAV_BART_epoch_" + str(epoch + 1) + "_val_results.csv"
        val_df.to_csv(file_name, index=False)
        print("Validation File saved")

    elif "Test" in desc:
        test_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
        file_name = RESULT_OUTPUT_DIR + "test/MAF_TAV_BART_epoch_" + str(epoch + 1) + "_test_results.csv"
        test_df.to_csv(file_name, index=False)
        print("Test File saved")

    del predictions
    del gold
    gc.collect()
    torch.cuda.empty_cache()

    return result


def prepare_for_training(model,
                         base_learning_rate: float,
                         new_learning_rate: float,
                         weight_decay: float):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name:
            new_params_list.append(param)
        else:
            base_params_list.append(param)

    optimizer = AdamW(
        [
            {'params': base_params_list, 'lr': base_learning_rate, 'weight_decay': weight_decay},
            {'params': new_params_list, 'lr': new_learning_rate, 'weight_decay': weight_decay}
        ],
        lr=base_learning_rate,
        weight_decay=weight_decay
    )

    del base_params_list
    del new_params_list
    gc.collect()
    torch.cuda.empty_cache()

    return optimizer


def train(model,
          tokenizer,
          train_data_loader,
          val_data_loader,
          test_data_loader,
          base_learning_rate,
          new_learning_rate,
          weight_decay,cfg,
          **gen_kwargs):
    optimizer = prepare_for_training(model=model,
                                     base_learning_rate=base_learning_rate,
                                     new_learning_rate=new_learning_rate,
                                     weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    val_rouge_2 = []
    patience = 1

    for epoch in range(cfg.MAX_EPOCHS):
        train_loss = train_epoch(model,
                                 train_data_loader,
                                 optimizer)
        train_losses.append(train_loss)

        val_loss = val_epoch(model,
                             val_data_loader,
                             optimizer)
        val_losses.append(val_loss)

        val_results = get_val_scores(model,
                                     tokenizer,
                                     val_data_loader,
                                     desc="Validation Generation Iteration",
                                     epoch=epoch,
                                     **gen_kwargs)
        val_rouge_2.append(val_results['rouge_2'])

        test_results = get_val_scores(model,
                                      tokenizer,
                                      test_data_loader,
                                      desc="Test Generation Iteration",
                                      epoch=epoch,
                                      **gen_kwargs)

        print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_validation_loss: {}".format(epoch + 1, train_loss, val_loss,
                                                                                        min(val_losses)))

        print(
            "\nval_rouge_1: {}\tval_rouge_2: {}\tval_rouge_L: {}\tval_bleu_1: {}\tval_bleu_2: {}\tval_bleu_3: {}\tval_bleu_4: {}\tval_meteor: {}".format(
                val_results['rouge_1'], val_results['rouge_2'], val_results['rouge_L'], val_results['bleu_1'],
                val_results['bleu_2'], val_results['bleu_3'], val_results['bleu_4'], val_results['meteor']))

        print(
            "\ntest_rouge_1: {}\ttest_rouge_2: {}\ttest_rouge_L: {}\ttest_bleu_1: {}\ttest_bleu_2: {}\ttest_bleu_3: {}\ttest_bleu_4: {}\ttest_meteor: {}".format(
                test_results['rouge_1'], test_results['rouge_2'], test_results['rouge_L'], test_results['bleu_1'],
                test_results['bleu_2'], test_results['bleu_3'], test_results['bleu_4'], test_results['meteor']))

        path = cfg.MODEL_OUTPUT_DIR + "MAF_TAV_BART_epoch_" + cfg.TARGET_COLUMN + "_epoch_" + str(
            epoch + 1) + "_" + datetime.now().strftime('%d-%m-%Y-%H:%M')
        print(path)
        save_model(model,
                   path,
                   tokenizer)
        print("Model saved at path: ", path)

        if val_results['rouge_2'] < max(val_rouge_2):
            patience = patience + 1
            if patience == cfg.EARLY_STOPPING_THRESHOLD:
                break

        else:
            patience = 1

        del train_loss
        del val_loss
        del path
        gc.collect()
        torch.cuda.empty_cache()


@hydra.main(config_path="conf", config_name="config",version_base='1.2.0')
def main(cfg: DictConfig):
    #lood pretrained bart model

    print("Model loaded...\n")
    #assert
    MODEL.to(DEVICE)
    #load tokenizer

    print("Tokenizer loaded...\n")

    # SOURCE_PREFIX = ''
    # TARGET_PREFIX = ''

    print(cfg.TARGET_COLUMN)
    print(cfg.MODEL_OUTPUT_DIR)
    print(cfg.RESULT_OUTPUT_DIR)
    print(SOURCE_PREFIX)
    print(TARGET_PREFIX)

    gc.collect()

    pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
    print("Total parameters: ", pytorch_total_params)
    pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    print("Total trainable parameters: ", pytorch_total_train_params)

    for name, param in MODEL.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name:
            print(name)

    # ------------------------------ READ DATASET ------------------------------ #

    train_dataset = datautils.set_up_data_loader(text_path=cfg.TEXT_INPUT_PATH + 'train_data_final_plain.json',
                                       acosutic_path=cfg.ACOUSTIC_INPUT_PATH + 'train_acoustic_features.pkl',
                                       visual_path=cfg.VISUAL_INPUT_PATH + 'train_visual_features.pkl',
                                       lowercase_utterances=cfg.LOWERCASE_UTTERANCES,
                                       unfolded_dialogue=cfg.UNFOLDED_DIALOGUE)
    print("\nTraining Data Loaded...")

    val_dataset = datautils.set_up_data_loader(text_path=cfg.TEXT_INPUT_PATH + 'val_data_final_plain.json',
                                     acosutic_path=cfg.ACOUSTIC_INPUT_PATH + 'val_acoustic_features.pkl',
                                     visual_path=cfg.VISUAL_INPUT_PATH + 'val_visual_features.pkl',
                                     lowercase_utterances=cfg.LOWERCASE_UTTERANCES,
                                     unfolded_dialogue=cfg.UNFOLDED_DIALOGUE)
    print("\nValidation Data Loaded...")

    test_dataset = datautils.set_up_data_loader(text_path=cfg.TEXT_INPUT_PATH + 'test_data_final_plain.json',
                                      acosutic_path=cfg.ACOUSTIC_INPUT_PATH + 'test_acoustic_features.pkl',
                                      visual_path=cfg.VISUAL_INPUT_PATH + 'test_visual_features.pkl',
                                      lowercase_utterances=cfg.LOWERCASE_UTTERANCES,
                                      unfolded_dialogue=cfg.UNFOLDED_DIALOGUE)
    print("\nTest Data Loaded...")
    gc.collect()
    # ------------------------------ TRAINING SETUP ------------------------------ #

    gen_kwargs = {
        'num_beams': cfg.NUM_BEAMS,
        'max_length': cfg.TARGET_MAX_LEN,
        'early_stopping': cfg.EARLY_STOPPING,
        'no_repeat_ngram_size': cfg.NO_REPEAT_NGRAM_SIZE
    }

    train(model=MODEL,
          tokenizer=TOKENIZER,
          train_data_loader=train_dataset,
          val_data_loader=val_dataset,
          test_data_loader=test_dataset,
          base_learning_rate=cfg.BASE_LEARNING_RATE,
          new_learning_rate=cfg.NEW_LEARNING_RATE,
          weight_decay=cfg.WEIGHT_DECAY,
          cfg=cfg,
          **gen_kwargs)

    print("Model Trained!")
if __name__ == "__main__":
    main()
