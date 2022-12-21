import evaluate
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer)

from base import TEST_DATA, TRAIN_DATA, VAL_DATA, MODEL_NAME, MODEL_PATH


def prepare_data():    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    df = pd.read_csv(TEST_DATA, usecols=['text','online_crime'])
    df.columns = ['text','label']
    df['label'] = df['label'].astype(int)

    train, test = train_test_split(df, test_size=0.3)
    train = Dataset.from_pandas(train)
    test = Dataset.from_pandas(test)

    return train.map(tokenize_function), test.map(tokenize_function), tokenizer


def prepare_model():
    id2label = {0: "safe", 1: "online_crime"}
    label2id = {"safe": 0, "online_crime": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = 2, 
        id2label = id2label, 
        label2id = label2id).to("cuda")

    training_args = TrainingArguments(
        output_dir = 'test_trainer_log',
        evaluation_strategy = 'epoch',
        per_device_train_batch_size = 10,
        per_device_eval_batch_size = 10,
        num_train_epochs = 5,
        report_to='none')
    
    return model, training_args


def train(model, training_args, tokenized_train, tokenized_test):
    def compute_metrics(eval_pred):
        metric = evaluate.load('f1')
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train,
        eval_dataset = tokenized_test,
        compute_metrics = compute_metrics)

    trainer.train()


def save(tokenizer, model):
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)


if __name__ == '__main__':
    tokenized_train, tokenized_test, tokenizer = prepare_data()
    model, training_args = prepare_model()
    train(model, training_args, tokenized_train, tokenized_test)
    save(tokenizer, model)