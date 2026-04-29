import argparse
from libs.dataset_loader import MulTweEmoDataset
from libs.utils import *
from libs.model import TweetMERModel
from datasets import Dataset
from transformers import AutoTokenizer
import os

def _preprocess_data(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    text = examples["tweet"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)        
    return encoding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_msa',
        description='Train MSA model',
    )
    parser.add_argument('-m', '--model', choices=["bert", "base", "base_captions", "base_augment", "high_support", "text_only"], type=str, default="base", help="the model to train")

    args = parser.parse_args()
    
    model_type = args.model

    test, _ = MulTweEmoDataset.load(csv_path="./dataset/test_MulTweEmo.csv", drop_something_else=True, test_split=None)

    if model_type=="bert":
        test = test.drop_duplicates(subset=["id"])
        test = Dataset.from_pandas(test)
        test = test.map(_preprocess_data, batched=True, remove_columns=[col for col in test.column_names if col != "labels"])

        model_class = BertWrapper

    elif model_type=="base":
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    elif model_type=="base_augment":
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    elif model_type=="base_captions":
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    elif model_type=="high_support":
        
        test, _ = MulTweEmoDataset.load(csv_path="./dataset/test_MulTweEmo.csv", drop_something_else=True, drop_low_support=True, test_split=None)
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    elif model_type=="text_only":
        test = test.drop_duplicates(subset=["id"])
        test = Dataset.from_pandas(TweetMERModel.preprocess_dataset(dataset=test, model="base", text_column="tweet", label_column="labels"))

        model_class = TweetMERWrapper

    ckp_list = os.listdir(f".ckp/{model_type}/")
    ckp = ckp_list[-1]
    model = model_class()
    model.from_pretrained(f".ckp/{model_type}/{ckp}")

    test_predictions, test_scores = model.score(test, test["labels"])
    print(test_scores)