import argparse
from libs.dataset_loader import MulTweEmoDataset
from libs.utils import *
from libs.model import TweetMERModel
from datasets import Dataset
from transformers import AutoTokenizer
import os

if __name__ == "__main__":
    
    # test, _ = MulTweEmoDataset.load(csv_path="./dataset/test_MulTweEmo.csv", drop_something_else=True, test_split=None)
    # test = Dataset.from_pandas(TweetMSA.preprocess_dataset(dataset=test.head(10), model="base", text_column="tweet", label_column="labels"))
    # print(test.pixel_values())
    ckp_list = os.listdir(f".ckp/base/")
    ckp = ckp_list[-1]
    model = TweetMERWrapper()
    model.from_pretrained(f".ckp/base/{ckp}")

    tmp_model = model._trainer.model
    weights = tmp_model.fc_layers[0][0].weight.data
    
    sum_txt, sum_img = 0, 0
    for i in range(weights.shape[0]):
        sum_txt = weights[i, :512].abs().sum()
        sum_img = weights[i, 512:].abs().sum()
        print(weights[i, 512:].abs().sum())
    sum_txt /= (512*weights.shape[0])
    sum_img /= (512*weights.shape[0])
    print(sum_txt, sum_img)
