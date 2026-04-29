#!/bin/bash

# script running experiments on zero-shot prediction using LLaVA


# experiments using binary prediction
python3 ./zero_shot_LLaVA.py -o "./zero_shot_results/binary/results_0.np" --binary_prediction -d "./dataset/val_MulTweEmo.csv" -p $'The image is paired with this text: \"{text}\" .When looking at both image and text, is the emotion evoked \"{emotion}\"? Answer with Yes or No without giving any further explanation.'

python3 ./zero_shot_LLaVA.py -o "./zero_shot_results/binary/results_1.np" --binary_prediction -d "./dataset/val_MulTweEmo.csv" -p $'The input image is taken from social media and is associated with a text. Considering the context given by both of them, do they evoke \"{emotion}\"? Answer with Yes or No without giving any further explanation.\nText: {text}\nAnswer:'

python3 ./zero_shot_LLaVA.py -o "./zero_shot_results/binary/results_2.np" --binary_prediction -d "./dataset/val_MulTweEmo.csv" -p $'The input image is taken from social media and is associated with a text. Considering the context given by both of them, did the user feel \"{emotion}\" when posting the content? Answer with Yes or No without giving any further explanation.\nText: \"{text}\"\nAnswer:'

python3 ./zero_shot_LLaVA.py -o "./zero_shot_results/binary/results_3.np" --binary_prediction -d "./dataset/val_MulTweEmo.csv" -p $'Act like you are an expert at identifying emotions in images and text. You are tasked with classifying content from social media consisting of an image and associated text. Considering both of them, did the user \"{emotion}\" when posting the content? Answer with Yes or No without giving any further explanation.\nText: \"{text}\"\nAnswer:'

# experiments using list outputs
python3 ./zero_shot_LLaVA.py -o "./zero_shot_results/list/results_0.np" -d "./dataset/val_MulTweEmo.csv" -p $'The image is paired with this text: \"{text}\". Considering both image and text, choose which emotions are most elicited among this list: {labels}. Answer with only the list of chosen emotions.'

python3 ./zero_shot_LLaVA.py -o "./zero_shot_results/list/results_1.np" -d "./dataset/val_MulTweEmo.csv" -p $'The input image is taken from social media and is associated with a text. Considering the context given by both of them, classify them with the emotions that are evoked.\nChoose among the following list of emotions and answer with only the list of chosen emotions.\nEmotions: {labels}\nText: {text}\nAnswer:'

python3 ./zero_shot_LLaVA.py -o "./zero_shot_results/list/results_2.np" -d "./dataset/val_MulTweEmo.csv" -p $'The input image is taken from social media and is associated with a text. Considering the context given by both of them, what emotions did the user feel when posting the content? Choose among the following list of emotions and answer with only the list of chosen emotions.\nEmotions: {labels}\nText: {text}\nAnswer:'

python3 ./zero_shot_LLaVA.py -o "./zero_shot_results/list/results_3.np" -d "./dataset/val_MulTweEmo.csv" -p $'Act like you are an expert at identifying emotions in images and text. You are tasked with classifying content from social media consisting of an image and associated text. Considering both of them, what emotions did the user feel when posting the content? Choose among the following list of emotions and answer with only the list of chosen emotions.\nEmotions: {labels}\nText: {text}\nAnswer:'