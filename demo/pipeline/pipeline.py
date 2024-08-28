from pipeline import pmodel, process_text
from transformers import AutoTokenizer,AutoModel
import torch
from underthesea import word_tokenize
import numpy as np

ASPECT = ['FACILITIES#CLEANLINESS', 'FACILITIES#COMFORT',
       'FACILITIES#DESIGN&FEATURES', 'FACILITIES#GENERAL',
       'FACILITIES#MISCELLANEOUS', 'FACILITIES#PRICES',
       'FACILITIES#QUALITY', 'FOOD&DRINKS#MISCELLANEOUS',
       'FOOD&DRINKS#PRICES', 'FOOD&DRINKS#QUALITY',
       'FOOD&DRINKS#STYLE&OPTIONS', 'HOTEL#CLEANLINESS', 'HOTEL#COMFORT',
       'HOTEL#DESIGN&FEATURES', 'HOTEL#GENERAL', 'HOTEL#MISCELLANEOUS',
       'HOTEL#PRICES', 'HOTEL#QUALITY', 'LOCATION#GENERAL',
       'ROOMS#CLEANLINESS', 'ROOMS#COMFORT', 'ROOMS#DESIGN&FEATURES',
       'ROOMS#GENERAL', 'ROOMS#MISCELLANEOUS', 'ROOMS#PRICES',
       'ROOMS#QUALITY', 'ROOM_AMENITIES#CLEANLINESS',
       'ROOM_AMENITIES#COMFORT', 'ROOM_AMENITIES#DESIGN&FEATURES',
       'ROOM_AMENITIES#GENERAL', 'ROOM_AMENITIES#MISCELLANEOUS',
       'ROOM_AMENITIES#PRICES', 'ROOM_AMENITIES#QUALITY',
       'SERVICE#GENERAL']

POLARITY = ['None','Negative', "Neutral", "Positive"]
PRETRAINED_PATH = "vinai/phobert-base"

def load_model():
    try:
        checkpoint = torch.load("./text_model.pth",map_location=torch.device('cpu'))
        model = pmodel.MyModel(PRETRAINED_PATH,len(ASPECT),4)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        raise FileNotFoundError("FILE NOT FOUND!!!")
    except:
        raise NotImplementedError("CANNOT LOAD MODEL!!!")
    return model

def preprocess_comments(comments):
    tnormalize = process_text.TextNormalize()

    comments = list(map(lambda x: "" if x == None else x,comments))
    comments = list(map(lambda x: process_text.convert_unicode(x),comments))
    comments = list(map(lambda x: tnormalize.normalize(x),comments)) 
    comments =  list(map(lambda x: " ".join(word_tokenize(x)),comments))  

    return comments

def predict(model, comments):
    output = []
    n_sample = len(comments)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    comment_tokenize = tokenizer(comments,padding='max_length', max_length = 150,truncation=True,return_tensors='pt')

    model_output = model(comment_tokenize)
    _, y_pred = torch.max(torch.nn.functional.softmax(model_output,dim=-1),dim=-1)
    y_pred = y_pred.detach().cpu()

    return y_pred

