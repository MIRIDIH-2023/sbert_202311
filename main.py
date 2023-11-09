from custom_dataset import custom_dataset
from test_dataset import test_dataset
from utils import *
from sentence_transformers import SentenceTransformer, losses
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_data(config):
    folder_path = config.folder_path
    data_path = config.data_path
    extract_path = config.extract_path

    keys_list, texts_list, datas_list = processing_data(folder_path=folder_path,data_path=data_path,extract_path=extract_path)

    return keys_list, texts_list, datas_list

def train(config, data) -> SentenceTransformer:
    
    keys_list, texts_list, datas_list = data
    
    model =  SentenceTransformer("distiluse-base-multilingual-cased-v1")
    
    train_dataset = custom_dataset(keys_list, texts_list, datas_list)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    
    train_loss = losses.CosineSimilarityLoss(model)
    num_epochs = config.num_epoch
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    
    model.fit(
    train_objectives=[(train_dataloader,train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps)
    
    return model

def evaluate(config,data,model):
    """
    evaluate model by metric reall
    """
    
    keys_list, texts_list, datas_list = data
    
    #it is not data_leakage, because we do not use front 1000 data when train
    test_dataset = test_dataset(keys_list[:1000],texts_list[:1000],datas_list[:1000])
    npts = len(test_dataset)

    embedded_key = []
    embedded_text = []

    for i in tqdm(range(len(test_dataset))):
        cur_text, cur_key = test_dataset[i]
        embedded_key.append(model.encode(cur_key))
        embedded_text.append(model.encode(cur_text))


    sims_vector = cosine_similarity(embedded_key,embedded_text)
    npts = len(test_dataset)

    #순서대로 recall@1, recall@5, recall@10, meadian, mean
    a,b,c,d,e = i2t(npts,sims_vector)
    print("keyword로 xml text search")
    print(f"recall@1: {a}  recall@5: {b}  recall@10: {c}  recall_meadian: {d}  recall_mean: {e}")

    a,b,c,d,e = i2t(npts,sims_vector.T)
    print("xml text로 keyword search")
    print(f"recall@1: {a}  recall@5: {b}  recall@10: {c}  recall_meadian: {d}  recall_mean: {e}")

def main(config):
    
    data = load_data(config)
    
    model = train(config,data)

    if(config.is_save):
        model.save(config.save_path)
    
    if(config.is_evaluate):
        evaluate(model,data)


if __name__=="__main__":
    
    class config:
        is_save = True
        is_evaluate = True
        save_path = None
        folder_path = None
        data_path = None
        extract_path = None
        num_epoch = 7
        batch_size = 64
    
    main(config)