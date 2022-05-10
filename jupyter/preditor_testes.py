import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

model.eval()
from scipy.special import softmax

# Preditor deve antes de rodar modelo
# > Limpar texto
# > Remover stopwords
# > Lematizar
# -- Caso o texto seja maior que 400 palavras
# >> Sumarizar texto (sem lematização e com stopwords)

def predictor(text):
    temp = pd.DataFrame(text,columns = ['plaintext'])
    temp['tokens'] = temp['plaintext'].apply(lambda x: np.array(tokenizer.encode(x,add_special_tokens=True, max_length = 128,pad_to_max_length=True)))
    values = torch.tensor(temp['tokens'].values.tolist()).to(torch.int64)
    values = values.to(device)

    results = []
    for value in values:

        with torch.no_grad():
            outputs = model(value.unsqueeze(0),token_type_ids=None)
        logits = outputs[0]
        logits = logits.cpu().detach().numpy()
        logits = softmax(logits)
        results.append(logits[0])

    results_array = np.array(results)

    return results_array
#%%
class_names = ['Ham','Spam']
#%%
explainer = LimeTextExplainer(class_names=class_names)
#%%
idx = 453
exp = explainer.explain_instance(df['CONTENT'][idx], predictor)
print('Comment no: %d' % idx)
print('Comment body: %s' % df['CONTENT'][idx])
print('True class: %s' % class_names[df['CLASS'][idx]])
#%%
print(exp.as_list())
#%%
exp.save_to_file('./id%d.html' % idx)
