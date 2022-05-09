#!/usr/bin/env python
# coding: utf-8

# In[1]:

#get_ipython().system('pip install -q transformers')
#get_ipython().system('pip install -q lime')

# In[2]:

#from google.colab import drive
#drive.mount('/content/drive')

# In[3]:

import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, roc_curve)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.file_utils import is_torch_available

# In[4]:

DIR_DADOS = './dados/'
#DIR_DADOS = 'drive/MyDrive/PUC/TCC/dados/'
dados_preproc = pd.read_csv(os.path.join(DIR_DADOS, 'fakebr_corpus_processado.csv'), sep = ',')
dados_preproc.info()
dados_preproc[['classe','texto','texto_processado','qtde_texto_processado']].sample(n=3)

# In[5]:

RANDOM_SEED = 42
def garantir_reprodutividade(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            return "cuda"
    return "cpu"

device = garantir_reprodutividade(RANDOM_SEED)

# In[5]:
MODELO_BERT = "neuralmind/bert-base-portuguese-cased"
model = BertForSequenceClassification.from_pretrained(MODELO_BERT, num_labels=2, output_attentions=False, output_hidden_states=False)
tokenizer = BertTokenizer.from_pretrained(MODELO_BERT, do_lower_case=False)


# In[6]:
dados_token = dados_preproc[['classe','texto_processado']].copy()
dados_token['TOKENS'] = dados_token['texto_processado'].apply(lambda x: np.array(tokenizer.encode(x, padding='max_length', truncation=True, max_length=400)))
dados_token['LABELS'] = dados_token['classe']
dados_token = dados_token[['TOKENS','LABELS']]

# In[6]:
# Divido em 70% para treino e 20% para validação e o restante para teste, fracionado aleatoriamente e com random seed 42
dados_treino, dados_validacao, dados_teste = np.split(dados_token.sample(frac=1), [int(.7*len(dados_token)), int(.8*len(dados_token))])
display(f"Treinamento: {dados_treino.shape[0]}, Teste: {dados_teste.shape[0]}, Validação: {dados_validacao.shape[0]}")

treino_valores = torch.tensor(dados_treino['TOKENS'].values.tolist()).to(torch.int64)
treino_classes = torch.tensor(dados_treino.LABELS.values.tolist()).to(torch.int64)
validacao_valores = torch.tensor(dados_validacao['TOKENS'].values.tolist()).to(torch.int64)
validacao_classes = torch.tensor(dados_validacao.LABELS.values.tolist()).to(torch.int64)
teste_valores = torch.tensor(dados_teste['TOKENS'].values.tolist()).to(torch.int64)
teste_classes = torch.tensor(dados_teste.LABELS.values.tolist()).to(torch.int64)

# In[13]:

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# In[14]:

#epocas = 5
#batch_size = 16
epocas = 2
tamanho_lote = 4

treino_dados = TensorDataset(treino_valores, treino_classes)
treino_leitor = DataLoader(treino_dados, sampler=RandomSampler(treino_dados), batch_size=tamanho_lote)

validacao_dados = TensorDataset(validacao_valores, validacao_classes)
validacao_leitor = DataLoader(validacao_dados, sampler=SequentialSampler(validacao_dados), batch_size=tamanho_lote)

model.to(device)

passos_total = len(treino_leitor) * epocas
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = passos_total)


# In[19]:
estatisticas = []
inicio_total = time.time()

for epoca in range(0, epocas):
    print('\n======== Época {:} / {:} ========'.format(epoca + 1, epocas))
    print('Treinando...')
    inicio_processo = time.time()

    # Reinicia o loss a cada época
    loss_por_epoca = 0
    model.train()
    for passo, lote in enumerate(treino_leitor):
        if passo % 10 == 0 and not passo == 0:
            tempo = format_time(time.time() - inicio_processo)

            print('  Passo {:>5,}  de  {:>5,}.  Tempo decorrido: {:}.'.format(passo, len(treino_leitor), tempo))

        lote_valores = lote[0].to(device)
        lote_classes = lote[1].to(device)

        model.zero_grad()

        outputs = model(lote_valores,
                             token_type_ids=None,
                             labels=lote_classes)
        loss_por_epoca += outputs.loss.item()
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Calcula a média do loss sobre todos lotes
    loss_treino_media = loss_por_epoca / len(treino_leitor)

    tempo_por_epoca = format_time(time.time() - inicio_processo)
    print("\n  Média do loss de treino: {0:.2f}".format(loss_treino_media))
    print("  Tempo do treino: {:}".format(tempo_por_epoca))

    print("\nValidando...")
    inicio_processo = time.time()

    # Coloca o modelo em modo de validação
    model.eval()

    # reseta variáveis para validação por epoca
    validacao_acuracia_total = 0
    validacao_loss_total = 0

    for lote in validacao_leitor:

        lote_valores = lote[0].to(device)
        lote_classes = lote[1].to(device)

        with torch.no_grad():

            outputs = model(lote_valores,
                                   token_type_ids=None,
                                   labels=lote_classes)

        classes_previstas = outputs.logits.detach().cpu().numpy()
        classes = lote_classes.to('cpu').numpy()
        predicao = np.argmax(classes_previstas, axis=1).flatten()
        classes = classes.flatten()
        validacao_acuracia_total += np.sum(predicao == classes) / len(classes)

    # mostra a acurácia de cada avaliação.
    validacao_acuracia_media = validacao_acuracia_total / len(validacao_leitor)
    print("  Acurácia: {0:.2f}".format(validacao_acuracia_media))

    # calcula a média da perda nos lotes.
    validacao_loss_medio = validacao_loss_total / len(validacao_leitor)

    # mede o tempo de cada avaliação.
    validation_time = format_time(time.time() - inicio_processo)

    print("  Loss de validação: {0:.2f}".format(validacao_loss_medio))
    print("  Tempo de validação: {:}".format(validation_time))

    # grava as estatísticas por época.
    estatisticas.append(
        {
            'epoca': epoca + 1,
            'Loss de Treinamento': loss_treino_media,
            'Loss de Validação': validacao_loss_medio,
            'Acurácia de Validação': validacao_acuracia_media,
            'Tempo de Treino': tempo_por_epoca,
            'Tempo de Validação': validation_time
        }
    )


# In[ ]:

print("\nTreinamento completo!")
print("Tempo total do treino {:} (h:mm:ss)".format(format_time(time.time()-inicio_total)))

#%%
teste_dados = TensorDataset(teste_valores, teste_classes)
teste_leitor = DataLoader(teste_dados, sampler=SequentialSampler(teste_dados), batch_size=tamanho_lote)
#%%
# Colocar o modelo em modo de avaliação
model.eval()

# In[ ]:
classes_previstas , classes_verdadeiras = [], []

for lote in teste_leitor:
  # coloca o lote na GPUS
  lote = tuple(t.to(device) for t in lote)

  # Unpack the inputs from our dataloader
  lote_valores, lote_classes = lote

  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(lote_valores, token_type_ids=None)


# In[ ]:

# Adiciona as prediões e as classes para a CPU
outputs = outputs.logits.detach().cpu().numpy()
classes = lote_classes.to('cpu').numpy()

# Salva as predições e as classes corretos
classes_previstas.append(outputs)
classes_verdadeiras.append(classes)


#%%
lista_previsoes = np.concatenate(classes_previstas, axis=0)

# In[ ]:
lista_previsoes = np.argmax(lista_previsoes, axis=1).flatten()
lista_classes_verdadeiras = np.concatenate(classes_verdadeiras, axis=0)

#%%
print(classification_report(lista_classes_verdadeiras, lista_previsoes,target_names=['confiável','falso']))

# In[ ]:
fpr, tpr, _ = roc_curve(lista_classes_verdadeiras, lista_previsoes)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa Falso Positivo')
plt.ylabel('Taxa Positive Real')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#%%
print(accuracy_score(lista_classes_verdadeiras, lista_previsoes))

# In[ ]:
conf = confusion_matrix(lista_classes_verdadeiras, lista_previsoes)
print(conf)

#%%
# Cria um dataframe das estatísticas de treino
pd.set_option('precision', 2)
df_estatisticas = pd.DataFrame(data=estatisticas)
df_estatisticas = df_estatisticas.set_index('epoca')

# In[ ]:
plt.figure()
# Mostra curva de aprendizado.
plt.plot(df_estatisticas['Loss de Treinamento'], 'b-o', label="Treinamento")
plt.plot(df_estatisticas['Loss de Validação'], 'g-o', label="Validação")
plt.title("Loss de Treinamento vs Validação")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])
plt.show()

# In[ ]:

plt.figure()
plt.plot(df_estatisticas['Acurácia de Validação'], 'g-o', label="Validação")
plt.title("Acurácia de Validação")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend(loc="lower right")
plt.xticks([1, 2, 3, 4])
plt.show()


# In[ ]:
CAMINHO_MODELO = "drive/MyDrive/PUC/TCC/modelos/bertimbau_avaliar_noticias"
model.save_pretrained(f"{CAMINHO_MODELO}/best_model")
tokenizer.save_pretrained(f"{CAMINHO_MODELO}/best_model")


# In[ ]:
from scipy.special import softmax

model.eval()

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
classes = ['Confiável','Falso']
#%%
explainer = LimeTextExplainer(class_names=classes)
#%%
idx = 453
exp = explainer.explain_instance(dados_preproc['texto_processado'][idx], predictor)
print('Chave: %d' % idx)
print('Texto: %s' % dados_preproc['texto_processado'][idx])
print('Classe verdadeira: %s' % classes[dados_preproc['classe'][idx]])
#%%
print(exp.as_list())
#%%
exp.save_to_file('./id%d.html' % idx)

