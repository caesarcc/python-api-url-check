import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from scipy.special import softmax
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.file_utils import is_torch_available

MODELO_BERT = "modelos/bertimbau_avaliar_noticias_whatsapp"
model = BertForSequenceClassification.from_pretrained(MODELO_BERT, num_labels=2, output_attentions=False, output_hidden_states=False)
tokenizer = BertTokenizer.from_pretrained(MODELO_BERT, do_lower_case=False)
model.eval()

# Preditor deve antes de rodar modelo
# > Limpar texto
# > Remover stopwords
# > Lematizar
# -- Caso o texto seja maior que 400 palavras
# >> Sumarizar texto (sem lematização e com stopwords)
def qual_device():
    if is_torch_available() and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def prever(texto):
    temp = pd.DataFrame(texto, columns = ['texto'])
    temp['tokens'] = temp['texto'].apply(lambda x: np.array(tokenizer.encode(x, add_special_tokens=True, max_length=400, padding='max_length', truncation=True)))
    valores = torch.tensor(np.array(temp['tokens'].values.tolist())).to(torch.int64)
    valores = valores.to(qual_device())

    resultados = []
    for valor in valores:

        with torch.no_grad():
            outputs = model(valor.unsqueeze(0), token_type_ids=None)
        logits = outputs.logits.cpu().detach().numpy()
        logits = softmax(logits)
        resultados.append(logits[0])

    retorno = np.array(resultados)

    return retorno

classes = ['Confiável','Falso']
explainer = LimeTextExplainer(class_names=classes)

artigo = "O popular apresentador Ratinho foi vítima de um acidente de carro com toda a sua família em São Paulo, e infelizmente não resistiu aos ferimentos e morreu"
exp = explainer.explain_instance(artigo, prever)

print('Artigo: %s' % artigo)
print(exp.as_list())

exp.save_to_file('./teste.html')
