# URL FakeNews Checker

### Rodar diretamente  
```
git clone https://github.com/caesarcc/python-tcc-url-fakenews-check.git
cd python-tcc-url-fakenews-check
docker-compose up
```

### Rodar em ambiente local (pip venv em ambiente windows)  
```
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
flask run
```

### Rodar em ambiente local (conda em ambiente windows, sem GPU)  
```
conda create --name tcc python=3.8
conda activate tcc
pip install --no-cache-dir -r requirements.txt
flask run
```

### Antes de rodar os notebooks do projeto localmente:
```
conda activate tcc
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c huggingface transformers
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

<BR>  

## Roadmap:  

- [X] Treinar modelo BERTimbau no corpus FakeBR
    - Melhores prátricas: https://github.com/amartyads/bert-practice-pytorch
<HR>

- [X] Sumarizar notícias com T5 em PTBR
    - Exemplo: https://www.thepythoncode.com/article/text-summarization-using-huggingface-transformers-python
<HR>

- [ ] Implementar a captura da URL, limpar e sumarizar (quando > 400 palavras)
    - Lib para Artigos: https://newspaper.readthedocs.io/en/latest/
    - Lib para quando anterior falhar: https://www.analyticsvidhya.com/blog/2021/04/automate-web-scraping-using-python-autoscraper-library/
<HR>

- [ ] Executar modelo (primeiro item) na notícia sumarizada (segundo item)
<HR>

- [ ] Apresentar resultado e gerar opção de CONFIRMAR FAKE ou VERDADE e salvar no SQLite
<HR>

- [ ] Avaliar a qualidade do uso do corpus FakeWhatsApp
    - Usar 2018, se balanceado e sem não classficidados
    - Tentar usar 2020
    - FakeWhatsApp corpus: https://github.com/cabrau/FakeWhatsApp.Br
<HR>

- [ ] Ler base de dados SQLite da aplicação, revisar, somar aos 2 corpus usados e treinar todo modelo novamente
<HR>


## PASSOS OPCIONAIS ##

- [ ] Caso FakeWhatsApp não seja bom, partir para Scraping de notícias
    - Tentar balancear e usar notícias novas
<HR>

- [ ] Analizar o sentimento da notícia
    - Análise de sentimentos com BERTimbau em PT-BR: https://github.com/Luzo0/GoEmotions_portuguese
    - Análise de sentimentos: https://medium.com/data-hackers/an%C3%A1lise-de-sentimentos-em-portugu%C3%AAs-utilizando-pytorch-e-python-91a232165ec0
    - Análise de sentimentos baseada em regras: https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/
<HR>

- [ ] Conferir quantidade de erros na notícia original
    - Spell Cheking: https://www.youtube.com/watch?v=rjXeG0aT-7w
<HR>

- [ ] Realizar pesquisa no Google sobre o assunto da notícia
    - Pesquisar no Google: https://www.youtube.com/watch?v=IBhdLRheKyM
<HR>


*Sites para captura de notícias para teste:*  
- Notícias do Lupa: https://piaui.folha.uol.com.br/lupa/2022/04/19/verificamos-trf-drogas-traficantes/
- Notícias do G1: https://g1.globo.com/fato-ou-fake/
- Boatos: https://www.boatos.com.br/
<HR>

*Insumos para produzir a documentação:*  

- Monografia exemplo: https://www.monografias.ufop.br/bitstream/35400000/3122/6/MONOGRAFIA_An%C3%A1liseNot%C3%ADciasFalsas.pdf  

- Explicação dos modelos escolhidos: https://simple.nama.ai/post/bert-x-t5-x-gpt-3  

- Explicação sobre a Bertimbau: https://neuralmind.ai/2020/11/29/bertimbau-da-neuralmind-e-recordista-em-downloads/  

- Explicação de Lematização: https://www.alura.com.br/artigos/lemmatization-vs-stemming-quando-usar-cada-uma  

- Projeto de detecção de fake news parado: https://jornal.usp.br/ciencias/ciencias-exatas-e-da-terra/ferramenta-para-detectar-fake-news-e-desenvolvida-pela-usp-e-pela-ufscar/ e https://nilc-fakenews.herokuapp.com/about  

- Projeto antigo no Kaggle: https://www.kaggle.com/code/aishwarya2210/prediction-of-fake-news-model-comparison

- Ferramentas para NLP em PT-BR: 
https://medium.com/turing-talks/ferramentas-para-processamento-de-linguagem-natural-em-portugu%C3%AAs-977c7f59c382
