import newspaper
import requests
from requests.exceptions import ConnectionError, InvalidURL, MissingSchema

#article = newspaper.Article('https://g1.globo.com/politica/noticia/2022/05/10/senado-aprova-elevar-para-70-anos-idade-limite-para-nomeacao-no-stf-e-em-tribunais-superiores.ghtml', language='pt')
#article.download()
#article.parse()
#print(article.text)

#res = requests.get('https://g1.globo.co/politica/noticia/2022/05/10/senado-aprova-elevar-para-70-anos-idade-limite-para-nomeacao-no-stf-e-em-tribunais-superiores.ghtml')

article = newspaper.Article('https://g1.globo.co/politica/noticia/2022/05/10/senado-aprova-elevar-para-70-anos-idade-limite-para-nomeacao-no-stf-e-em-tribunais-superiores.ghtml', language='pt')
try:
    res = requests.get('https://g1.globo.com/politica/noticia/2022/05/10/senado-aprova-elevar-para-70-anos-idade-limite-para-nomeacao-no-stf-e-em-tribunais-superiores.ghtml')
    article.download()
    article.parse()
    print(article.text)
except (MissingSchema, ConnectionError, InvalidURL) as ex:
    print(ex)
