import newspaper

cnn_paper = newspaper.build('https://g1.globo.com/fato-ou-fake/', language='pt')

for article in cnn_paper.articles[:3]:
    print(article.url)

article = newspaper.Article('http://www.e-farsas.com/respostas-para-10-das-duvidas-mais-recorrentes-sobre-o-e-farsas-e-as-fake-news.html')
article.download()
print(article.html)
article.parse()
print(article.text)

