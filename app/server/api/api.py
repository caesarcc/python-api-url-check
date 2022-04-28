
import requests
from app.server.db.extensions import db
from app.server.db.models import Noticia
from flask_restful import Resource, reqparse

captura_parser = reqparse.RequestParser()
confirmacao_parser = reqparse.RequestParser()

captura_parser.add_argument('url', type=str, help='Parametro URL não informado', required=True)


class Captura(Resource):

    @staticmethod
    def post():
        args = captura_parser.parse_args()
        url = args['url']
        url_noticia = url if url.startswith('http') else ('http://' + url)

        try:
            res = requests.get(url_noticia)

            if res.status_code == 200:
                noticia = Noticia(url=url)

                db.session.add(noticia)
                db.session.commit()

                return dict(
                    uuid=noticia.uuid,
                    url_noticia=noticia.url,
                    success=True
                ), 200

            else:
                return dict(
                    success=False,
                    message='não foi possível capturar a notícia (page_not_found)'
                ), 404

        except (requests.exceptions.MissingSchema, requests.exceptions.ConnectionError, requests.exceptions.InvalidURL):
            return dict(
                success=False,
                message='não foi possível capturar a notícia (page_not_found)'
            ), 404


class confirmacao(Resource):

    @staticmethod
    def get():
        args = confirmacao_parser.parse_args()
        uuid = args['uuid'].split('/')[-1]

        try:
            url = Noticia.query.filter_by(uuid=uuid).first()

            return dict(
                total=0,
                uuid=url.uuid,
                url_noticia=url.url,
                success=True
            ), 200

        except AttributeError:
            return dict(
                success=False,
                message='não foi possível confirmar a avaliação (page_not_found)'
            ), 404
