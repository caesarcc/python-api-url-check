
import os

from app.server.api.api import Captura, confirmacao
from app.server.db.extensions import db
from app.server.db.models import Noticia
from app.server.routes.confirmacao import confirmacao_blueprint
from app.server.routes.error import error_blueprint
from app.server.routes.index import index_blueprint
from app.server.routes.internal.capturar_url import capturar_url_blueprint
from app.server.routes.internal.favicon import app_blueprint
from app.server.routes.noticia_revisao import noticia_revisao_blueprint
from app.server.routes.page_not_found import page_not_found_blueprint
from dotenv import load_dotenv
from flask import Flask
from flask_restful import Api


def create_app(config_file):
    app_path = os.path.dirname(os.path.abspath(__file__))
    project_folder = os.path.expanduser(app_path)
    load_dotenv(os.path.join(project_folder, '.env'))

    app = Flask(__name__, template_folder='../client/templates', static_folder='../client/static')
    api = Api(app)
    app.config.from_pyfile(config_file)

    db.init_app(app)

    with app.app_context():

        noticia = db.Model.metadata.tables['noticia']
        noticia.create(bind=db.engine, checkfirst=True)

        api.add_resource(Captura, '/api/capturar')
        api.add_resource(confirmacao, '/api/confirmacao')

        app.register_blueprint(index_blueprint)
        app.register_blueprint(page_not_found_blueprint)
        app.register_blueprint(noticia_revisao_blueprint)
        app.register_blueprint(confirmacao_blueprint)
        app.register_blueprint(error_blueprint)
        app.register_blueprint(app_blueprint) #favicon
        app.register_blueprint(capturar_url_blueprint)
        return app
