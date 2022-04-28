
import flask
from flask import Blueprint, render_template, request

noticia_revisao_blueprint = Blueprint('noticia_revisao_blueprint', __name__, template_folder='templates')


@noticia_revisao_blueprint.route('/noticia_revisao')
def noticia_revisao():
    url_noticia = request.args['url_noticia']
    uuid = request.args['uuid']
    base_url = flask.url_for("index_blueprint.index", _external=True)
    full_uuid = f'{base_url}{uuid}'
    full_uuid = full_uuid.replace('http://www.', '')

    return render_template('noticia_revisao.html',
                           url_noticia=url_noticia,
                           uuid=uuid,
                           full_uuid=full_uuid)
