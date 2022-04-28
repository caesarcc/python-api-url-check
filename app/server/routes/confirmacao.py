import json

import flask
import requests
from flask import Blueprint, render_template, request

confirmacao_blueprint = Blueprint('confirmacao_blueprint', __name__, template_folder='templates')


@confirmacao_blueprint.route('/confirmacao')
def confirmacao():
    uuid = request.args['uuid']
    base_url = flask.url_for("index_blueprint.index", _external=True)
    confirmacao_endpoint = base_url + 'api/confirmacao'

    params = {
        'url': uuid
    }

    response = requests.get(confirmacao_endpoint, params=params)
    total_url_clicks = json.loads(response.text)['total']
    return render_template('obrigado.html', confirmacao=total_url_clicks)
