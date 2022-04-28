
import json

import flask
import requests
from flask import Blueprint, redirect, request, url_for

capturar_url_blueprint = Blueprint('capturar_url_blueprint', __name__, template_folder='templates')


@capturar_url_blueprint.route('/capturar', methods=['POST'])
def capturar_url():
    base_url = flask.url_for("index_blueprint.index", _external=True)
    url_noticia = request.form['url_noticia']
    capturar_endpoint = base_url + 'api/capturar'

    params = {
        'url': url_noticia
    }

    headers = {
        'content-type' : 'application/json'
    }

    response = requests.post(capturar_endpoint, headers=headers, json=params)

    if response.status_code == 200:
        response = json.loads(response.text)
        uuid = response['uuid']
        url_noticia = response['url_noticia']

        return redirect(url_for('noticia_revisao_blueprint.noticia_revisao',
                                uuid=uuid,
                                url_noticia=url_noticia))
    else:
        return redirect(url_for('error_blueprint.error'))
