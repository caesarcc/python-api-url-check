
import string
from datetime import datetime
from random import choices

from .extensions import db


class Noticia(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(5), unique=True)
    url = db.Column(db.String(512))
    resumo = db.Column(db.Text(), nullable=True)
    #visitas = db.Column(db.Integer, default=0)
    data_acesso = db.Column(db.DateTime, default=datetime.now)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uuid = self.generate_uuid()

    def generate_uuid(self):
        characters = string.digits + string.ascii_letters
        uuid = ''.join(choices(characters, k=5))
        url = self.query.filter_by(uuid=uuid).first()

        if url:
            return self.generate_uuid()

        return uuid

    def __repr__(self):
        return f'{self.url}, {self.uuid}'
