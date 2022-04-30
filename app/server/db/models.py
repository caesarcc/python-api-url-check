
import string
import uuid
from datetime import datetime
from random import choices

from .extensions import db


class Noticia(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(16), unique=True)
    url = db.Column(db.String(512), nullable=True)
    resumo = db.Column(db.Text(), nullable=True)
    #visitas = db.Column(db.Integer, default=0)
    data_acesso = db.Column(db.DateTime, default=datetime.now)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uuid = uuid.uuid4()
        #url = self.query.filter_by(uuid=uuid).first()

    def __repr__(self):
        return f'{self.url}, {self.uuid}'
