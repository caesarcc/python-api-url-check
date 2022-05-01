import os

basedir = os.path.dirname(__file__)
rootdir, basedir = os.path.split(basedir)
while basedir != 'app':
    rootdir, basedir = os.path.split(rootdir)
dblocaldir = os.path.join(rootdir, 'dados\\fakenews_check.db')
SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI') or "sqlite:///" + dblocaldir
SQLALCHEMY_TRACK_MODIFICATIONS = False
