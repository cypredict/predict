#coding=utf-8
from flask_login import LoginManager,login_user,UserMixin,logout_user,login_required
from run import login_manger
from run import db

from sqlalchemy.ext.declarative import declarative_base
#from .ext import db
Base = declarative_base()
metadata = Base.metadata


class Users(UserMixin,db.Model):
    __tablename__ = 'user'#对应mysql数据库表
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    password = db.Column(db.String(64), unique=True, index=True)
    email = db.Column(db.String(64), unique=True, index=True)
    def __init__(self,username,password):
        self.username=username
        self.password=password
        
    def get_id(self):
        return self.id
        #return unicode(self.id)

    def __repr__(self):
        return '<User %r>' % self.username

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False


class Movie(Base,db.Model):
    __tablename__ = 'movie'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.VARCHAR(255))
    movie_href = db.Column(db.VARCHAR(255))
    pic_href = db.Column(db.VARCHAR(255))
    movie_id = db.Column(db.Integer)
