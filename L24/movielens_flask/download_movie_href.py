import requests
from lxml import etree
from urllib.parse import urljoin
import pandas as pd

from sqlalchemy import Column, String, Integer, DateTime, UniqueConstraint, create_engine
from sqlalchemy.orm import sessionmaker

# 创建&返回session
def get_db_session():
    engine = create_engine('mysql+mysqlconnector://root:passw0rdcc4@localhost:3306/movielens')
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    # 创建session对象:
    session = DBSession()
    #sql_stmt = "SET SESSION sql_mode='STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'"
    #session.execute(sql_stmt)

    return engine, session

# 添加电影链接及图片链接
def add_movie(title, movie_href, pic_href):
    insert_stmt = 'insert ignore into movie (title, movie_href, pic_href) VALUES \
                    (:title, :movie_href, :pic_href)'
    session.execute(insert_stmt, {'title': title, 'movie_href': movie_href, 'pic_href': pic_href})
    session.commit()

# 更新电影的movie_href, pic_href
def update_movie(title, movie_href, pic_href):
    update_stmt = 'UPDATE movie SET movie_href = :movie_href, pic_href = :pic_href, download = :download WHERE title = :title'
    session.execute(update_stmt, {'title': title, 'movie_href': movie_href, 'pic_href': pic_href, 'download': 1})
    session.commit()

def get_movie_pic_and_url(name):
    url = 'https://www.imdb.com/find?ref_=nv_sr_fn&q='+name
    print(name)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
    search_html = requests.get(url, headers=headers).content
    search_html_x = etree.HTML(search_html)
    try:
        movie_href=urljoin(url, search_html_x.xpath('//table[@class="findList"]//a[1]/@href')[0])

        movie_html = requests.get(movie_href, headers=headers).content
        movie_html_x = etree.HTML(movie_html)
        pic_href = movie_html_x.xpath('//div[@class="poster"]//img/@src')[0]
    except:
        movie_href, pic_href = '', ''
    return movie_href, pic_href

#print(get_movie_pic_and_url("Schindler's List (1993)"))

import threading

# 获取mysql session
engine, session = get_db_session()
# 数据加载
"""
df = pd.read_csv('movies.csv')
i = 0
for title in df['title']:
    i = i + 1
    if i< 3298:
        continue
    print('downloading ' + title)

    movie_href, pic_href = get_movie_pic_and_url(title)
    #add_movie(title, movie_href, pic_href)
    update_movie(title, movie_href, pic_href)
"""
title = 'Victory (a.k.a. Escape to Victory) (1981)'
movie_href, pic_href = get_movie_pic_and_url(title)
update_movie(title, movie_href, pic_href)
