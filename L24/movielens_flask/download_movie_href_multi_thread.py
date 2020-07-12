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
    #print(name)
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
df = pd.read_csv('movies.csv')
i = 0

"""
for title in df['title']:
    i = i + 1
    if i< 3298:
        continue
    print('downloading ' + title)

    movie_href, pic_href = get_movie_pic_and_url(title)
    #add_movie(title, movie_href, pic_href)
    update_movie(title, movie_href, pic_href)
"""
class myThread (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.movies = []
    def add_movie(self, title):
        self.movies.append(title)

    def run(self):
        print ("开始线程：{}".format(self.threadID))
        for title in self.movies:
            print('downloading ' + title)
            movie_href, pic_href = get_movie_pic_and_url(title)
            #add_movie(title, movie_href, pic_href)
            update_movie(title, movie_href, pic_href)

        """
        for temp in range(min_id, max_id):
            title = df.iloc[temp]['title']
            print('downloading ' + title)
            movie_href, pic_href = get_movie_pic_and_url(title)
            #add_movie(title, movie_href, pic_href)
            update_movie(title, movie_href, pic_href)
        """

        #for title in df['title']:
        #print_time(self.name, self.counter, 5)
        print ("退出线程：" + str(self.threadID))

def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            threadName.exit()
        time.sleep(delay)
        print ("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1


# 创建线程队列
threads = []
sql_stmt = 'SELECT id, title, movie_id FROM movie WHERE (movie_href = ""  or movie_href is null) and (download=0)'
rows = session.execute(sql_stmt)
# 创建5个线程
for i in range(5):
    temp_thread = myThread(i+1)
    threads.append(temp_thread)

import random
for row in rows:
    #print(row['id'], row['title'], row['movie_id'])
    print(row['id'])
    # 随机分配给线程1-5
    random_thread = random.randint(1,5)
    threads[random_thread-1].add_movie(row['title'])

for i in range(5):
    threads[i].start()

for temp_thread in threads:
    temp_thread.join()

"""
ids = [(3510, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000), (8000, 9000), (9000, 9742)]
i = 1
for (min_id, max_id) in ids:
    print(min_id, max_id)
    temp_thread = myThread(i, min_id, max_id)
    temp_thread.start()
    i = i+1
    threads.append(temp_thread)

for temp_thread in threads:
    temp_thread.join()
"""