# 生成数据：clients, loans, payments
import pandas as pd
import numpy as np
from datetime import datetime
import random
 
rand_dates = []
# 生成随机日期
for _ in range(1000):
  year = random.choice(range(2000, 2015))
  month = random.choice(range(1, 13))
  day = random.choice(range(1, 29))
  rdate = datetime(year, month, day)
  rand_dates.append(rdate)

# 创建clients数据表
clients = pd.DataFrame(columns = ['client_id', 'joined', 'income', 'credit_score'])
for _ in range(25):
  clients = clients.append(pd.DataFrame({'client_id': np.random.randint(25000, 50000, size = 1)[0], 'joined': random.choice(rand_dates),
                           'income': np.random.randint(30500, 240000, size = 1)[0], 'credit_score': np.random.randint(500, 850, size = 1)[0]}, index = [0]), ignore_index = True)
print(clients.head())

# 创建loans数据表 
loans = pd.DataFrame(columns = ['client_id', 'loan_type', 'loan_amount', 'repaid', 'loan_id', 'loan_start', 'loan_end', 'rate'])
for client in clients['client_id'].unique():
  for _ in range(20):
    time_created = pd.datetime(np.random.randint(2000, 2015, size = 1)[0],
                               np.random.randint(1, 13, size = 1)[0],
                               np.random.randint(1, 29, size = 1)[0]) 
    time_ended = time_created + pd.Timedelta(days = np.random.randint(500, 1000, size = 1)[0]) 
    loans = loans.append(pd.DataFrame({'client_id': client, 'loan_type': random.choice(['cash', 'credit', 'home', 'other']),
                                                         'loan_amount': np.random.randint(500, 15000, size = 1)[0],
                                                         'repaid': random.choice([0, 1]), 
                                                         'loan_id': np.random.randint(10000, 12000, size = 1)[0],
                                                         'loan_start': time_created,
                                                         'loan_end': time_ended,
                                                         'rate': round(abs(4 * np.random.randn(1)[0]), 2)}, index = [0]), ignore_index = True)
print(loans.head())
# 创建payments数据表
payments = pd.DataFrame(columns = ['loan_id', 'payment_amount', 'payment_date', 'missed']) 
for _, row in loans.iterrows():
  time_created = row['loan_start']
  payment_date = time_created + pd.Timedelta(days = 30)
  loan_amount = row['loan_amount']
  loan_id = row['loan_id']
  payment_id = np.random.randint(10000, 12000, size = 1)[0]
  for _ in range(np.random.randint(5, 10, size = 1)[0]):
    payment_id += 1
    payment_date += pd.Timedelta(days = np.random.randint(10, 50, size = 1)[0])
    payments = payments.append(pd.DataFrame({'loan_id': loan_id, 
                                                               'payment_amount': np.random.randint(int(loan_amount / 10), int(loan_amount / 5), size = 1)[0],
                                                               'payment_date': payment_date, 'missed': random.choice([0, 1])}, index = [0]), ignore_index = True)
print(payments.head())

# 去掉冗余数据
clients = clients.drop_duplicates(subset = 'client_id')
loans = loans.drop_duplicates(subset = 'loan_id')
# 保存创建文件 
clients.to_csv('clients.csv', index = False)
loans.to_csv('loans.csv', index = False)
payments.to_csv('payments.csv', index = False) 