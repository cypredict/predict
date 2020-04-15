import pandas as pd
import numpy as np
import featuretools as ft
 
# 数据加载，parse_dates为解析时间格式
clients = pd.read_csv('clients.csv', parse_dates = ['joined'])
loans = pd.read_csv('loans.csv', parse_dates = ['loan_start', 'loan_end'])
payments = pd.read_csv('payments.csv', parse_dates = ['payment_date'])

#创建一个空的实体集
es = ft.EntitySet(id="clients")
#clients指定索引为client_id，时间索引字段为joined
es = es.entity_from_dataframe(entity_id = 'clients', dataframe = clients, index = 'client_id', time_index = 'joined')
#print(es['clients'])
#print(clients)

# payments数据表，make_index=True表示没有原始索引列，新建索引payment_id，指定missed是一个类别特性，时间索引字段为payment_date
es = es.entity_from_dataframe(entity_id = 'payments', dataframe = payments, variable_types = {'missed': ft.variable_types.Categorical}, make_index = True, index = 'payment_id', time_index = 'payment_date')
#print(es['payments'])
#print(es)

#loans指定索引为loan_id，repaid是一个类别特性，时间索引为loan_start
es = es.entity_from_dataframe(entity_id = 'loans', dataframe = loans, variable_types = {'repaid': ft.variable_types.Categorical}, index = 'loan_id', time_index = 'loan_start')

# 将两张表进行关联
es=es.add_relationship(ft.Relationship(es["clients"]["client_id"], es["loans"]["client_id"]))
#r_payments = ft.Relationship(es['loans']['loan_id'], es['payments']['loan_id']) 
#es = es.add_relationship(r_payments)
es.add_relationship(ft.Relationship(es['loans']['loan_id'], es['payments']['loan_id']) )
print(es)

# 开始创建新特征，使用特征基元primitives
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
print(primitives)
primitives.to_csv('primitives.csv')
print(primitives[primitives['type'] == 'aggregation'].head(10))

# 使用特征基元创建新的特征
features, feature_names = ft.dfs(entityset = es, target_entity = 'clients', 
                                 agg_primitives = ['mean', 'max', 'percent_true', 'last'],
                                 #trans_primitives = ['years', 'month', 'subtract', 'divide'])
                                 trans_primitives = ['year', 'month', 'subtract_numeric', 'divide_numeric'])


# 打印一阶特征
print(pd.DataFrame(features['MEAN(loans.loan_amount)'].head(3)))
# 打印二阶特征
print(pd.DataFrame(features['LAST(loans.MEAN(payments.payment_amount))'].head(3)))
# 深度特征组合，最大深度为2
features, feature_names = ft.dfs(entityset=es, target_entity='clients', max_depth = 2)


print('features=', features)
print('feature_names=', feature_names)
#features.to_csv('features.csv')
