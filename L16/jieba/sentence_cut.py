import jieba
import jieba.posseg as pseg
from pprint import pprint

sentence = '美国新冠肺炎确诊超57万，纽约州死亡人数过万。当地时间4月13日，世卫组织发布最新一期新冠肺炎每日疫情报告。截至欧洲中部时间4月13日10时（北京时间4月13日16时），全球确诊新冠肺炎1773084例（新增76498例），死亡111652例（新增5702例）。其中，疫情最为严重的欧洲区域已确诊913349例（新增33243例），死亡77419例（新增3183例）。'
# 获取分词
#jieba.add_word('新冠肺炎')
seg_list = jieba.cut(sentence)
print(' '.join(seg_list))

# 获取分词和词性
words = pseg.lcut(sentence)
temp = [(word, flag) for word, flag in words]
pprint(temp)

# 输出所有地名
temp = [(word, flag) for word, flag in words if flag=='ns']
print(temp)
