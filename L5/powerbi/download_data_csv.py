import requests
import json
import pandas as pd

def get_html_text(url):
    try:
        res = requests.get(url,timeout = 30)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        return res.text
    except:
        return "Error"

def parse_data():
    result = []
    country_len = len(areaTree_json)
    for i in range(0, country_len):
        #如果为中国则说明具有省份信息
        if(areaTree_json[i]["name"]=="中国"):
            #获取省份长度
            province_len = len(areaTree_json[i]["children"])  
            for j in range(0,province_len):
                #获取地区长度
                area_len=len(areaTree_json[i]["children"][j]["children"])
                for n in range(0,area_len):
                    #获取地区的总体疫情情况
                    total=areaTree_json[i]["children"][j]["children"][n]["total"]
                    country = '中国'
                    province = areaTree_json[i]["children"][j]["name"]
                    area = areaTree_json[i]["children"][j]["children"][n]["name"]
                    confirm = total["confirm"]
                    dead = total["dead"]
                    heal = total["heal"]
                    suspect = total["suspect"]
                    temp = {'city':area, 'province':province, 'country':country, \
                        'confirm':confirm,'heal':heal,'dead':dead, 'suspect':suspect, 'update_time':update_time} 
                    result.append(temp)
    return result

page_url = "https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5"
#获取Json
text = get_html_text(page_url)
#将json数据中的data字段的数据提取处理
json_text = json.loads(text)["data"]
#将提取出的字符串转换为json数据
json_text = json.loads(json_text)
#更新时间
update_time = json_text["lastUpdateTime"]
#每日汇总信息
chinaTotal_json = json_text["chinaTotal"]
confirmCount = str(chinaTotal_json["confirm"])
suspectCount = str(chinaTotal_json["suspect"])
deadCount = str(chinaTotal_json["dead"])
heal = str(chinaTotal_json["heal"])

print("更新时间：" + update_time + "\n" + "确诊人数为：" + confirmCount + "人\n" + "死亡人数为：" + deadCount + "人\n" + \
    "疑似人数为：" + suspectCount + "人\n" + "治愈人数为：" + heal + "人\n" )

#包含国家、省份、地区的所有信息，且国家为首索引
areaTree_json=json_text["areaTree"]
#获取信息并获取长度
result = parse_data()
# 写入CSV
data = pd.DataFrame(result)
data.to_csv('city.csv')
