import  json
import pandas as pd
import pickle

merge_whole_dict = {}
# 处理合成的假新闻
json_file_fake = 'gossipcop_v3-3_integration_based_fake_tn200.json'
with open(json_file_fake,'r',encoding='utf-8') as f_fake:
    data_fake = json.load(f_fake)
content_list_fake = []
label_list_fake = []
for index1,key in enumerate(data_fake):
    content_list_fake.append(data_fake[key]["generated_text"])
    label_list_fake.append(0)
fake_length = len(content_list_fake)
for i in range(len(content_list_fake)):
    merge_whole_dict[i] = {"content":content_list_fake[i],'label':label_list_fake[i]}

# 处理合成的真新闻
json_file_real = 'gossipcop_v3-7_integration_based_legitimate_tn300.json'
with open(json_file_real,'r',encoding='utf-8') as f_real:
    data_real = json.load(f_real)
content_list_real = []
label_list_real = []
for index1,key in enumerate(data_real):
    # print("index:%s key: %s  value:%s"%(index1,key,data[key]))
    content_list_real.append(data_real[key]["generated_text_t01"])
    label_list_real.append(1)
true_length = len(content_list_real)
for i in range(len(content_list_real)):
    merge_whole_dict[i+fake_length] = {"content":content_list_real[i],'label':label_list_real[i]}

print(fake_length,true_length)


with open("merge_whole.json", "w",encoding='utf-8') as outfiles:
        json.dump(merge_whole_dict, outfiles, indent = 4,ensure_ascii=False)