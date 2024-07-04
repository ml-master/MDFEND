
import json
import  random
category_cist = ["Technology", "Military", "Education and Test", "Disaster incident", "Politics", "Medicine and Health", "Financial Business", "Entertainment", "Social Life"]
# -------------------------------------------------
# 随机选择category
# with open('merge_whole.json','r',encoding='utf-8') as f:
#     data = json.load(f)
# print(len(data))
# for key,value in data.items():
#     print(key,value)
#     data[key]['category']  = random.choice(category_cist)
#
# print(data)
# with open("category_random.json", "w", encoding='utf-8') as outfiles:
#     json.dump(data, outfiles, indent=4, ensure_ascii=False)

# --------------------------------------------------------------------------------------------------------------------------------
# or 根据llm选择category
prompt = "Please read the above paragraph and decide which topic in the folloing list ['Technology','Military','Education and Testing','Disaster incident', 'Politics', 'Medicine and Health', 'Financial Business', 'Entertainment', 'social life'] is the most relevant to the meaning of the above paragraph. You only need to choose the only one of the most relevant topic string name in the above list  and answer one topic string name in the above list to output Please Don't print anything extra, no more than 3 words."
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(r"/root/autodl-tmp/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained(r"/root/autodl-tmp/chatglm3-6b", trust_remote_code=True).half().cuda()
model = model.eval()

def llm(content):
    prompt_llm = content+prompt
    print("prompt_llm:",prompt_llm)
    response, history = model.chat(tokenizer, prompt_llm,history=None)
    return response

with open('merge_whole.json','r',encoding='utf-8') as f:
    data = json.load(f)
# print(data)
f.close()

for key,value in data.items():
    print("---------------")
    print('现在已经进行到：',key)
    response = llm(value['content'])
    print("respnse:",response)
    data[key]['category'] = response
# print(data)

with open('category.json','w',encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
f.close()
# --------------------------------------------------------------------------------------------------------------------------------
# 数据预处理（将生成不准确的种类替换为entertainment，因为大多数都是娱乐新闻）
with open('category.json','r',encoding='utf-8') as f:
    data = json.load(f)
for key,value in data.items():
    print(key,value)
    if value['category'] not in category_cist:
        # 将生成不准确的种类替换为entertainment，因为大多数都是娱乐新闻
        data[key]['category']= 'Entertainment'
with open("category_change_no2entertainment.json", "w", encoding='utf-8') as outfiles:
    json.dump(data, outfiles, indent=4, ensure_ascii=False)
# ----------------------------------------------------------------------------------------------------------------------------------
# 转变为train,valie test,pkl文件
import pandas as pd
import  random
with open('category_change_no2entertainment.json','r',encoding='utf-8') as f:
    data = json.load(f)
data_list_of_dicts =[]
data_length = len(data)
    # 划分训练集，测试集，验证集
train_len = int(data_length * 0.6)
test_len = int(data_length * 0.2)

for key,value in data.items():
    # print(key,value)
    data_list_of_dicts.append(value)

df = pd.DataFrame(data_list_of_dicts)
print(df)
idx = list(df.index)
random.shuffle(idx)  # 将index列表打乱
print(idx)
df_train = df.loc[idx[:train_len]]
df_test = df.loc[idx[train_len:train_len + test_len]]
df_valid = df.loc[idx[train_len + test_len:]]  # 剩下的就是valid

print(df_train)
print(df_test)
print(df_valid)

# 写入pkl
import  pickle
with open('gossip/train_me_gossip.pkl', 'wb') as f:
    pickle.dump(df_train, f)
f.close()

with open('gossip/test_me_gossip.pkl', 'wb') as f:
    pickle.dump(df_test, f)
f.close()

with open('gossip/val_me_gossip.pkl', 'wb') as f:
    pickle.dump(df_valid, f)
f.close()
# -----------------------------------------------------------------------------------------------------------------



