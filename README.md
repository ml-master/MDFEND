学生姓名：唐璟玥

复现论文：MDFEND: Multi-domain Fake News Detection（CIKM2021）

if you want to know more detail about this project, you can open the pdf of `MDFEND_detail.pdf`

# MDFEND: Multi-domain Fake News Detection
This is an official implementation for [**MDFEND: Multi-domain Fake News Detection**](https://dl.acm.org/doi/abs/10.1145/3459637.3482139) which has been accepted by CIKM2021.
## Dataset
The splited dataset (i.e., train, val, test) are in the `MDFEND-Weibo21/data/weibo21` folder.

The new dataset introduced in class are in the `MDFEND-Weibo21/data/gossip` folder.

You can have access to the original dataset of Weibo21 only after an ["Application to Use Weibo21 for Fake News Detection"](https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAO__Q4mnQlURFcxUTBYOEZSWEk1SFA2Q1BRRDhaOTRQQi4u) has been submitted. 
## Code
### Requirements
Refer to requirements.txt

You can run `pip install -r requirements.txt` to deploy the environment quickly.
### pretrained_model 
You can download pretrained model (Roberta) from https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing and move all the files in the folder into the path `MDFEND-Weibo21/pretrained_model/chinese_roberta_wwm_base_ext_pytorch`.

You can download pretrained model (bert-base-case) from https://modelscope.cn/models/AI-ModelScope/bert-base-cased/filesand move all the files in the folder into the path `MDFEND-Weibo21/pretrained_model/bert-base-cased`.

### Data Preparation
After you download the **Weibo21** dataset, move the `train.pkl`, `val.pkl` and `test.pkl` into the path `MDFEND-Weibo21/data`.

Use the preprocessing described in the `/data/readme.md` to get the new combined data set introduced in class.

### Run
You can run this model through(new gossip datasets introduced in  class):
```python
python main.py --model_name mdfend 
```
if you want to run the origin code from the origin paper(MFDEND),you can download the project from github https://github.com/kennqiang/MDFEND-Weibo21 or you can open this project and restore the weibo21 code commented in it.

### Train_result_plot folder

Weibo21 data and the new data set recorded the loss and evaluation indicators during each training round. The visual line chart of loss is drawn using plot.py.


### param_model

This folder MDFEND stores the model parameters of the original paper used weibo21 dataset.
The gossip folder holds the model parameters for the new dataset for the class.



## Reference

```
Nan Q, Cao J, Zhu Y, et al. MDFEND: Multi-domain Fake News Detection[C]//Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021: 3343-3347.
```
or in bibtex style:
```
@inproceedings{nan2021mdfend,
  title={MDFEND: Multi-domain Fake News Detection},
  author={Nan, Qiong and Cao, Juan and Zhu, Yongchun and Wang, Yanyan and Li, Jintao},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3343--3347},
  year={2021}
}
```

