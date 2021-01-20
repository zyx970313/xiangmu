import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# corpus=["圆筒 不在位 F计算机 屏幕显示 1-4位置 均为空 F组合 上 四个圆筒 Q飞灯 均不亮"
#         "电B通路 检查故障 程序执行过程中 检测 初始状态 异常 F计算机 屏幕显示 “X#圆筒 电B通路 控制故障” “X#圆筒B组 激活检查故障” 或“X#号圆筒舵机 激活检查故障”"
#         "F解锁故障（竖起不到位） 程序执行过程中 检测 F解锁故障 F计算机屏幕 显示 “F解锁故障！！竖起不到位” 信息"
#         "手动操作 电机不能启动 将 手控台 置于“手动” 打开 “电机启动”开关 电机不能正常启动"
#         "程控 电机不能启动 将 手控台 置于“程控”， 打开 “电机启动”开关 ， 电机不能正常启动"
#         "手动状态 下 液压系统 无法建压 ： 将 手控台 置于 “手动” ， 打开 “电机启动”开关 ， 电机正常启动 ，但 调节 “调压” 电位器系统 不建压"
#         "吊机 红色报警灯 故障 ： ZT车 吊装过程 中，随车吊机 红色报警灯 亮起"
#         "吊具 刚柔转换钢丝绳 不同步故障 ： ZT车 吊装过程 中，吊具 刚柔转换钢丝绳 不同步"
#         "数传故障 ： HL单元 功能检查 时，单路 有线不通"
#         "圆筒 无法锁紧 或 解锁故障 ： 运输车 正常行驶 后，进行 圆筒 的 吊卸， 使用 手柄组合 无法 将 圆筒 锁紧机构 松开，或 将 圆筒 吊装 到 运输车上 后， 用 手柄 对 圆筒 进行 锁定，锁定无法完成"
# ]
corpus=[]

for i in open("KB_query\\external_dict\\features.txt",encoding='utf-8'):
    corpus.append(i)

# print(corpus)

corpus_sol = []

for i in open("KB_query\\external_dict\\solutions.txt",encoding='utf-8'):
    corpus_sol.append(i)

#
# def corpuslist():
#     corpuslist = [line.strip() for line in open('corpus.txt', encoding='UTF-8').readlines()]
#     return corpuslist

 # 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('KB_query\\external_dict\\delete.txt', encoding='UTF-8').readlines()]
    return stopwords


# 对句子进行中文分词
def seg_depart(sentence):
    jieba.load_userdict("KB_query\\external_dict\\features.txt")
    sentence_depart = jieba.cut(sentence.strip())
    stopwords = stopwordslist()
     # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


# 对句子进行中文分词
def seg_depart1(sentence):
    # 对文档中的每一行进行中文分词
    jieba.load_userdict("KB_query\\external_dict\\solutions.txt")
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
     # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

def tfi(text):
    line_seg = seg_depart(text)
    print('line_seg',line_seg)
    len_line_seg=len(line_seg.split( ))
    print(line_seg.split( ))
    corpus.append(line_seg)
    # print('corpus',corpus)
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    arr_last = len(weight)

    list1 = []
    for j in range(len(word)):
        # print(word[j], weight[arr_last - 1][j])
        list1.append(weight[arr_last - 1][j])
    #list1：输入句子分词后在词袋中的权重
    # print('list1',list1)
    list2 = sorted(list1, reverse=True)
    # print('list2',list2)
    list3 = np.argsort(list1)
    # print('list3',list3)
    list4 = []
    for a in list3:
        list4.append(a)
    # print('list4',list4)
    list4.reverse()
    # print('list4',list4)
    list5=[]
    for b in list4[:]:
        # print(b)
        if list1[b]!=0:
            list5.append(word[b])
            # print('word[b]',word[b])
    len_list5=len(list5)
    print('len_list5',len_list5)
    return list5


def tfi1(text):
    line_seg = seg_depart1(text)
    # print('line_seg', line_seg)
    len_line_seg = len(line_seg.split())
    # print(line_seg.split())
    corpus_sol.append(line_seg)

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus_sol))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    arr_last = len(weight)
    list1 = []
    for j in range(len(word)):
        # print(word[j], weight[arr_last - 1][j])
        list1.append(weight[arr_last - 1][j])
    list2 = sorted(list1, reverse=True)
    # print('list2',list2)
    list3 = np.argsort(list1)
    # print('list3',list3)
    list4 = []
    for a in list3:
        list4.append(a)
    # print(list4)
    list4.reverse()
    # print(list4)
    list5=[]
    for b in list4[:]:
        if list1[b]!=0:
            list5.append(word[b])
            # print('word[b]',word[b])
    len_list5=len(list5)
    print('len_list5',len_list5)
    return list5


if __name__ == "__main__":
    text =  "节流阀全开位置全闭位置卡死不流通111"
    text1="检查连接比例溢流阀YV1电缆插头；检查更换“调压”电位器 检查更换车辆转接盒；检查更换比例溢流阀YV1"
    list_res=tfi(text)
    list_res1=tfi1(text1)
    print(list_res)
    print(list_res1)
