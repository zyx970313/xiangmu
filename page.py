"""
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import jieba
from KB_query.word_tagging import Tagger
import KB_query.jena_sparql_endpoint
import KB_query.question2sparql
import xlrd
import time
import pymysql
import pandas as pd
from KB_query.jena_sparql_endpoint import JenaFuseki
from KB_query.question2sparql import Question2Sparql
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
import pywinauto
from pywinauto.keyboard import send_keys
from selenium.webdriver import Chrome, ChromeOptions
from aip import AipOcr
import sys,os
from urllib.request import  urlopen
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tf import tfi,tfi1

#提取错误特征
def fault_cut(s):
    tagger = Tagger(
        ['./KB_query/external_dict/f1.txt', './KB_query/external_dict/f2.txt', './KB_query/external_dict/f3.txt',
         './KB_query/external_dict/f4.txt',
         './KB_query/external_dict/f5.txt'])
    list1 = []
    for i in tagger.get_word_objects(s):
        print(i.token.decode('utf8'), i.pos)
        if (i.pos == 'one'):
            list1.append(i.token.decode('utf-8'))
        if (i.pos == 'two'):
            list1.append(i.token.decode('utf-8'))
        if (i.pos == 'three'):
            list1.append(i.token.decode('utf-8'))
        if (i.pos == 'four'):
            list1.append(i.token.decode('utf-8'))
        if (i.pos == 'five'):
            list1.append(i.token.decode('utf-8'))
    return list1

#提取解决方案特征
def solution_cut(s):
    tagger = Tagger(
        ['./dict/s1.txt', './dict/s2.txt', './dict/s3.txt', './dict/s4.txt', './dict/s5.txt'])
    list1 = ['', '', '', '', '']
    for i in tagger.get_word_objects(s):
        print(i.token.decode('utf8'), i.pos)
        if (i.pos == 'six'):
            list1[0] = i.token.decode('utf-8')
        if (i.pos == 'seven'):
            list1[1] = i.token.decode('utf-8')
        if (i.pos == 'eight'):
            list1[2] = i.token.decode('utf-8')
        if (i.pos == 'nine'):
            list1[3] = i.token.decode('utf-8')
        if (i.pos == 'ten'):
            list1[4] = i.token.decode('utf-8')
    return list1

# 建立语料库
corpus = {
    "F电源HL单元功能圆筒不在位飞灯不亮F计算机": "供电开关位置F电源F电源保险管F电源控制模块F电源模块",
    "F组合车辆功能检查电B通路控制故障圆筒B组激活圆筒舵机激活": "更换F组合相应圆筒控制板相应圆筒IO插板相应圆筒模拟板",
    "F计算机Y协同上课F解锁故障程序执行过程竖起不到位": "检查控制机柜输出信号1X11插头6、7两点5#424插板",
    "手控台程控电机不能启动手动操作车辆简程检查": "检查功能“电机启动”开关控制箱继电器W128电缆电源箱",
    "手控台车辆功能检查电机不能启动程控操作": "开关量输入板III开关量输出板“电机启动”开关功能控制箱继电器W128电缆电源箱",
    "手控台车辆功能检查液压系统无法建压手动状态": "比例溢流阀YV1电缆插头“调压”电位器车辆转接盒比例溢流阀YV1",
    "吊机吊装操作红色报警灯故障吊装过程": "操作吊机减小起重力矩报警指示灯OLP按钮厂家解决",
    "吊具吊装操作刚柔转换钢丝绳不同步吊装过程": "检查钢丝绳调整螺母位置刚柔转换油缸维修",
    "数传系统HL单元功能单路有线不通": "检查参数设置电缆连接更换接口更换从站有线数传单元",
    "停放架锁紧机构XL操作圆筒无法锁紧吊卸过程": "检查更换手柄锁止机构销轴运动正常"
}




# 建立词典，对于所有语料库中的问题进行jieba分词
all_question = ""
for question in corpus:
    all_question += question
dictionaries=fault_cut(all_question)
#dictionaries = list(set(jieba.cut(all_question)))  # 加上set是为了方便去重




# 单个词典转换为向量
def transform_vector(date):
    vector_list = []
    for wd in dictionaries:
        if wd in list(jieba.cut(date)):
            vector_list.append(1)
        else:
            vector_list.append(0)
    return np.array(vector_list).reshape(1, -1)  # .reshape(1,-1)是为了后期余弦计算


# 单个余弦相似度计算
def get_cosine(user_question, corpus_question):
    similar_list = cosine_similarity(transform_vector(user_question), transform_vector(corpus_question))
    similar_num = similar_list[0][0]  # 相似度
    return similar_num


# 计算语料库中的所有相似度
def get_corpus_consine(user_question):
    ori_question_dict = {}
    ori_answer_dict = {}
    similar_list = []  # 保存所有余弦值
    for key in corpus:
        similar_num = get_cosine(user_question, key)  # 获得余弦值
        similar_list.append(similar_num)  # 保存所有余弦值
        ori_question_dict[similar_num] = key  # 获取原问题并储存
        ori_answer_dict[similar_num] = corpus[key]  # 获取原答案并储存
    return similar_list, ori_question_dict, ori_answer_dict


# 得到最佳回答
def get_best_answer(similar_list, ori_question_dict, ori_answer_dict):
    max_similar = max(similar_list)
    if max_similar == 0: # 没有匹配项
        best_similar = 0
    else:
        best_similar = max_similar
        best_question = ori_question_dict[max_similar]
        best_answer = ori_answer_dict[max_similar]
    return [best_similar, best_question, best_answer]


# 得到三个近似回答
def get_three_answer(similar_list, ori_question_dict, ori_answer_dict):
    print(similar_list)
    three_similar =sorted(similar_list,reverse=True)
    best_similar = three_similar[0]
    best_question = ori_question_dict[best_similar]
    best_answer = ori_answer_dict[best_similar]
    second_similar =three_similar[1]
    second_question = ori_question_dict[second_similar]
    second_answer = ori_answer_dict[second_similar]
    third_similar = three_similar[2]
    third_question = ori_question_dict[third_similar]
    third_answer = ori_answer_dict[third_similar]
    return [best_similar,best_question,best_answer,second_similar,second_question,second_answer, third_similar,third_question,third_answer]



def upload_files(a):
    url = "http://localhost:3030/dataset.html?tab=upload&ds=/faults3.html"
    opt = ChromeOptions()  # 创建Chrome参数对象
    opt.headless = True  # 把Chrome设置成可视化无界面模式
    browser = Chrome(options=opt)  # 创建Chrome无界面对象
    # 访问图片上传的网页地址
    browser.get(url=url)
    time.sleep(3)
    # 点击图片上传按钮，打开文件选择窗口
    browser.find_element_by_name("files[]").send_keys(r"D:\pycode\library\d2rq\d2rq-0.8.1\faults_new.nt")
    button = browser.find_element_by_css_selector("[class='btn btn-primary start action-upload-all']")
    button.click()
    browser.close()
    print(a)
    return url

def create_nt(a):
    command1 = "cd /d D:\pycode\library\d2rq\d2rq-0.8.1"
#    command2 = "generate-mapping -u root -p zhangyixin -o faults3.ttl jdbc:mysql:///faults2"
    command3 = r".\dump-rdf.bat -o faults_new.nt .\faults_new.ttl"
#    cmd = "{0} && {1} && {2}".format(command1, command2, command3)
    cmd = "{0} && {2}".format(command1, command3)
    os.system(cmd)
    print(a)
    return command1


app = Flask(__name__)
app.secret_key = 'test'
# 设置数据库的连接地址
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:zhangyixin@127.0.0.1:3306/faults2"
# 是否监听数据库变化  一般不打开, 比较消耗性能
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# 创建数据库操作对象(建立了数据库连接)
db = SQLAlchemy(app)


# 故障特征表
class Feature(db.Model):
    __tablename__ = "features"
    id = db.Column(db.Integer, primary_key=True)
    feature1 = db.Column(db.String(64))
    feature2 = db.Column(db.String(64))
    feature3 = db.Column(db.String(64))
    feature4 = db.Column(db.String(64))
    feature5 = db.Column(db.String(64))
    feature6 = db.Column(db.String(64))
    sols = db.relationship("Fea2Sol", backref="feature_record")  # 关系属性



# 关系表
class Fea2Sol(db.Model):
    __tablename__ = "fea2sol"
    sol_id = db.Column(db.Integer, db.ForeignKey("sol_features.id"), primary_key=True)
    fea_id = db.Column(db.Integer, db.ForeignKey("features.id"),primary_key=True)



# 解决方案特征表
class Sol_feature(db.Model):
    __tablename__ = "sol_features"
    id = db.Column(db.Integer, primary_key=True)
    sol_feature1 = db.Column(db.String(64))
    sol_feature2 = db.Column(db.String(64))
    sol_feature3 = db.Column(db.String(64))
    sol_feature4 = db.Column(db.String(64))
    sol_feature5 = db.Column(db.String(64))
    sol_feature6 = db.Column(db.String(64))







#精确查询中的五个特征及解决办法
y1,y2,y3,y4,y5,y6=" "," "," "," "," "," "

s1,s2,s3,s4,s5,s6="","","","","",""
#精确查询中每个特征在数据库中的数量
num1,num2,num3,num4,num5,num6="","","","","",""
#模糊查询中的三个案例的id及特征
id1,id2,id3="","",""
f1,f2,f3,f4,f5,f6=None,None,None,None,None,None
#批量上传数据
list_fs=[]
fuzz1=['','','','','','']
fuzz2=['','','','','','']
fuzz3=['','','','','','']




@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route("/flow", methods=['GET', 'POST'])
def test():
    return render_template("flow.html")

@app.route("/map", methods=['GET', 'POST'])
def map():
    return render_template("map.html")


@app.route("/single_fault",methods=['GET', 'POST'])
def single():
    return render_template("single_fault.html")

@app.route("/single_solution",methods=['GET', 'POST'])
def single_solution():
    return render_template("single_solution.html")


#流程图录入
@app.route("/flow_zhin",methods=['GET', 'POST'])
def flow_zhin():
    list1=["F计算机","1-4圆筒位置为空","1-4圆筒Q飞灯不亮"]
    list2=["更换","F电源保险管",""]
    list3=["F计算机","1-4圆筒位置为空","1-4圆筒Q飞灯不亮","更换F电源保险管","无效"]
    list4=["更换","F电源控制模块",""]
    list5=["F计算机","1-4圆筒位置为空","1-4圆筒Q飞灯不亮","更换F电源保险管","F电源控制模块","无效"]
    list6=["更换","F电源模块",""]
    return render_template("flow.html",a11=list1[0],a12=list1[1],a13=list1[2],
                           b11=list2[0],b12=list2[1],b13=list2[2],
                           a21=list3[0],a22=list3[1],a23=list3[2],a24=list3[3],a25=list3[4],
                           b21=list4[0],b22=list4[1],b23=list4[2],
                           a31=list5[0],a32=list5[1],a33=list5[2],a34=list5[3],a35=list5[4],a36=list5[5],
                           b31=list6[0],b32=list6[1],b33=list6[2]
                           )

@app.route("/flow_input",methods=['GET', 'POST'])
def flow_input():
    return render_template("flow.html")





#故障特征录入
@app.route("/input_fault" ,methods=['GET', 'POST'])
def input_fault():
    feature1 = request.form.get("Program_features1")
    feature2 = request.form.get("Program_features2")
    feature3 = request.form.get("Program_features3")
    feature4 = request.form.get("Program_features4")
    feature5 = request.form.get("Program_features5")
    feature6 = request.form.get("Program_features6")
    new_feature = Feature(feature1=feature1, feature2=feature2, feature3=feature3,
                          feature4=feature4, feature5=feature5, feature6=feature6
                          )
    db.session.add(new_feature)
    db.session.commit()
    flash("录入成功")
    return render_template("single_fault.html")



#故障特征智能提取
@app.route("/zhin",methods=['GET', 'POST'])
def zhin():

    s=request.form['maintenance_plan']
    print(s)
    if s =='':
        list_five=["","","","","",""]
    else:

        list_five=tfi(s)
        print(list_five)
        flash("识别成功")
    return render_template("single_fault.html",a=list_five[0],b=list_five[1],c=list_five[2],d=list_five[3])


#解决方案智能识别
@app.route("/zhin_sol",methods=['GET', 'POST'])
def zhin_sol():
    s=request.form.get("maintenance_plan")
    print(s)
    if s =='':
        list1=["","","","","","","",""]
    else:
        list1=tfi1(s)
        print(list1)
        flash("识别成功")
    return render_template("single_solution.html",a=list1[0],b=list1[1],c=list1[2],d=list1[3])


#解决方案录入
@app.route("/input_sol" ,methods=['GET', 'POST'])
def input_sol():

    feature1 = request.form.get("Sol_features1")
    feature2 = request.form.get("Sol_features2")
    feature3 = request.form.get("Sol_features3")
    feature4 = request.form.get("Sol_features4")
    feature5 = request.form.get("Sol_features5")
    feature6 = request.form.get("Sol_features6")
    new_feature = Sol_feature(sol_feature1=feature1, sol_feature2=feature2,
                              sol_feature3=feature3, sol_feature4=feature4,
                              sol_feature5=feature5, sol_feature6=feature6
                              )
    db.session.add(new_feature)
    db.session.commit()

    count = Feature.query.filter(Feature.id).count()
    new_fs = Fea2Sol(sol_id=count, fea_id=count)
    db.session.add(new_fs)
    db.session.commit()
    print("提交成功")
    create_nt(feature1)
    print("creat_nt")
    upload_files(feature1)
    print("upload")

    flash("录入成功")
    return render_template("single_solution.html")


#批量上传
@app.route("/batch_input",methods=['GET', 'POST'])
def batch_input():

    file = request.files['file']
    dst = os.path.join(os.path.dirname(__file__), file.filename)
    print(dst)
    book = xlrd.open_workbook(dst)
    # 获取所有的esheet
    list = book.sheets()
    sheet_names = book.sheet_names()
    # 建立mysql的连接
    conn = pymysql.connect(
        host='localhost',
        user='root',
        passwd='zhangyixin',
        db='faults2',
        port=3306,
        charset='utf8'
    )

    # 获得游标
    cur = conn.cursor()
    query = 'insert into features(feature1,feature2,feature3,feature4,feature5) values(%s, %s, %s, %s, %s);'
    query1 = 'insert into sol_features(sol_feature1,sol_feature2,sol_feature3,sol_feature4,sol_feature5) values(%s, %s, %s, %s, %s);'
    query2 = 'insert into fea2sol(sol_id,fea_id) values(%s, %s);'
    global list_fs
    list_fs=[]
    count = Feature.query.filter(Feature.id).count()

    for i in sheet_names:
        sheet_i = book.sheet_by_name(i)

        # 创建一个for循环迭代读取xls文件每行数据的, 从第二行开始是要跳过标题行
        for r in range(1, sheet_i.nrows):
            feature1 = sheet_i.cell(r, 0).value
            feature2 = sheet_i.cell(r, 1).value
            feature3 = sheet_i.cell(r, 2).value
            feature4 = sheet_i.cell(r, 3).value
            feature5 = sheet_i.cell(r, 4).value
            solution1=sheet_i.cell(r,5).value
            solution2 = sheet_i.cell(r, 6).value
            solution3 = sheet_i.cell(r, 7).value
            solution4 = sheet_i.cell(r, 8).value
            solution5 = sheet_i.cell(r, 9).value
            list_fs.append(feature1)
            list_fs.append(feature2)
            list_fs.append(feature3)
            list_fs.append(feature4)
            list_fs.append(feature5)
            list_fs.append(solution1)
            list_fs.append(solution2)
            list_fs.append(solution3)
            list_fs.append(solution4)
            list_fs.append(solution5)
            values = (feature1, feature2, feature3, feature4, feature5)
            values1=(solution1,solution2,solution3,solution4,solution5)
            values2=(count+int(r),count+int(r))
            # 执行sql语句
            cur.execute(query, values)
            cur.execute(query1,values1)
            cur.execute(query2,values2)

    cur.close()
    conn.commit()
    conn.close()
    flash("导入成功")
    create_nt()
    upload_files()
    return render_template("batch_fault.html")



@app.route("/batchData",methods=['GET', 'POST'])
def get_batchData():
    global list_fs
    print(list_fs)
    length=len(list_fs)
    print(length)
    group=(int)(length/10)
    print(group)
    links = [

        {"source": '案例1', "target": '1特征'},
        {"source": '案例1', "target": '1故障方案'},
        {"source": '特征', "target": list_fs[0]},
        {"source": '特征', "target": list_fs[1]},
        {"source": '特征', "target": list_fs[2]},
        {"source": '特征', "target": list_fs[3]},
        {"source": '特征', "target": list_fs[4]},
        {"source": '解决方案', "target": list_fs[5]},
        {"source": '解决方案', "target": list_fs[6]},
        {"source": '解决方案', "target": list_fs[7]},
        {"source": '解决方案', "target": list_fs[8]},
        {"source": '解决方案', "target": list_fs[9]},

    ]
    print(links)
    return json.dumps({'name': links})


@app.route("/batch_fault",methods=['GET', 'POST'])
def batch():
    return render_template("batch_fault.html")


@app.route("/fuzz_query",methods=['GET', 'POST'])
def fuzz():
    return render_template("fuzz_query.html")


#模糊查询
@app.route("/submit_fuzz",methods=['GET', 'POST'])
def submit_fuzz():
    global f1, f2, f3, f4, f5,f6
    f1,f2,f3,f4,f5,f6=None,None,None,None,None,None
    feature1 = request.form.get("fault_features1")
    feature2 = request.form.get("fault_features2")
    feature3 = request.form.get("fault_features3")
    feature4 = request.form.get("fault_features4")
    feature5 = request.form.get("fault_features5")
    feature6 = request.form.get("fault_features6")
    f1, f2, f3, f4, f5,f6=feature1,feature2,feature3,feature4,feature5,feature6
    features = feature1 + feature2 + feature3 + feature4 + feature5+feature6
    similar_list, ori_question_dict, ori_answer_dict = get_corpus_consine(features)  # 计算所有相似度
    result=get_three_answer(similar_list,ori_question_dict,ori_answer_dict)
    print(result)
    print(result[1])
    print(result[2])
    print(result[4])
    list1=tfi(result[1])
    print(list1)
    list2=tfi(result[4])
    list3=tfi(result[7])
    global id1,id2,id3,fuzz1,fuzz2,fuzz3


    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 passwd='zhangyixin',
                                 db='faults2',
                                 port=3306,
                                 charset='utf8'
                                 )
    cur = connection.cursor()  # 游标（指针）cursor的方式操作数据
    sql1 = 'SELECT id FROM features WHERE feature1="%s" and feature2="%s" and feature3="%s" and feature4="%s" and feature5="%s"  '%(list1[0],list1[1],list1[2],list1[3],list1[4])
    sql2 = 'SELECT id FROM features WHERE feature1="%s" and feature2="%s" and feature3="%s" and feature4="%s" and feature5="%s"' %(list2[0],list2[1],list2[2],list2[3],list2[4])
    sql3 = 'SELECT id FROM features WHERE feature1="%s" and feature2="%s" and feature3="%s" and feature4="%s" and feature5="%s"' %(list3[0],list3[1],list3[2],list3[3],list3[4])


    cur.execute(sql1)  # execute(query, args):执行单条sql语句。
    see1 = cur.fetchone()  # 使结果全部可看
    print(see1)
    cur.execute(sql2)  # execute(query, args):执行单条sql语句。
    see2 = cur.fetchone()  # 使结果全部可看
    print(see2)
    cur.execute(sql3)  # execute(query, args):执行单条sql语句。
    see3 = cur.fetchone()  # 使结果全部可看
    print(see3)
    id1=see1[0]
    id2=see2[0]
    id3=see3[0]
    connection.commit()
    connection.close()
    # TODO 连接Fuseki服务器。
    fuseki = JenaFuseki()
    # TODO 初始化自然语言到SPARQL查询的模块，参数是外部词典列表。
    q2s = Question2Sparql(
        ['./KB_query/external_dict/f1.txt', './KB_query/external_dict/f2.txt', './KB_query/external_dict/f3.txt',
         './KB_query/external_dict/f4.txt',
         './KB_query/external_dict/f5.txt'])

    my_query1 = q2s.get_sparql(result[1].encode('utf-8'))
    my_query2=q2s.get_sparql(result[4].encode('utf-8'))
    my_query3=q2s.get_sparql(result[7].encode('utf-8'))
    print(my_query1)
    if my_query1 is not None:
        result1 = fuseki.get_sparql_result(my_query1)
        value1 = fuseki.get_sparql_result_value(result1)
        if len(value1) == 6:
            fuzz1[0] = value1[0]
            fuzz1[1] = value1[1]
            fuzz1[2] = value1[2]
            fuzz1[3] = value1[3]
            fuzz1[4] = value1[4]
            fuzz1[5] = value1[5]
        if len(value1) == 5:
            fuzz1[0] = value1[0]
            fuzz1[1] = value1[1]
            fuzz1[2] = value1[2]
            fuzz1[3] = value1[3]
            fuzz1[4] = value1[4]
        if len(value1) == 4:
            fuzz1[0] = value1[0]
            fuzz1[1] = value1[1]
            fuzz1[2] = value1[2]
            fuzz1[3] = value1[3]


    else:
        # TODO 自然语言问题无法匹配到已有的正则模板上，回答“无法理解”
        print('I can\'t understand. :(')
    if my_query2 is not None:
        result2 = fuseki.get_sparql_result(my_query2)
        value2 = fuseki.get_sparql_result_value(result2)
        if len(value2) == 6:
            fuzz2[0] = value2[0]
            fuzz2[1] = value2[1]
            fuzz2[2] = value2[2]
            fuzz2[3] = value2[3]
            fuzz2[4] = value2[4]
            fuzz2[5] = value2[5]
        if len(value2) == 5:
            fuzz2[0] = value2[0]
            fuzz2[1] = value2[1]
            fuzz2[2] = value2[2]
            fuzz2[3] = value2[3]
            fuzz2[4] = value2[4]
        if len(value2) == 4:
            fuzz2[0] = value2[0]
            fuzz2[1] = value2[1]
            fuzz2[2] = value2[2]
            fuzz2[3] = value2[3]

    else:
        # TODO 自然语言问题无法匹配到已有的正则模板上，回答“无法理解”
        print('I can\'t understand. :(')
    if my_query3 is not None:
        result3 = fuseki.get_sparql_result(my_query3)
        value3 = fuseki.get_sparql_result_value(result3)
        if len(value3) == 6:
            fuzz3[0] = value3[0]
            fuzz3[1] = value3[1]
            fuzz3[2] = value3[2]
            fuzz3[3] = value3[3]
            fuzz3[4] = value3[4]
            fuzz3[5] = value3[5]
        if len(value3) == 5:
            fuzz3[0] = value3[0]
            fuzz3[1] = value3[1]
            fuzz3[2] = value3[2]
            fuzz3[3] = value3[3]
            fuzz3[4] = value3[4]
        if len(value3) == 4:
            fuzz3[0] = value3[0]
            fuzz3[1] = value3[1]
            fuzz3[2] = value3[2]
            fuzz3[3] = value3[3]
    else:
        # TODO 自然语言问题无法匹配到已有的正则模板上，回答“无法理解”
        print('I can\'t understand. :(')

    return render_template("fuzz_query.html",a=feature1,b=feature2,c=feature3,d=feature4,e=feature5,f=feature6)

@app.route("/precise_query",methods=['GET', 'POST'])
def precise():
    return render_template("precise_query.html")


#精确查询
@app.route("/submit_precise",methods=['GET', 'POST'])
def submit_precise():
    global y1,y2,y3,y4,y5,y6
    y1,y2,y3,y4,y5,y6=None,None,None,None,None,None
    y1 = request.form.get("fault_features1")
    y2 = request.form.get("fault_features2")
    y3 = request.form.get("fault_features3")
    y4 = request.form.get("fault_features4")
    y5 = request.form.get("fault_features5")
    y6 = request.form.get("fault_features6")
    print("y5",y5)
    print("y6",y6)
    if y6 is not None:
        features=y1+y2+y3+y4+y5+y6
    elif y5 is not None:
        features=y1+y2+y3+y4+y5
    else:
        features=y1+y2+y3+y4


    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 passwd='zhangyixin',
                                 db='faults2',
                                 port=3306,
                                 charset='utf8'
                                 )
    cur = connection.cursor()  # 游标（指针）cursor的方式操作数据
    sql1 = 'SELECT COUNT(feature1) FROM features WHERE feature1="%s"' % y1
    sql2 = 'SELECT COUNT(feature2) FROM features WHERE feature2="%s"' % y2
    sql3 = 'SELECT COUNT(feature3) FROM features WHERE feature3="%s"' % y3
    sql4 = 'SELECT COUNT(feature4) FROM features WHERE feature4="%s"' % y4
    sql5 = 'SELECT COUNT(feature5) FROM features WHERE feature5="%s"' % y5
    sql6 = 'SELECT COUNT(feature6) FROM features WHERE feature6="%s"' % y6

    cur.execute(sql1)  # execute(query, args):执行单条sql语句。
    see1 = cur.fetchone()  # 使结果全部可看
    cur.execute(sql2)  # execute(query, args):执行单条sql语句。
    see2 = cur.fetchone()  # 使结果全部可看
    cur.execute(sql3)  # execute(query, args):执行单条sql语句。
    see3 = cur.fetchone()  # 使结果全部可看
    cur.execute(sql4)  # execute(query, args):执行单条sql语句。
    see4= cur.fetchone()  # 使结果全部可看
    cur.execute(sql5)  # execute(query, args):执行单条sql语句。
    see5 = cur.fetchone()  # 使结果全部可看
    cur.execute(sql6)  # execute(query, args):执行单条sql语句。
    see6 = cur.fetchone()  # 使结果全部可看
    global num1,num2,num3,num4,num5,num6
    data = []
    data.append(see1)
    data.append(see2)
    data.append(see3)
    data.append(see4)
    data.append(see5)
    data.append(see6)
    print(data)
    num1=data[0][0]
    num2=data[1][0]
    num3 = data[2][0]
    num4 = data[3][0]
    num5 = data[4][0]
    num6 = data[5][0]

    connection.commit()
    connection.close()


    # TODO 连接Fuseki服务器。
    fuseki = JenaFuseki()
    # TODO 初始化自然语言到SPARQL查询的模块，参数是外部词典列表。
    q2s = Question2Sparql(
        ['./KB_query/external_dict/f1.txt', './KB_query/external_dict/f2.txt', './KB_query/external_dict/f3.txt',
         './KB_query/external_dict/f4.txt',
         './KB_query/external_dict/f5.txt'])

    my_query = q2s.get_sparql(features.encode('utf-8'))
    print(my_query)
    global solution,s1,s2,s3,s4,s5,s6

    if my_query is not None:
        result = fuseki.get_sparql_result(my_query)
        value = fuseki.get_sparql_result_value(result)
        print(value)

        # TODO 判断结果是否是布尔值，是布尔值则提问类型是"ASK"，回答“是”或者“不知道”。
        if isinstance(value, bool):
            if value is True:
                print('Yes')
            else:
                print('I don\'t know. :(')
        else:
            # TODO 查询结果为空，根据OWA，回答“不知道”
            if len(value) == 0:
                print('I don\'t know. :(')
            elif len(value) == 5:
                s1= value[0]
                s2=value[1]
                s3=value[2]
                s4=value[3]
                s5=value[4]

                print("s1:")
                print(s1)
                print("s2")
                print(s2)
            else:
                output = ''
                for v in value:
                    output += v + u'、'
                print(output[0:-1])

    else:
        # TODO 自然语言问题无法匹配到已有的正则模板上，回答“无法理解”
        print('I can\'t understand. :(')

    return render_template("precise_query.html",t1=y1,t2=y2,t3=y3,t4=y4,t5=y5,t6=y6)




#饼状图数据
@app.route("/barData")
def get_bar_chart():
    global num1,num2,num3,num4,num5,num6
    value = [num1,num2,num3,num4,num5,num6]
    print(value)
    name=["feature1","feature2","feature3","feature4","feature5","feature6"]
    print(name)
    return json.dumps({'name':name,'value':value})

#精确查询数据
@app.route("/linkData",methods=['GET', 'POST'])
def get_linkData():
    global y1,y2,y3,y4,y5,y6
    global s1,s2,s3,s4,s5,s6
    links =[
        {"source": '案例', "target": '解决方案'},
        {"source": '案例', "target": '特征'},
        {"source": '特征', "target":y1},
        {"source": '特征', "target":y2},
        {"source": '特征', "target":y3},
        {"source": '特征', "target":y4},
        {"source": '解决方案', "target": s1,'rela':'步骤1'},
        {"source": '解决方案', "target": s2,'rela':'步骤2'},
        {"source": '解决方案', "target": s3,'rela':'步骤3'},
        {"source": '解决方案', "target": s4,'rela':'步骤4'},
    ]
    print("y5:",y5)
    if y5!='':
        links.append({"source": '特征', "target":y5},)
    if s5!='':
        links.append({"source": '解决方案', "target": s5,'rela':'步骤5'},)
    if y6!='':
        links.append({"source": '特征', "target":y6},)
    if s6 !="":
        links.append({"source": '解决方案', "target": s6,'rela':'步骤6'},)

    return json.dumps({'name':links})

#模糊查询三个案例数据
@app.route("/fuzzData",methods=['GET', 'POST'])
def get_fuzzData():
    global y1,y2,y3,y4,y5,y6
    global fuzz1,fuzz2,fuzz3
    global id1,id2,id3,f1,f2,f3,f4,f5,f6
    fuzz_features=""
    print(f1)
    print(f2)
    if f1 !='':
        fuzz_features+=f1+"&"
    if f2 !='':
        fuzz_features += f2 + "&"
    if f3 !='':
        fuzz_features += f3 + "&"
    if f4 !='':
        fuzz_features += f4 + "&"
    if f5 !='':
        fuzz_features += f5 + "&"
    if f6 !='':
        fuzz_features += f5 + "&"
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 passwd='zhangyixin',
                                 db='faults2',
                                 port=3306,
                                 charset='utf8'
                                 )
    cur = connection.cursor()  # 游标（指针）cursor的方式操作数据
    sql1 = 'SELECT feature1,feature2,feature3,feature4,feature5,feature6 FROM features WHERE id="%s"' % id1
    sql2 = 'SELECT feature1,feature2,feature3,feature4,feature5,feature6 FROM features WHERE id="%s"' % id2
    sql3 = 'SELECT feature1,feature2,feature3,feature4,feature5,feature6 FROM features WHERE id="%s"' % id3
    cur.execute(sql1)
    see1 = cur.fetchone()
    cur.execute(sql2)
    see2 = cur.fetchone()
    cur.execute(sql3)
    see3 = cur.fetchone()
    connection.commit()
    connection.close()
    links = [
        {"source": '特征:'+fuzz_features, "target": '案例' + str(id1)},
        {"source": '特征:'+fuzz_features, "target": '案例' + str(id2)},
        {'source': '特征:'+fuzz_features, 'target': '案例' + str(id3)},
        {"source": '案例' + str(id1), "target": str(id1) + '特征'},
        {"source": '案例' + str(id1), "target": str(id1) + '故障方案'},
        {"source": '案例' + str(id2), "target": str(id2) + '特征'},
        {"source": '案例' + str(id2), "target": str(id2) + '故障方案'},
        {"source": '案例' + str(id3), "target": str(id3) + '特征'},
        {"source": '案例' + str(id3), "target": str(id3) + '故障方案'},
        ]
    for i in range(5):
        links.append({"source": str(id1)+'特征', "target": see1[i]},)
        links.append({"source": str(id1) + '故障方案', "target": fuzz1[i],'rela':'步骤'+str(i+1)},)
        links.append({"source": str(id2)+'特征', "target": see2[i]},)
        links.append({"source": str(id2) + '故障方案', "target": fuzz2[i],'rela':'步骤'+str(i+1)},)
        links.append({"source": str(id3) + '特征', "target": see3[i]},)
        links.append({"source": str(id3) + '故障方案', "target": fuzz3[i],'rela':'步骤'+str(i+1)},)
    return json.dumps({'name': links})

#整体知识地图
@app.route("/allData",methods=['GET', 'POST'])
def get_allData():
    temp1=0
    temp2=0
    count = Feature.query.filter(Feature.id).count()
    print(count)
    all_feature = Feature.query.all()
    all_sol = Sol_feature.query.all()
    links = [
    ]
    for i in range(count):
        links.append({"source": '案例' + str(i), "target": str(i) + '特征'}, )
        links.append({"source": '案例' + str(i), "target": str(i) + '故障方案'}, )
    for all in all_feature:
        links.append({"source": str(temp1) + '特征', "target": all.__dict__["feature1"],'rela':'特征1'}, )
        links.append({"source": str(temp1) + '特征', "target": all.__dict__["feature2"],'rela':'特征2'}, )
        links.append({"source": str(temp1) + '特征', "target": all.__dict__["feature3"],'rela':'特征3'}, )
        links.append({"source": str(temp1) + '特征', "target": all.__dict__["feature4"],'rela':'特征4'}, )
        if all.__dict__["feature5"] is not None:
            links.append({"source": str(temp1) + '特征', "target": all.__dict__["feature5"],'rela':'特征5'}, )
        if all.__dict__["feature6"] is not None:
            links.append({"source": str(temp1) + '特征', "target": all.__dict__["feature6"],'rela':'特征6'}, )
        temp1=temp1+1
    for alls in all_sol:
        links.append({"source": str(temp2) + '故障方案', "target": alls.__dict__["sol_feature1"],'rela':'步骤1'}, )
        links.append({"source": str(temp2) + '故障方案', "target": alls.__dict__["sol_feature2"],'rela':'步骤2'}, )
        links.append({"source": str(temp2) + '故障方案', "target": alls.__dict__["sol_feature3"],'rela':'步骤3'}, )
        links.append({"source": str(temp2) + '故障方案', "target": alls.__dict__["sol_feature4"],'rela':'步骤4'}, )
        if alls.__dict__["sol_feature5"] is not None:
            links.append({"source": str(temp2) + '特征', "target": alls.__dict__["sol_feature5"],'rela':'步骤5'}, )
        if alls.__dict__["sol_feature6"] is not None:
            links.append({"source": str(temp2) + '特征', "target": alls.__dict__["sol_feature6"],'rela':'步骤6'}, )
        temp2=temp2+1
    return json.dumps({'name': links})




if __name__ == '__main__':
    # 会删除所有继承db.Model的表
    db.drop_all()
    # 会创建所有继承自db.Model的表
    db.create_all()

    # 生成数据
    fea1 = Feature(feature1='F电源',feature2='HL单元功能',feature3='圆筒',feature4='不在位',feature5='飞灯不亮',feature6='F计算机')
    fea2 = Feature(feature1='F组合',feature2='车辆功能检查',feature3='电B通路',feature4='控制故障',feature5='圆筒B组激活',feature6='圆筒舵机激活')
    fea3 = Feature(feature1='F计算机',feature2='Y协同上课',feature3='F解锁故障',feature4='解锁故障',feature5='程序执行过程',feature6='竖起不到位')
    fea4 = Feature(feature1='手控台', feature2='程控', feature3='电机', feature4='不能启动', feature5='手动操作',feature6='车辆简程检查')
    fea5 = Feature(feature1='手控台', feature2='车辆功能检查', feature3='电机', feature4='不能启动', feature5='程控操作')
    fea6 = Feature(feature1='手控台', feature2='车辆功能检查', feature3='液压系统', feature4='无法建压', feature5='手动状态')
    fea7 = Feature(feature1='吊机', feature2='吊装操作', feature3='红色报警灯', feature4='故障', feature5='吊装过程')
    fea8 = Feature(feature1='吊具', feature2='吊装操作', feature3='刚柔转换钢丝绳', feature4='不同步', feature5='吊装过程')
    fea9 = Feature(feature1='数传系统', feature2='HL单元功能', feature3='单路', feature4='有线', feature5='不通')
    fea10 = Feature(feature1='停放架锁紧机构', feature2='XL操作', feature3='圆筒', feature4='无法锁紧', feature5='吊卸过程')


    f2s1 = Fea2Sol(fea_id='1',sol_id='1')
    f2s2 = Fea2Sol(fea_id='2', sol_id='2')
    f2s3 = Fea2Sol(fea_id='3', sol_id='3')
    f2s4 = Fea2Sol(fea_id='4', sol_id='4')
    f2s5 = Fea2Sol(fea_id='5', sol_id='5')
    f2s6 = Fea2Sol(fea_id='6', sol_id='6')
    f2s7 = Fea2Sol(fea_id='7', sol_id='7')
    f2s8 = Fea2Sol(fea_id='8', sol_id='8')
    f2s9 = Fea2Sol(fea_id='9', sol_id='9')
    f2s10 = Fea2Sol(fea_id='10', sol_id='10')
    # 生成数据
    sol_fea1 = Sol_feature(sol_feature1='供电',sol_feature2='开关位置',sol_feature3='F电源',sol_feature4='F电源保险管',sol_feature5='F电源控制模块',sol_feature6='F电源模块')
    sol_fea2 = Sol_feature(sol_feature1='更换',sol_feature2='F组合',sol_feature3='相应圆筒控制板',sol_feature4='相应圆筒IO插板',sol_feature5='相应圆筒模拟板')
    sol_fea3 = Sol_feature(sol_feature1='检查',sol_feature2='控制机柜',sol_feature3='输出信号',sol_feature4='1X11插头6、7两点',sol_feature5='5#424插板')
    sol_fea4 = Sol_feature(sol_feature1='检查功能', sol_feature2='“电机启动”开关', sol_feature3='控制箱继电器', sol_feature4='W128电缆', sol_feature5='电源箱')
    sol_fea5 = Sol_feature(sol_feature1='开关量输入板III', sol_feature2='开关量输出板', sol_feature3='“电机启动”开关', sol_feature4='控制箱继电器', sol_feature5='W128电缆', sol_feature6='电源箱')
    sol_fea6 = Sol_feature(sol_feature1='检查', sol_feature2='比例溢流阀YV1电缆插头', sol_feature3='“调压”电位器', sol_feature4='车辆转接盒', sol_feature5='比例溢流阀YV1')
    sol_fea7 = Sol_feature(sol_feature1='操作吊机', sol_feature2='减小起重力矩', sol_feature3='报警指示灯', sol_feature4='OLP按钮', sol_feature5='厂家解决')
    sol_fea8 = Sol_feature(sol_feature1='检查', sol_feature2='钢丝绳', sol_feature3='调整螺母位置', sol_feature4='刚柔转换油缸', sol_feature5='维修')
    sol_fea9 = Sol_feature(sol_feature1='检查', sol_feature2='参数设置', sol_feature3='电缆连接', sol_feature4='更换接口', sol_feature5='更换从站有线数传单元')
    sol_fea10 = Sol_feature(sol_feature1='检查', sol_feature2='更换', sol_feature3='手柄', sol_feature4='锁止机构销轴', sol_feature5='运动正常')
    # 把数据提交给用户会话
    db.session.add_all([fea1, fea2, fea3,fea4, fea5, fea6,fea7, fea8, fea9,fea10])
    db.session.add_all([f2s1, f2s2, f2s3, f2s4, f2s5, f2s6, f2s7, f2s8, f2s9, f2s10])
    db.session.add_all([sol_fea1, sol_fea2, sol_fea3, sol_fea4, sol_fea5, sol_fea6, sol_fea7, sol_fea8, sol_fea9, sol_fea10])
    # 提交会话
    db.session.commit()
    app.run(debug=True)

"""