from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pymysql

app = Flask(__name__)
app.secret_key = 'test'


# 设置数据库的连接地址
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:zhangyixin@127.0.0.1:3306/faults_last"
# 是否监听数据库变化  一般不打开, 比较消耗性能
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# 创建数据库操作对象(建立了数据库连接)
db = SQLAlchemy(app)

# 故障特征表
class Feature(db.Model):
    __tablename__ = "features"
    features_id = db.Column(db.Integer, primary_key=True)
    features = db.Column(db.String(64))
    group_id=db.Column(db.Integer)
    weight=db.Column(db.Float)

# 关系表
class Fea2Sol(db.Model):
    __tablename__ = "fea2sol"
    features_id = db.Column(db.Integer, db.ForeignKey("features.features_id"), primary_key=True)
    sol_features_id = db.Column(db.Integer, db.ForeignKey("sol_features.sol_features_id"),primary_key=True)

# 解决方案特征表
class Sol_feature(db.Model):
    __tablename__ = "sol_features"
    sol_features_id = db.Column(db.Integer, primary_key=True)
    sol_features = db.Column(db.String(500))
    group_id = db.Column(db.Integer)

class Users(db.Model):
    __tablename__="users"
    userid=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(128))
    password=db.Column(db.String(128))

if __name__ == '__main__':
    # 会删除所有继承db.Model的表
    db.drop_all()
    # 会创建所有继承自db.Model的表
    db.create_all()
    user1 = Users(userid='1', username='111',password='111')
    user2 = Users(userid='2', username='222',password='222')

    # 生成数据
    fea_arr=['5℃','58%','976hpa','400m','1ppm','10℃','装填车','3号车','吊装操作','吊具','执行动作迟缓',
             '10℃', '40%', '1076hpa', '500m', '0.8ppm', '15℃', '装填车', '4号车','吊装操作', '吊具', '脉动现象',
             '15℃', '35%', '1276hpa', '560m', '0.7ppm', '9℃', '装填车', '5号车','吊装操作', '吊具', '噪声过大',
             '25℃', '40%', '1076hpa', '500m', '0.9ppm', '5℃', '装填车', '6号车','吊装操作', '吊具', '噪声过大',
             '10℃', '35%', '1276hpa', '500m', '0.8ppm', '15℃', '装填车', '3号车','吊装操作', '吊具', '无反应',

             '10℃', '58%', '976hpa', '400m', '0.6ppm', '10℃', '装填车', '4号车','吊装操作',  '吊具', '无反应',
             '10℃', '40%', '1076hpa', '500m', '0.8ppm', '15℃', '装填车', '5号车','吊装操作', '吊具', '不能换向',
             '15℃', '35%', '1276hpa', '560m', '0.7ppm', '9℃', '装填车', '6号车','吊装操作', '吊具', '不能换向',
             '25℃', '40%', '1076hpa', '500m', '0.9ppm', '5℃', '装填车', '2号车','吊装操作', '吊具', '不能换向',
             '10℃', '35%', '1276hpa', '500m', '0.8ppm', '15℃', '装填车', '4号车','吊装操作', '吊具', '换向','运动迟缓',

             '5℃', '58%', '976hpa', '400m', '1ppm', '10℃', '装填车', '5号车', '吊装操作', '吊具', '换向', '运动迟缓',
             '10℃', '40%', '1076hpa', '500m', '0.8ppm', '15℃', '装填车', '6号车', '吊具','换向', '换向不灵',
             '15℃', '35%', '1276hpa', '560m', '0.7ppm', '9℃', '装填车', '7号车', '吊装操作', '吊具','换向', '换向不灵',
             '25℃', '40%', '1076hpa', '500m', '0.9ppm', '5℃', '装填车', '8号车', '吊装操作', '吊具','换向', '换向不灵',
             '10℃', '35%', '1276hpa', '500m', '0.8ppm', '15℃', '装填车', '9号车', '吊装操作', '吊具','换向', '冲击','噪声',

             '10℃', '58%', '976hpa', '400m', '0.6ppm', '10℃', '装填车', '10号车', '吊装操作', '吊具','换向', '冲击','噪声',
             '10℃', '40%', '1076hpa', '500m', '0.8ppm', '15℃', '装填车', '5号车', '吊装操作',  '液压压力表', '指针','波动大',
             '15℃', '35%', '1276hpa', '560m', '0.7ppm', '9℃', '装填车', '6号车', '吊装操作', '液压压力表', '指针','波动大',
             '25℃', '40%', '1076hpa', '500m', '0.9ppm', '5℃', '装填车', '7号车', '吊装操作', '液压压力表', '指针','波动大',
             '10℃', '35%', '1276hpa', '500m', '0.8ppm', '15℃', '装填车', '9号车', '吊装操作', '推力不足', '速度下降', '工作不稳定',

             '10℃', '35%', '1276hpa', '500m', '0.8ppm', '15℃', '装填车', '8号车', '吊装操作', '推力不足', '速度下降', '工作不稳定',


             ]

    fea=['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','',
         '','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '','', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',]
    for i in range(0,11):
        fea[i]=Feature(features=fea_arr[i],group_id='1',weight=0)
        db.session.add(fea[i])
    for i in range(11,22):
        fea[i]=Feature(features=fea_arr[i],group_id='2',weight=0)
        db.session.add(fea[i])
    for i in range(22,33):
        fea[i]=Feature(features=fea_arr[i],group_id='3',weight=0)
        db.session.add(fea[i])
    for i in range(33,44):
        fea[i]=Feature(features=fea_arr[i],group_id='4',weight=0)
        db.session.add(fea[i])
    for i in range(44,55):
        fea[i]=Feature(features=fea_arr[i],group_id='5',weight=0)
        db.session.add(fea[i])
    for i in range(55,66):
        fea[i]=Feature(features=fea_arr[i],group_id='6',weight=0)
        db.session.add(fea[i])
    for i in range(66,77):
        fea[i]=Feature(features=fea_arr[i],group_id='7',weight=0)
        db.session.add(fea[i])
    for i in range(77,88):
        fea[i]=Feature(features=fea_arr[i],group_id='8',weight=0)
        db.session.add(fea[i])
    for i in range(88,99):
        fea[i]=Feature(features=fea_arr[i],group_id='9',weight=0)
        db.session.add(fea[i])
    for i in range(99,111):
        fea[i]=Feature(features=fea_arr[i],group_id='10',weight=0)
        db.session.add(fea[i])
    for i in range(111,123):
        fea[i]=Feature(features=fea_arr[i],group_id='11',weight=0)
        db.session.add(fea[i])
    for i in range(123,134):
        fea[i]=Feature(features=fea_arr[i],group_id='12',weight=0)
        db.session.add(fea[i])
    for i in range(134,146):
        fea[i]=Feature(features=fea_arr[i],group_id='13',weight=0)
        db.session.add(fea[i])
    for i in range(146,158):
        fea[i]=Feature(features=fea_arr[i],group_id='14',weight=0)
        db.session.add(fea[i])
    for i in range(158,171):
        fea[i]=Feature(features=fea_arr[i],group_id='15',weight=0)
        db.session.add(fea[i])
    for i in range(171,184):
        fea[i]=Feature(features=fea_arr[i],group_id='16',weight=0)
        db.session.add(fea[i])
    for i in range(184,196):
        fea[i]=Feature(features=fea_arr[i],group_id='17',weight=0)
        db.session.add(fea[i])
    for i in range(196,208):
        fea[i]=Feature(features=fea_arr[i],group_id='18',weight=0)
        db.session.add(fea[i])
    for i in range(208,220):
        fea[i]=Feature(features=fea_arr[i],group_id='19',weight=0)
        db.session.add(fea[i])
    for i in range(220,232):
        fea[i]=Feature(features=fea_arr[i],group_id='20',weight=0)
        db.session.add(fea[i])
    for i in range(232,244):
        fea[i]=Feature(features=fea_arr[i],group_id='21',weight=0)
        db.session.add(fea[i])


    sol_arr = ['首先检查液压油量，油量符合规定，排除油箱液压油不足问题；然后检查有关连接接头，接头均连接良好，排除密封不严空气进入的问题；最后检查柱塞泵，发现柱塞与缸体之间存在间隙磨损，导致接触面接触不好，且柱塞回程不够，引起缸体与配油盘间失去密封，更换柱塞泵之后故障排除。',
               '首先检查相关的连接接头，未发现接头连接问题；然后检查柱塞泵缸体与配油盘、柱塞与缸孔之间不存在磨损导致的内泄漏问题；最后检查发现操作变量机构不协调，只是操作脉动发生，更换变量机构之后故障排除。',
               '首先检查液压油量是否符合规定，发现液压油量符合规定，排除因油液液面过低，液压泵吸空导致噪声的问题；然后对滤油器进行检查未发现堵塞情况，之后对系统进行排气，故障未排除；最后检查发现柱塞泵安装轴承单边磨损，更换轴承之后故障排除。',
               '首先检查液压油量是否符合规定，发现液压油量符合规定，排除因油液液面过低，液压泵吸空导致噪声的问题；然后对滤油器进行检查未发现堵塞情况，之后对系统进行排气，故障未排除；检查柱塞泵安装轴承未发现磨损问题；最后检查柱塞泵发现柱塞与滑靴球头连接严重松动，更换柱塞泵之后故障排除。',
               '首先检查装填车供电正常，液压油量符合规定；然后检查各连接机构均无异常；最后检查柱塞泵，发现柱塞与缸孔卡死，原因是液压油污染严重，液压油高温粘结所致，最终导致泵轴不能转动，操作无反应，更换液压油后故障排除。',
               '首先检查装填车供电正常，液压油量符合规定；然后检查各连接机构均无异常；最后检查柱塞泵，发现柱塞球头折断，导致泵轴不能转动，应更换柱塞泵后故障排除。',
               '首先检查各连接机构均无异常；然后检查相对应方向单向阀，未发现异常；最后检查换向阀，发现换向阀中电磁铁因滑阀卡住，铁心吸不到底而烧毁，致使换向阀不能换向，吊具不能正常操作，更换电磁铁后故障排除。',
               '首先检查各连接机构均无异常；然后检查相对应方向单向阀，未发现异常；最后检查换向阀，发现液动换向阀上的节流阀堵塞，致使换向阀不能换向，更换节流阀后故障排除。',
               '首先检查各连接机构均无异常；然后检查相对应方向单向阀，未发现异常；之后检查换向阀、节流阀，均无异常；最后检查液压回油路上定量阀，发现定量阀关闭，待更换定量阀或等待一定时间定量阀中复位弹簧自动复位，故障排除。',
               '首先检查液压油量符合规定，滤油器无堵塞情况；然后检查各连接机构均无异常；之后检查相对应方向单向阀，未发现异常；最后检查换向阀，发现换向阀中换向推杆长期撞击磨损而变短，阀芯行程不足，开孔及流量变小，更换推杆后故障排除。',
               '首先检查液压油量符合规定，滤油器无堵塞情况；然后检查各连接机构均无异常；之后检查相对应方向单向阀，未发现异常；最后检查换向阀，发现换向阀中衔铁接触点磨损，阀芯行程不足，开孔及流量变小，更换电磁铁后故障排除。',
               '首先检查液压油量符合规定，液压油无污染，滤油器无堵塞情况，排除因油液导致卡住滑阀的问题；然后检查各连接机构均无异常；之后检查换向阀，发现换向阀中电磁铁铁心接触部位有污染物，致使接触不良，清除铁心接触部位污染物后故障排除。',
               '首先检查液压油量符合规定，液压油无污染，滤油器无堵塞情况，排除因油液导致卡住滑阀的问题；然后检查各连接机构均无异常；之后检查换向阀中电磁铁未发现异常；最后检查发现换向阀中滑阀与阀体因磨损间隙过大，更换换向阀后故障排除。',
               '首先检查液压油量符合规定，液压油无污染，滤油器无堵塞情况，排除因油液导致卡住滑阀的问题；然后检查各连接机构均无异常；之后检查换向阀中电磁铁未发现异常；最后检查发现电磁换向阀的推杆磨损后长度不足，阀芯移动过小，引起换向不灵，更换换向阀中推杆后故障排除。',
               '首先检查各连接机构均无异常；然后检查换向回路中单向阀，未发现异常；之后检查换向阀中滑阀未发现异常；最后检查发现换向阀中固定电磁铁的螺栓松动而产生振动，紧固螺栓并加防松垫圈后故障排除。',
               '首先检查各连接机构均无异常；然后检查换向阀未发现异常；之后检查换向回路中单向阀，未发现异常；最后检查发现换向回路中单向节流阀阀芯与孔配合间隙过大，单向阀弹簧弹力过小，阻力失效，产生冲击，更换单向阀弹簧后故障排除。',
               '首先拆装检查压力表，未发现异常；然后检查液压系统，未发现油液泄露问题；之后液压系统排气，故障现象未消失；最后检查溢流阀，发现调压的控制阀芯弹簧弯曲，不能维持稳定的压力，更换溢流阀弹簧后故障排除。',
               '首先拆装检查压力表，未发现异常；然后检查液压系统，未发现油液泄露问题；之后液压系统排气，故障现象未消失；最后检查溢流阀，发现锥阀与阀座配合不良，内泄忽大忽小，导致压力时高时低，更换溢流阀后故障排除。',
               '首先拆装检查压力表，未发现异常；然后检查液压系统，未发现油液泄露问题；之后液压系统排气，故障现象未消失；最后进行液压油化验，发现油液精度不符合规定，进一步检查发现，因油液污染导致溢流阀主阀上的阻尼时大时小，造成压力波动，更换液压油，并清洗溢流阀后故障排除。',
               '首先检查液压油量、液压油精度，均符合规定；然后检查液压系统，未发现油液外泄露问题；之后对液压系统排气，故障现象未消失；最后检查发现液压缸工作段磨损不均匀，由于局部密封不好而内泄漏严重，更换液压油缸后故障排除。',
               '首先检查液压油量、液压油精度，均符合规定；然后检查液压系统，未发现油液外泄露问题；之后对液压系统排气，故障现象未消失；最后检查发现液压活塞上的密封圈失去密封作用内泄严重，更换密封圈后故障排除。',
                '','','',
    ]
    sol=['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','',
         '','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    for j in range(0,1):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='1')
        db.session.add(sol[j])
    for j in range(1,2):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='2')
        db.session.add(sol[j])
    for j in range(2,3):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='3')
        db.session.add(sol[j])
    for j in range(3,4):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='4')
        db.session.add(sol[j])
    for j in range(4,5):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='5')
        db.session.add(sol[j])
    for j in range(5,6):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='6')
        db.session.add(sol[j])
    for j in range(6,7):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='7')
        db.session.add(sol[j])
    for j in range(7,8):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='8')
        db.session.add(sol[j])
    for j in range(8,9):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='9')
        db.session.add(sol[j])
    for j in range(9,10):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='10')
        db.session.add(sol[j])
    for j in range(10,11):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='11')
        db.session.add(sol[j])
    for j in range(11,12):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='12')
        db.session.add(sol[j])
    for j in range(12,13):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='13')
        db.session.add(sol[j])
    for j in range(13,14):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='14')
        db.session.add(sol[j])
    for j in range(14,15):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='15')
        db.session.add(sol[j])
    for j in range(15,16):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='16')
        db.session.add(sol[j])
    for j in range(16,17):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='17')
        db.session.add(sol[j])
    for j in range(17,18):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='18')
        db.session.add(sol[j])
    for j in range(18,19):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='19')
        db.session.add(sol[j])
    for j in range(19,20):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='20')
        db.session.add(sol[j])
    for j in range(20,21):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='21')
        db.session.add(sol[j])


    # 提交会话
    db.session.commit()

    db = pymysql.connect(host='localhost', user='root', password='zhangyixin', port=3306, db='faults_last')

    for i in range(1, 12):
        for j in range(1, 2):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(12, 23):
        for j in range(2, 3):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(23, 34):
        for j in range(3, 4):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(34, 45):
        for j in range(4, 5):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(45, 56):
        for j in range(5, 6):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(56, 67):
        for j in range(6, 7):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(67, 78):
        for j in range(7, 8):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(78, 89):
        for j in range(8, 9):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(89, 100):
        for j in range(9, 10):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(100, 112):
        for j in range(10, 11):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(112, 124):
        for j in range(11, 12):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(124, 135):
        for j in range(12, 13):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(135, 147):
        for j in range(13, 14):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(147, 159):
        for j in range(14, 15):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(159, 172):
        for j in range(15, 16):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(172, 185):
        for j in range(16, 17):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(185, 197):
        for j in range(17, 18):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(197, 209):
        for j in range(18, 19):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(209, 221):
        for j in range(19, 20):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(221, 233):
        for j in range(20, 21):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #21
    for i in range(233, 245):
        for j in range(21, 22):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()





    db.commit()
    db.close()


