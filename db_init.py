from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pymysql

app = Flask(__name__)
app.secret_key = 'test'


# 设置数据库的连接地址
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:zhangyixin@127.0.0.1:3306/faults_test"
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

# 关系表
class Fea2Sol(db.Model):
    __tablename__ = "fea2sol"
    features_id = db.Column(db.Integer, db.ForeignKey("features.features_id"), primary_key=True)
    sol_features_id = db.Column(db.Integer, db.ForeignKey("sol_features.sol_features_id"),primary_key=True)

# 解决方案特征表
class Sol_feature(db.Model):
    __tablename__ = "sol_features"
    sol_features_id = db.Column(db.Integer, primary_key=True)
    sol_features = db.Column(db.String(64))
    group_id = db.Column(db.Integer)



if __name__ == '__main__':
    # 会删除所有继承db.Model的表
    db.drop_all()
    # 会创建所有继承自db.Model的表
    db.create_all()

    # 生成数据
    fea_arr=['F电源','HL单元功能','圆筒','不在位','飞灯不亮','F计算机','F组合','车辆功能检查','电B通路','控制故障','圆筒B组激活','圆筒舵机激活',
             'F计算机','Y协同上课', 'F解锁故障', '解锁故障','程序执行过程','竖起不到位','手控台','程控','电机','不能启动', '手动操作','车辆简程检查',
             '手控台', '车辆功能检查', '电机', '不能启动', '程控操作','手控台','车辆功能检查','液压系统','无法建压','手动状态',
             '吊机', '吊装操作','红色报警灯', '故障', '吊装过程','吊具', '吊装操作','刚柔转换钢丝绳','不同步','吊装过程',
             '数传系统',  'HL单元功能','单路', '有线','不通','停放架锁紧机构','XL操作','圆筒','无法锁紧','吊卸过程',
             '节流阀','全开位置','全闭位置','卡死','节流阀','节流口','污染物','堵塞','节流阀','单向阀','密封不良','弹簧变形',
             '节流阀','阀芯','阀体','间隙过大','节流阀','油液','污染','时堵时通','节流阀','油温和粘度','变化','导致','流量变化',
             '节流阀','锁紧装置','松动','节流口','通流面积','变化','负载','突变','节流阀','丧失稳定','节流阀','内外泄漏','流量','不稳定',
             '单向阀','接合面','存在','尺寸误差','密封元件','间隙','过大',
             '密封元件','压力','过高','密封元件','沟槽尺寸','不合适','密封元件','安装','不到位',
             '密封元件','温度过高','老化','密封元件','低温硬化','开裂','密封元件','横向负载作用','扭曲','密封元件','表面','损伤','密封元件','装配','损伤',
             '密封元件','润滑不良','磨损','密封元件','压力过高','负载过大','密封元件','液压油','不相容','密封元件','溶剂','溶解','密封元件','液压油老化','膨胀',
             '滤油器','滤芯','变形','滤油器','滤芯','掉粒','网式滤油器','金属网与骨架','脱焊','蓄能器','供油','不均','蓄能器','无氮气','气压不足',
             '蓄能器','气阀','漏气','蓄能器','气囊','漏气','蓄能器','器盖','漏气','蓄能器','供油','压力低','蓄能器','充气压力','不足',
             '蓄能器','工作压力范围小','压力过高','蓄能器','容量小','供油量不足','蓄能器','活塞','气囊','运动阻力不均','冷却器','破裂','漏水','油中进水',
             '冷却器','冷却水量','风量','不足','冷却器','冷却水温','过高','轴向柱塞泵','吸油管','堵塞','轴向柱塞泵','滤油器','堵塞','轴向柱塞泵','吸油管','阻力太大','轴向柱塞泵','滤油器',
             '阻力太大','轴向柱塞泵','吸油路','管径','过小',
             '轴向柱塞泵','液压油箱','油面','太低','轴向柱塞泵','泵体内','有空气','轴向柱塞泵','密封不严','有空气','轴向柱塞泵','液压泵内','泄漏','轴向柱塞泵','柱塞','不能回程',
             '转向缸','压力表显示值','稍偏低','液压缸两端','爬行','有噪声','转向缸','液压缸','无力','油箱','起泡','不排气','转向缸','活塞杆表面','发白','有声响','转向缸','爬行部位','规律性强','活塞杆局部','发白','转向缸','液压泵','供油','不足',
             '换向阀','电磁铁','推不动','阀芯','换向阀','滑阀','卡住','铁心','烧毁','干式电磁铁','推杆处','密封圈','磨损','板式换向阀','结合面','渗油','换向阀','线圈','过热',
             '溢流阀','卸荷时','压力波','冲击声','溢流阀','回油管路中','背压','过大','溢流阀','流量','超过','允许值','溢流阀','弹簧','变形','产生噪声','溢流阀','锥阀','磨损',
             '换向阀','系统','压力低','换向阀','漏阀','卡住','全开位置','换向阀','弹簧','弯曲','压力','波动','换向阀','滑阀','不灵','换向阀','管接头','松动',
             '轴向柱塞泵','输出油液','压力','不足','轴向柱塞泵','内泄漏','过大','轴向柱塞泵','变量机构','不协调','脉动','轴向柱塞泵','泵内','有空气','轴向柱塞泵','管路','振动',
             '液压泵','吸气','严重','发热','液压泵','油液粘度','过高','变量机构','控制油路','堵塞','变量机构','控制油路','单项阀','卡断','液压泵','泵轴','柱塞','卡死','滑靴','拉脱',
             '轴向柱塞马达','输出转速','输出转矩','低','轴向柱塞马达','弹簧','疲劳','内泄漏','轴向柱塞马达','轴端','密封圈','损坏','外泄漏','轴向柱塞马达','连轴器','不同心','轴向柱塞马达','空气','进入','内部','异常声响',
             '转向缸','排气','不良','外泄漏','转向缸','活塞','密封圈','损坏','内泄漏','转向缸','滑动金属面','摩擦声响','液压缸','动力油压','太低','液压缸','运动部位','配合太紧','不能动作',


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
    for i in range(0,6):
        fea[i]=Feature(features=fea_arr[i],group_id='1')
        db.session.add(fea[i])
    for i in range(6,12):
        fea[i]=Feature(features=fea_arr[i],group_id='2')
        db.session.add(fea[i])
    for i in range(12,18):
        fea[i]=Feature(features=fea_arr[i],group_id='3')
        db.session.add(fea[i])
    for i in range(18,24):
        fea[i]=Feature(features=fea_arr[i],group_id='4')
        db.session.add(fea[i])
    for i in range(24,29):
        fea[i]=Feature(features=fea_arr[i],group_id='5')
        db.session.add(fea[i])
    for i in range(29,34):
        fea[i]=Feature(features=fea_arr[i],group_id='6')
        db.session.add(fea[i])
    for i in range(34,39):
        fea[i]=Feature(features=fea_arr[i],group_id='7')
        db.session.add(fea[i])
    for i in range(39,44):
        fea[i]=Feature(features=fea_arr[i],group_id='8')
        db.session.add(fea[i])
    for i in range(44,49):
        fea[i]=Feature(features=fea_arr[i],group_id='9')
        db.session.add(fea[i])
    for i in range(49,54):
        fea[i]=Feature(features=fea_arr[i],group_id='10')
        db.session.add(fea[i])
    for i in range(54,58):
        fea[i]=Feature(features=fea_arr[i],group_id='11')
        db.session.add(fea[i])
    for i in range(58,62):
        fea[i]=Feature(features=fea_arr[i],group_id='12')
        db.session.add(fea[i])
    for i in range(62,66):
        fea[i]=Feature(features=fea_arr[i],group_id='13')
        db.session.add(fea[i])
    for i in range(66,70):
        fea[i]=Feature(features=fea_arr[i],group_id='14')
        db.session.add(fea[i])
    for i in range(70,74):
        fea[i]=Feature(features=fea_arr[i],group_id='15')
        db.session.add(fea[i])
    for i in range(74,79):
        fea[i]=Feature(features=fea_arr[i],group_id='16')
        db.session.add(fea[i])
    for i in range(79,85):
        fea[i]=Feature(features=fea_arr[i],group_id='17')
        db.session.add(fea[i])
    for i in range(85,89):
        fea[i]=Feature(features=fea_arr[i],group_id='18')
        db.session.add(fea[i])
    for i in range(89,93):
        fea[i]=Feature(features=fea_arr[i],group_id='19')
        db.session.add(fea[i])
    for i in range(93,97):
        fea[i]=Feature(features=fea_arr[i],group_id='20')
        db.session.add(fea[i])
    for i in range(97,100):
        fea[i]=Feature(features=fea_arr[i],group_id='21')
        db.session.add(fea[i])
    for i in range(100,103):
        fea[i]=Feature(features=fea_arr[i],group_id='22')
        db.session.add(fea[i])
    for i in range(103,106):
        fea[i]=Feature(features=fea_arr[i],group_id='23')
        db.session.add(fea[i])
    for i in range(106,109):
        fea[i]=Feature(features=fea_arr[i],group_id='24')
        db.session.add(fea[i])
    for i in range(109,112):
        fea[i]=Feature(features=fea_arr[i],group_id='25')
        db.session.add(fea[i])
    for i in range(112,115):
        fea[i]=Feature(features=fea_arr[i],group_id='26')
        db.session.add(fea[i])
    for i in range(115,118):
        fea[i]=Feature(features=fea_arr[i],group_id='27')
        db.session.add(fea[i])
    for i in range(118,121):
        fea[i]=Feature(features=fea_arr[i],group_id='28')
        db.session.add(fea[i])
    for i in range(121,124):
        fea[i]=Feature(features=fea_arr[i],group_id='29')
        db.session.add(fea[i])
    for i in range(124,127):
        fea[i]=Feature(features=fea_arr[i],group_id='30')
        db.session.add(fea[i])
    for i in range(127,130):
        fea[i]=Feature(features=fea_arr[i],group_id='31')
        db.session.add(fea[i])
    for i in range(130,133):
        fea[i]=Feature(features=fea_arr[i],group_id='32')
        db.session.add(fea[i])
    for i in range(133,136):
        fea[i]=Feature(features=fea_arr[i],group_id='33')
        db.session.add(fea[i])
    for i in range(136,139):
        fea[i]=Feature(features=fea_arr[i],group_id='34')
        db.session.add(fea[i])
    #35
    for i in range(139,142):
        fea[i]=Feature(features=fea_arr[i],group_id='35')
        db.session.add(fea[i])
    for i in range(142,145):
        fea[i]=Feature(features=fea_arr[i],group_id='36')
        db.session.add(fea[i])
    for i in range(145,148):
        fea[i]=Feature(features=fea_arr[i],group_id='37')
        db.session.add(fea[i])
    for i in range(148,151):
        fea[i]=Feature(features=fea_arr[i],group_id='38')
        db.session.add(fea[i])
    for i in range(151,154):
        fea[i]=Feature(features=fea_arr[i],group_id='39')
        db.session.add(fea[i])
        #40
    for i in range(154,157):
        fea[i]=Feature(features=fea_arr[i],group_id='40')
        db.session.add(fea[i])
    for i in range(157,160):
        fea[i]=Feature(features=fea_arr[i],group_id='41')
        db.session.add(fea[i])
    for i in range(160,163):
        fea[i]=Feature(features=fea_arr[i],group_id='42')
        db.session.add(fea[i])
    for i in range(163,166):
        fea[i]=Feature(features=fea_arr[i],group_id='43')
        db.session.add(fea[i])
    for i in range(166,169):
        fea[i]=Feature(features=fea_arr[i],group_id='44')
        db.session.add(fea[i])
    for i in range(169,172):
        fea[i]=Feature(features=fea_arr[i],group_id='45')
        db.session.add(fea[i])
    for i in range(172,175):
        fea[i]=Feature(features=fea_arr[i],group_id='46')
        db.session.add(fea[i])
    for i in range(175,179):
        fea[i]=Feature(features=fea_arr[i],group_id='47')
        db.session.add(fea[i])
    for i in range(179,183):
        fea[i]=Feature(features=fea_arr[i],group_id='48')
        db.session.add(fea[i])
    for i in range(183,187):
        fea[i]=Feature(features=fea_arr[i],group_id='49')
        db.session.add(fea[i])
    #50
    for i in range(187,190):
        fea[i]=Feature(features=fea_arr[i],group_id='50')
        db.session.add(fea[i])
    for i in range(190,193):
        fea[i]=Feature(features=fea_arr[i],group_id='51')
        db.session.add(fea[i])
    for i in range(193,196):
        fea[i]=Feature(features=fea_arr[i],group_id='52')
        db.session.add(fea[i])
    for i in range(196,199):
        fea[i]=Feature(features=fea_arr[i],group_id='53')
        db.session.add(fea[i])
    for i in range(199,202):
        fea[i]=Feature(features=fea_arr[i],group_id='54')
        db.session.add(fea[i])
    for i in range(202,206):
        fea[i]=Feature(features=fea_arr[i],group_id='55')
        db.session.add(fea[i])
    for i in range(206,210):
        fea[i]=Feature(features=fea_arr[i],group_id='56')
        db.session.add(fea[i])
    for i in range(210,213):
        fea[i]=Feature(features=fea_arr[i],group_id='57')
        db.session.add(fea[i])
    for i in range(213,216):
        fea[i]=Feature(features=fea_arr[i],group_id='58')
        db.session.add(fea[i])
    for i in range(216,219):
        fea[i]=Feature(features=fea_arr[i],group_id='59')
        db.session.add(fea[i])
    for i in range(219,222):
        fea[i]=Feature(features=fea_arr[i],group_id='60')
        db.session.add(fea[i])
    for i in range(222,228):
        fea[i]=Feature(features=fea_arr[i],group_id='61')
        db.session.add(fea[i])
    for i in range(228,234):
        fea[i]=Feature(features=fea_arr[i],group_id='62')
        db.session.add(fea[i])
    for i in range(234,238):
        fea[i]=Feature(features=fea_arr[i],group_id='63')
        db.session.add(fea[i])
    for i in range(238,243):
        fea[i]=Feature(features=fea_arr[i],group_id='64')
        db.session.add(fea[i])
    for i in range(243,247):
        fea[i]=Feature(features=fea_arr[i],group_id='65')
        db.session.add(fea[i])
    for i in range(247,251):
        fea[i]=Feature(features=fea_arr[i],group_id='66')
        db.session.add(fea[i])
    for i in range(251,256):
        fea[i]=Feature(features=fea_arr[i],group_id='67')
        db.session.add(fea[i])
    for i in range(256,260):
        fea[i]=Feature(features=fea_arr[i],group_id='68')
        db.session.add(fea[i])
    for i in range(260,263):
        fea[i]=Feature(features=fea_arr[i],group_id='69')
        db.session.add(fea[i])
    for i in range(263,266):
        fea[i]=Feature(features=fea_arr[i],group_id='70')
        db.session.add(fea[i])
    for i in range(266,270):
        fea[i]=Feature(features=fea_arr[i],group_id='71')
        db.session.add(fea[i])
    for i in range(270,274):
        fea[i]=Feature(features=fea_arr[i],group_id='72')
        db.session.add(fea[i])
    for i in range(274,278):
        fea[i]=Feature(features=fea_arr[i],group_id='73')
        db.session.add(fea[i])
    for i in range(278,282):
        fea[i]=Feature(features=fea_arr[i],group_id='74')
        db.session.add(fea[i])
    for i in range(282,285):
        fea[i]=Feature(features=fea_arr[i],group_id='75')
        db.session.add(fea[i])
    for i in range(285,288):
        fea[i]=Feature(features=fea_arr[i],group_id='76')
        db.session.add(fea[i])
    for i in range(288,292):
        fea[i]=Feature(features=fea_arr[i],group_id='77')
        db.session.add(fea[i])
    for i in range(292,297):
        fea[i]=Feature(features=fea_arr[i],group_id='78')
        db.session.add(fea[i])
    for i in range(297,300):
        fea[i]=Feature(features=fea_arr[i],group_id='79')
        db.session.add(fea[i])
    for i in range(300,303):
        fea[i]=Feature(features=fea_arr[i],group_id='80')
        db.session.add(fea[i])
    for i in range(303,307):
        fea[i]=Feature(features=fea_arr[i],group_id='81')
        db.session.add(fea[i])
    for i in range(307,310):
        fea[i]=Feature(features=fea_arr[i],group_id='82')
        db.session.add(fea[i])
    for i in range(310,314):
        fea[i]=Feature(features=fea_arr[i],group_id='83')
        db.session.add(fea[i])
    for i in range(314,317):
        fea[i]=Feature(features=fea_arr[i],group_id='84')
        db.session.add(fea[i])
    for i in range(317,320):
        fea[i]=Feature(features=fea_arr[i],group_id='85')
        db.session.add(fea[i])
    for i in range(320,324):
        fea[i]=Feature(features=fea_arr[i],group_id='86')
        db.session.add(fea[i])
    for i in range(324,327):
        fea[i]=Feature(features=fea_arr[i],group_id='87')
        db.session.add(fea[i])
    for i in range(327,330):
        fea[i]=Feature(features=fea_arr[i],group_id='88')
        db.session.add(fea[i])
    for i in range(330,334):
        fea[i]=Feature(features=fea_arr[i],group_id='89')
        db.session.add(fea[i])
    for i in range(334,340):
        fea[i]=Feature(features=fea_arr[i],group_id='90')
        db.session.add(fea[i])
    for i in range(340,344):
        fea[i]=Feature(features=fea_arr[i],group_id='91')
        db.session.add(fea[i])
    for i in range(344,348):
        fea[i]=Feature(features=fea_arr[i],group_id='92')
        db.session.add(fea[i])
    for i in range(348,353):
        fea[i]=Feature(features=fea_arr[i],group_id='93')
        db.session.add(fea[i])
    for i in range(353,356):
        fea[i]=Feature(features=fea_arr[i],group_id='94')
        db.session.add(fea[i])
    for i in range(356,361):
        fea[i]=Feature(features=fea_arr[i],group_id='95')
        db.session.add(fea[i])
    for i in range(361,365):
        fea[i]=Feature(features=fea_arr[i],group_id='96')
        db.session.add(fea[i])
    for i in range(365,370):
        fea[i]=Feature(features=fea_arr[i],group_id='97')
        db.session.add(fea[i])
    for i in range(370,373):
        fea[i]=Feature(features=fea_arr[i],group_id='98')
        db.session.add(fea[i])
    for i in range(373,376):
        fea[i]=Feature(features=fea_arr[i],group_id='99')
        db.session.add(fea[i])
    for i in range(376,380):
        fea[i]=Feature(features=fea_arr[i],group_id='100')
        db.session.add(fea[i])

    sol_arr = ['供电', '开关位置', 'F电源', 'F电源保险管', 'F电源控制模块', 'F电源模块', '更换', 'F组合', '相应圆筒控制板', '相应圆筒IO插板', '相应圆筒模拟板',
               '检查', '控制机柜', '输出信号', '1X11插头6、7两点', '5#424插板', '检查功能', '“电机启动”开关', '控制箱继电器', 'W128电缆', '电源箱',
                '开关量输入板III', '开关量输出板', '“电机启动”开关', '控制箱继电器', 'W128电缆', '电源箱', '检查', '比例溢流阀YV1电缆插头', '“调压”电位器', '车辆转接盒', '比例溢流阀YV1',
                '操作吊机', '减小起重力矩', '报警指示灯', 'OLP按钮', '厂家解决', '检查', '钢丝绳', '调整螺母位置', '刚柔转换油缸', '维修',
                '检查', '参数设置', '电缆连接', '更换接口', '更换从站有线数传单元', '检查', '更换', '手柄', '锁止机构销轴', '运动正常',
               '及时','拆修','更换','阀芯','及时','拆卸清洗','保证','液压油','清洁','修整','单向阀','更换','弹簧','修整','节流阀','更换','新阀',
               '清洗','保证','液压油','清洁','采用','合适粘度','油液','加强降温','注意','日常维护保养','定期检查','拧紧','锁紧装置',
               '寻找','系统','压力变化原因','对症解决','查找','原因','减小','泄漏','修理','单向阀','更换','新阀',

               '检修','更换','调低','压力','调整','更换','调整','重新安装','检查','油温','加强','润滑',
               '挡圈','消除','更换','新件','修理','更换','更换','密封件','增设','支撑圈','挡圈','更换','液压油','密封圈','防止','接触溶剂','更换','液压油',
               '更换','油液','高强度滤油器','更换','高质量产品','更换','新滤油器','检查','密封圈','更换','氮气瓶','漏气附件',
               '更换','已损零件','更换','已损零件','更换','已损零件','及时','补气','及时','补气','调整','系统','更换','大容量蓄能器','检查','原因','修理','更换',
               '调大','水量','风量','设置','降温装置',

               '及时','清洗','排除','阻塞','及时','清洗','排除','阻塞','及时','清洗','排除','阻塞','及时','清洗','排除','阻塞','加大','管径',
               '检查','油量','加油','灌油','排气','拧紧','有关接头','修复','液压泵','更换','中心弹簧',
               '设置','排气装置','诊断','液压泵','吸油端管路','调整','密封圈压紧度','校正','更换','活塞杆','检查','液压泵','调整','流量调节阀',
               
               '检测','电源电压','更换','电磁铁','更换','密封圈','拧紧','螺钉','改用','湿式直流电磁铁',
               '增加','卸荷时间','增大','回油管径','选用','匹配的','溢流阀','更换','调压弹簧','及时','修理',
               '密封','卸荷口','及时','清洗','更换','弹簧','修理','更换','拧紧','更换',
               '清除','吸油口','堵塞','更换','接触面','更换','变量机构','排气','检查','进气部位','采取','隔离','消振','措施',
               '检查','油灌','密封部位','确保','密封','更换','油液','增大','冷却装置','净化','液压油','更换','弹簧','更换','滑靴',
               '改善','供油','更换','弹簧','更换','密封圈','校正','同心','采取','密封措施',
               '增设','排气装置','更换','密封圈','加强','润滑','检查','油源','调整','密封程度',
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

    for j in range(0,6):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='1')
        db.session.add(sol[j])
    for j in range(6,11):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='2')
        db.session.add(sol[j])
    for j in range(11,16):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='3')
        db.session.add(sol[j])
    for j in range(16,21):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='4')
        db.session.add(sol[j])
    for j in range(21,27):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='5')
        db.session.add(sol[j])
    for j in range(27,32):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='6')
        db.session.add(sol[j])
    for j in range(32,37):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='7')
        db.session.add(sol[j])
    for j in range(37,42):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='8')
        db.session.add(sol[j])
    for j in range(42,47):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='9')
        db.session.add(sol[j])
    for j in range(47,52):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='10')
        db.session.add(sol[j])
    for j in range(52,56):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='11')
        db.session.add(sol[j])
    for j in range(56,61):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='12')
        db.session.add(sol[j])
    for j in range(61,65):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='13')
        db.session.add(sol[j])
    for j in range(65,69):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='14')
        db.session.add(sol[j])
    for j in range(69,73):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='15')
        db.session.add(sol[j])
    for j in range(73,77):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='16')
        db.session.add(sol[j])
    for j in range(77,82):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='17')
        db.session.add(sol[j])
    for j in range(82,86):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='18')
        db.session.add(sol[j])
    for j in range(86,90):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='19')
        db.session.add(sol[j])
    for j in range(90,94):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='20')
        db.session.add(sol[j])
    for j in range(94,96):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='21')
        db.session.add(sol[j])
    for j in range(96,98):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='22')
        db.session.add(sol[j])
    for j in range(98,100):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='23')
        db.session.add(sol[j])
    for j in range(100,102):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='24')
        db.session.add(sol[j])
    for j in range(102,104):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='25')
        db.session.add(sol[j])
    for j in range(104,106):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='26')
        db.session.add(sol[j])
    for j in range(106,108):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='27')
        db.session.add(sol[j])
    for j in range(108,110):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='28')
        db.session.add(sol[j])
    for j in range(110,112):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='29')
        db.session.add(sol[j])
    for j in range(112,114):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='30')
        db.session.add(sol[j])
    for j in range(114,117):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='31')
        db.session.add(sol[j])
    for j in range(117,120):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='32')
        db.session.add(sol[j])
    for j in range(120,122):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='33')
        db.session.add(sol[j])
    for j in range(122,124):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='34')
        db.session.add(sol[j])
    for j in range(124,127):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='35')
        db.session.add(sol[j])
    for j in range(127,129):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='36')
        db.session.add(sol[j])
    for j in range(129,131):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='37')
        db.session.add(sol[j])
    for j in range(131,133):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='38')
        db.session.add(sol[j])
    for j in range(133,136):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='39')
        db.session.add(sol[j])
        #40
    for j in range(136,138):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='40')
        db.session.add(sol[j])
    for j in range(138,140):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='41')
        db.session.add(sol[j])
    for j in range(140,142):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='42')
        db.session.add(sol[j])
    for j in range(142,144):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='43')
        db.session.add(sol[j])
    for j in range(144,146):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='44')
        db.session.add(sol[j])
    for j in range(146,148):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='45')
        db.session.add(sol[j])
    for j in range(148,150):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='46')
        db.session.add(sol[j])
    for j in range(150,152):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='47')
        db.session.add(sol[j])
    for j in range(152,154):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='48')
        db.session.add(sol[j])
    for j in range(154,157):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='49')
        db.session.add(sol[j])
    #50
    for j in range(157,159):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='50')
        db.session.add(sol[j])
    for j in range(159,163):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='51')
        db.session.add(sol[j])
    for j in range(163,167):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='52')
        db.session.add(sol[j])
    for j in range(167,171):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='53')
        db.session.add(sol[j])
    for j in range(171,175):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='54')
        db.session.add(sol[j])
    for j in range(175,177):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='55')
        db.session.add(sol[j])
    for j in range(177,180):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='56')
        db.session.add(sol[j])
    for j in range(180,182):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='57')
        db.session.add(sol[j])
    for j in range(182,184):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='58')
        db.session.add(sol[j])
    for j in range(184,186):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='59')
        db.session.add(sol[j])
    for j in range(186,188):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='60')
        db.session.add(sol[j])
    for j in range(188,190):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='61')
        db.session.add(sol[j])
    for j in range(190,193):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='62')
        db.session.add(sol[j])
    for j in range(193,195):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='63')
        db.session.add(sol[j])
    for j in range(195,198):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='64')
        db.session.add(sol[j])
    for j in range(198,202):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='65')
        db.session.add(sol[j])
    for j in range(202,204):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='66')
        db.session.add(sol[j])
    for j in range(204,206):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='67')
        db.session.add(sol[j])
    for j in range(206,208):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='68')
        db.session.add(sol[j])
    for j in range(208,210):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='69')
        db.session.add(sol[j])
    for j in range(210,212):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='70')
        db.session.add(sol[j])
    for j in range(212,214):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='71')
        db.session.add(sol[j])
    for j in range(214,216):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='72')
        db.session.add(sol[j])
    for j in range(216,219):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='73')
        db.session.add(sol[j])
    for j in range(219,221):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='74')
        db.session.add(sol[j])
    for j in range(221,223):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='75')
        db.session.add(sol[j])
    for j in range(223,225):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='76')
        db.session.add(sol[j])
    for j in range(225,227):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='77')
        db.session.add(sol[j])
    for j in range(227,229):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='78')
        db.session.add(sol[j])
    for j in range(229,231):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='79')
        db.session.add(sol[j])
    for j in range(231,233):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='80')
        db.session.add(sol[j])
    for j in range(233,236):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='81')
        db.session.add(sol[j])
    for j in range(236,238):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='82')
        db.session.add(sol[j])
    for j in range(238,240):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='83')
        db.session.add(sol[j])
    for j in range(240,243):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='84')
        db.session.add(sol[j])
    for j in range(243,247):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='85')
        db.session.add(sol[j])
    for j in range(247,252):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='86')
        db.session.add(sol[j])
    for j in range(252,256):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='87')
        db.session.add(sol[j])
    for j in range(256,258):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='88')
        db.session.add(sol[j])
    for j in range(258,260):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='89')
        db.session.add(sol[j])
    for j in range(260,262):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='90')
        db.session.add(sol[j])
    for j in range(262,264):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='91')
        db.session.add(sol[j])
    for j in range(264,266):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='92')
        db.session.add(sol[j])
    for j in range(266,268):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='93')
        db.session.add(sol[j])
    for j in range(268,270):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='94')
        db.session.add(sol[j])
    for j in range(270,272):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='95')
        db.session.add(sol[j])
    for j in range(272,274):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='96')
        db.session.add(sol[j])
    for j in range(274,276):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='97')
        db.session.add(sol[j])
    for j in range(276,278):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='98')
        db.session.add(sol[j])
    for j in range(278,280):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='99')
        db.session.add(sol[j])
    for j in range(280,282):
        sol[j]=Sol_feature(sol_features=sol_arr[j],group_id='100')
        db.session.add(sol[j])




    # 提交会话
    db.session.commit()

    db = pymysql.connect(host='localhost', user='root', password='zhangyixin', port=3306, db='faults_test')

    for i in range(1, 7):
        for j in range(1, 7):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(7, 13):
        for j in range(7, 12):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(13, 19):
        for j in range(12, 17):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(19, 25):
        for j in range(17, 22):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(25, 30):
        for j in range(22, 28):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(30, 35):
        for j in range(28, 33):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(35, 40):
        for j in range(33, 38):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(40, 45):
        for j in range(38, 43):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(45, 50):
        for j in range(43, 48):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(50, 55):
        for j in range(48, 53):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(55, 59):
        for j in range(53, 57):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(59, 63):
        for j in range(57, 62):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(63, 67):
        for j in range(62, 66):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(67, 71):
        for j in range(66, 70):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(71, 75):
        for j in range(70, 74):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(75, 80):
        for j in range(74, 78):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(80, 86):
        for j in range(78, 83):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(86, 90):
        for j in range(83, 87):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(90, 94):
        for j in range(87, 91):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(94, 98):
        for j in range(91, 95):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #21
    for i in range(98, 101):
        for j in range(95, 97):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(101, 104):
        for j in range(97, 99):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(104, 107):
        for j in range(99, 101):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(107, 110):
        for j in range(101, 103):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #25
    for i in range(110, 113):
        for j in range(103, 105):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(113, 116):
        for j in range(105, 107):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(116, 119):
        for j in range(107, 109):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(119, 122):
        for j in range(109, 111):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(122, 125):
        for j in range(111, 113):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #30
    for i in range(125, 128):
        for j in range(113, 115):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(128, 131):
        for j in range(115, 118):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(131, 134):
        for j in range(118, 121):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(134, 137):
        for j in range(121, 123):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(137, 140):
        for j in range(123, 125):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #35
    for i in range(140, 143):
        for j in range(125, 128):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(143, 146):
        for j in range(128, 130):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(146, 149):
        for j in range(130, 132):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(149, 152):
        for j in range(132, 134):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(152, 155):
        for j in range(134, 137):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #40
    for i in range(155, 158):
        for j in range(137, 139):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(158, 161):
        for j in range(139, 141):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(161, 164):
        for j in range(141, 143):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(164, 167):
        for j in range(143, 145):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(167, 170):
        for j in range(145, 147):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #45
    for i in range(170, 173):
        for j in range(147, 149):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(173, 176):
        for j in range(149, 151):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(176, 180):
        for j in range(151, 153):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(180, 184):
        for j in range(153, 155):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(184, 188):
        for j in range(155, 158):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #50
    for i in range(188, 191):
        for j in range(158, 160):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(191, 194):
        for j in range(160, 164):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(194, 197):
        for j in range(164, 168):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(197, 200):
        for j in range(168, 172):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(200, 203):
        for j in range(172, 176):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(203, 207):
        for j in range(176, 178):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(207, 211):
        for j in range(178, 181):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(211, 214):
        for j in range(181, 183):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(214, 217):
        for j in range(183, 185):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(217, 220):
        for j in range(185, 187):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #60
    for i in range(220, 223):
        for j in range(187, 189):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(223, 229):
        for j in range(189, 191):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(229, 235):
        for j in range(191, 194):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(235, 239):
        for j in range(194, 196):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(239, 244):
        for j in range(196, 199):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(244, 248):
        for j in range(199, 203):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(248, 252):
        for j in range(203, 205):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(252, 257):
        for j in range(205, 207):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(257,261):
        for j in range(207, 209):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(261, 264):
        for j in range(209, 211):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #70
    for i in range(264, 267):
        for j in range(211, 213):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(267, 271):
        for j in range(213, 215):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #72
    for i in range(271, 275):
        for j in range(215, 217):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(275, 279):
        for j in range(217, 220):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #74
    for i in range(279, 283):
        for j in range(220, 222):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(283, 286):
        for j in range(222, 224):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #76
    for i in range(286, 289):
        for j in range(224, 226):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(289, 293):
        for j in range(226, 228):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #78
    for i in range(293, 298):
        for j in range(228, 230):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(298, 301):
        for j in range(230, 232):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #80
    for i in range(301, 304):
        for j in range(232, 234):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(304, 308):
        for j in range(234, 237):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #82
    for i in range(308, 311):
        for j in range(237, 239):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(311, 315):
        for j in range(239, 241):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #84
    for i in range(315, 318):
        for j in range(241, 244):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(318, 321):
        for j in range(244, 248):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #86
    for i in range(321, 325):
        for j in range(248, 253):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(325, 328):
        for j in range(253, 257):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #88
    for i in range(328, 331):
        for j in range(257, 259):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(331, 335):
        for j in range(259, 261):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #90
    for i in range(335, 341):
        for j in range(261, 263):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(341, 345):
        for j in range(263, 265):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #92
    for i in range(345, 349):
        for j in range(265, 267):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(349, 354):
        for j in range(267, 269):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #94
    for i in range(354, 357):
        for j in range(269, 271):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(357, 362):
        for j in range(271, 273):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #96
    for i in range(362, 366):
        for j in range(273, 275):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(366, 371):
        for j in range(275, 277):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #98
    for i in range(371, 374):
        for j in range(277, 279):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    for i in range(374, 377):
        for j in range(279, 281):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    #100
    for i in range(377, 381):
        for j in range(281, 283):
            cursor = db.cursor()
            sql = 'insert into fea2sol values ("%s", "%s")' % (i, j)
            cursor.execute(sql)
            cursor.close()
    db.commit()
    db.close()


