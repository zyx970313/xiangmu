# encoding=utf-8
"""
@desc:
设置问题模板，为每个模板设置对应的SPARQL语句。demo提供如下模板：
"""
from refo import finditer, Predicate, Star, Any, Disjunction
import re

# TODO SPARQL前缀和模板
SPARQL_PREXIX = u"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX : <http://www.kgdemo.com#> 
"""

SPARQL_SELECT_TEM = u"{prefix}\n" + \
             u"SELECT DISTINCT {select} WHERE {{\n" + \
             u"{expression}\n" + \
             u"}}\n"

SPARQL_COUNT_TEM = u"{prefix}\n" + \
             u"SELECT COUNT({select}) WHERE {{\n" + \
             u"{expression}\n" + \
             u"}}\n"


class W(Predicate):
    def __init__(self, token=".*", pos=".*"):
        self.token = re.compile(token + "$")
        self.pos = re.compile(pos + "$")
        super(W, self).__init__(self.match)

    def match(self, word):
        m1 = self.token.match(word.token.decode('utf8'))
        m2 = self.pos.match(word.pos)
        return m1 and m2


class Rule(object):
    def __init__(self, condition_num, condition=None, action=None):
        assert condition and action
        self.condition = condition
        self.action = action
        self.condition_num = condition_num

    def apply(self, sentence):
        matches = []
        # print("before applying....")
        for m in finditer(self.condition, sentence):
            i, j = m.span()
            # print(i,j)
            # for s in sentence[i:j]:
            #     print(s.token.decode('utf8'),s.pos)
            matches.extend(sentence[i:j])

        return self.action(matches), self.condition_num


class KeywordRule(object):
    def __init__(self, condition=None, action=None):
        assert condition and action
        self.condition = condition
        self.action = action

    def apply(self, sentence):
        matches = []
        for m in finditer(self.condition, sentence):
            i, j = m.span()
            matches.extend(sentence[i:j])
        if len(matches) == 0:
            return None
        else:
            return self.action()


class QuestionSet:
    def __init__(self):
        pass

    @staticmethod
    def has_solution18(word_objects):
        select = u"?x"
        sparql = None
        a = None
        b = None
        c = None
        d = None
        e = None
        f = None
        g = None
        h = None
        i = None
        j = None
        k = None
        l = None
        m = None
        n = None
        o = None
        p = None
        q = None
        r = None

        for w in word_objects:
            if w.pos == pos_fea6:
                if a is None:
                    a = w.token
                elif b is None:
                    b = w.token
                elif c is None:
                    c = w.token
                elif d is None:
                    d = w.token
                elif e is None:
                    e = w.token
                elif f is None:
                    f = w.token
        if a is not None and b is not None and c is not None and d is not None and e is not None and f is not None:
            e = u"?p1 :features '{fea1}'." \
                u"?p2 :features '{fea2}'." \
                u"?p3 :features '{fea3}'." \
                u"?p4 :features '{fea4}'." \
                u"?p5 :features '{fea5}'." \
                u"?p6 :features '{fea6}'." \
                u"?p1 :fea2sol ?m." \
                u"?p2 :fea2sol ?m." \
                u"?p3 :fea2sol ?m." \
                u"?p4 :fea2sol ?m." \
                u"?p5 :fea2sol ?m." \
                u"?p6 :fea2sol ?m." \
                u" ?m :sol_features ?x.".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'), fea3=c.decode('utf-8'),
                                                fea4=d.decode('utf-8'), fea5=e.decode('utf-8'), fea6=f.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution7(word_objects):
        select = u"?x"
        sparql = None
        a = None
        b = None
        c = None
        d = None
        e = None
        f = None
        g=None
        for w in word_objects:
            if w.pos == pos_fea6:
                if a is None:
                    a = w.token
                elif b is None:
                    b = w.token
                elif c is None:
                    c = w.token
                elif d is None:
                    d = w.token
                elif e is None:
                    e = w.token
                elif f is None:
                    f = w.token
                elif g is None:
                    g = w.token
        if a is not None and b is not None and c is not None and d is not None and e is not None and f is not None and g is not None:
            e = u"?p1 :features '{fea1}'." \
                u"?p2 :features '{fea2}'." \
                u"?p3 :features '{fea3}'." \
                u"?p4 :features '{fea4}'." \
                u"?p5 :features '{fea5}'." \
                u"?p6 :features '{fea6}'." \
                u"?p7 :features '{fea7}'." \
                u"?p1 :fea2sol ?m." \
                u"?p2 :fea2sol ?m." \
                u"?p3 :fea2sol ?m." \
                u"?p4 :fea2sol ?m." \
                u"?p5 :fea2sol ?m." \
                u"?p6 :fea2sol ?m." \
                u"?p7 :fea2sol ?m." \
                u" ?m :sol_features ?x.".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'), fea3=c.decode('utf-8'),
                                                fea4=d.decode('utf-8'), fea5=e.decode('utf-8'), fea6=f.decode('utf-8'), fea7=f.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution6(word_objects):
        select = u"?x"
        sparql = None
        a=None
        b=None
        c = None
        d = None
        e = None
        f = None
        for w in word_objects:
            if w.pos == pos_fea6 :
                if a is None:
                    a=w.token
                elif b is None:
                    b=w.token
                elif c is None:
                    c=w.token
                elif d is None:
                    d=w.token
                elif e is None:
                    e=w.token
                elif f is None:
                    f=w.token
        if a is not None and b is not None and c is not None and d is not None and e is not None and f is not None:
            e = u"?p1 :features '{fea1}'." \
                u"?p2 :features '{fea2}'." \
                u"?p3 :features '{fea3}'." \
                u"?p4 :features '{fea4}'." \
                u"?p5 :features '{fea5}'." \
                u"?p6 :features '{fea6}'." \
                u"?p1 :fea2sol ?m." \
                u"?p2 :fea2sol ?m."\
                u"?p3 :fea2sol ?m." \
                u"?p4 :fea2sol ?m." \
                u"?p5 :fea2sol ?m." \
                u"?p6 :fea2sol ?m." \
                u" ?m :sol_features ?x.".format(fea1=a.decode('utf-8'),fea2=b.decode('utf-8'),fea3=c.decode('utf-8'),fea4=d.decode('utf-8'),fea5=e.decode('utf-8'),fea6=f.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution5(word_objects):
        select = u"?x"
        sparql = None
        a=None
        b=None
        c = None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea6 :
                if a is None:
                    a=w.token
                elif b is None:
                    b=w.token
                elif c is None:
                    c=w.token
                elif d is None:
                    d=w.token
                elif e is None:
                    e=w.token
        if a is not None and b is not None and c is not None and d is not None and e is not None:
            e = u"?p1 :features '{fea1}'." \
                u"?p2 :features '{fea2}'." \
                u"?p3 :features '{fea3}'." \
                u"?p4 :features '{fea4}'." \
                u"?p5 :features '{fea5}'." \
                u"?p1 :fea2sol ?m." \
                u"?p2 :fea2sol ?m."\
                u"?p3 :fea2sol ?m." \
                u"?p4 :fea2sol ?m." \
                u"?p5 :fea2sol ?m." \
                u" ?m :sol_features ?x.".format(fea1=a.decode('utf-8'),fea2=b.decode('utf-8'),fea3=c.decode('utf-8'),fea4=d.decode('utf-8'),fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution4(word_objects):
        select = u"?x"
        sparql = None
        a = None
        b = None
        c = None
        d = None
        for w in word_objects:
            if w.pos == pos_fea6:
                if a is None:
                    a = w.token
                elif b is None:
                    b = w.token
                elif c is None:
                    c = w.token
                elif d is None:
                    d = w.token
        if a is not None and b is not None and c is not None and d is not None :
            e = u"?p1 :features '{fea1}'." \
                u"?p2 :features '{fea2}'." \
                u"?p3 :features '{fea3}'." \
                u"?p4 :features '{fea4}'." \
                u"?p1 :fea2sol ?m." \
                u"?p2 :fea2sol ?m." \
                u"?p3 :fea2sol ?m." \
                u"?p4 :fea2sol ?m." \
                u"?m :sol_features ?x".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'), fea3=c.decode('utf-8'),
                                           fea4=d.decode('utf-8'))
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3(word_objects):
        select = u"?x"
        sparql = None
        a = None
        b = None
        c = None
        for w in word_objects:
            if w.pos == pos_fea6:
                if a is None:
                    a = w.token
                elif b is None:
                    b = w.token
                elif c is None:
                    c = w.token
        if a is not None and b is not None and c is not None:
            e = u"?p1 :features '{fea1}'." \
                u"?p2 :features '{fea2}'." \
                u"?p3 :features '{fea3}'." \
                u"?p1 :fea2sol ?m." \
                u"?p2 :fea2sol ?m." \
                u"?p3 :fea2sol ?m." \
                u"?m :sol_features ?x".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'),
                                              fea3=c.decode('utf-8')
                                              )
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution2(word_objects):
        select = u"?x"
        sparql = None
        a = None
        b = None
        for w in word_objects:
            if w.pos == pos_fea6:
                if a is None:
                    a = w.token
                elif b is None:
                    b = w.token
        if a is not None and b is not None :
            e = u"?p1 :features '{fea1}'." \
                u"?p2 :features '{fea2}'." \
                u"?p1 :fea2sol ?m." \
                u"?p2 :fea2sol ?m." \
                u"?m :sol_features ?x".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8')
                                              )
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution1(word_objects):
        select = u"?x"
        sparql = None
        a = None
        for w in word_objects:
            if w.pos == pos_fea6:
                a = w.token
        if a is not None :
            e = u"?p1 :features '{fea1}'." \
                u"?p1 :fea2sol ?m." \
                u"?m :sol_features ?x".format(fea1=a.decode('utf-8')
                                              )
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_fea6(word_objects):
        select = u"?x"
        sparql = None
        a = None
        b = None
        c = None
        d = None
        e = None
        f = None
        for w in word_objects:
            if w.pos == pos_fea7:
                if a is None:
                    a = w.token
                elif b is None:
                    b = w.token
                elif c is None:
                    c = w.token
                elif d is None:
                    d = w.token
                elif e is None:
                    e = w.token
                elif f is None:
                    f = w.token
        if a is not None and b is not None and c is not None and d is not None and e is not None and f is not None:
            e = u"?p1 :sol_features '{fea1}'." \
                u"?p2 :sol_features '{fea2}'." \
                u"?p3 :sol_features '{fea3}'." \
                u"?p4 :sol_features '{fea4}'." \
                u"?p5 :sol_features '{fea5}'." \
                u"?p6 :sol_features '{fea6}'." \
                u"?m :fea2sol ?p1." \
                u"?m :fea2sol ?p2." \
                u"?m :fea2sol ?p3." \
                u"?m :fea2sol ?p4." \
                u"?m :fea2sol ?p5." \
                u"?m :fea2sol ?p6." \
                u" ?m :features ?x.".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'),
                                                fea3=c.decode('utf-8'), fea4=d.decode('utf-8'),
                                                fea5=e.decode('utf-8'), fea6=f.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_fea5(word_objects):
        select = u"?x"
        sparql = None
        a = None
        b = None
        c = None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea7:
                if a is None:
                    a = w.token
                elif b is None:
                    b = w.token
                elif c is None:
                    c = w.token
                elif d is None:
                    d = w.token
                elif e is None:
                    e = w.token

        if a is not None and b is not None and c is not None and d is not None and e is not None :
            e = u"?p1 :sol_features '{fea1}'." \
                u"?p2 :sol_features '{fea2}'." \
                u"?p3 :sol_features '{fea3}'." \
                u"?p4 :sol_features '{fea4}'." \
                u"?p5 :sol_features '{fea5}'." \
                u"?m :fea2sol ?p1." \
                u"?m :fea2sol ?p2." \
                u"?m :fea2sol ?p3." \
                u"?m :fea2sol ?p4." \
                u"?m :fea2sol ?p5." \
                u" ?m :features ?x.".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'),
                                            fea3=c.decode('utf-8'), fea4=d.decode('utf-8'),
                                            fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_fea4(word_objects):
        select = u"?x"
        sparql = None
        a = None
        b = None
        c = None
        d = None
        for w in word_objects:
            if w.pos == pos_fea7:
                if a is None:
                    a = w.token
                elif b is None:
                    b = w.token
                elif c is None:
                    c = w.token
                elif d is None:
                    d = w.token
        if a is not None and b is not None and c is not None and d is not None :
            e = u"?p1 :sol_features '{fea1}'." \
                u"?p2 :sol_features '{fea2}'." \
                u"?p3 :sol_features '{fea3}'." \
                u"?p4 :sol_features '{fea4}'." \
                u"?m :fea2sol ?p1." \
                u"?m :fea2sol ?p2." \
                u"?m :fea2sol ?p3." \
                u"?m :fea2sol ?p4." \
                u" ?m :features ?x.".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'),
                                            fea3=c.decode('utf-8'), fea4=d.decode('utf-8'),
                                          )

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None


# TODO 定义关键词
# pos_fea1 = "one"
# pos_fea2 = "two"
# pos_fea3 = "three"
# pos_fea4 = "four"
# pos_fea5 = "five"
pos_fea6="features"
pos_fea7="solutions"

# fea1_entity = (W(pos=pos_fea1))
# fea2_entity = (W(pos=pos_fea2))
# fea3_entity = (W(pos=pos_fea3))
# fea4_entity = (W(pos=pos_fea4))
# fea5_entity = (W(pos=pos_fea5))
fea6_entity = (W(pos=pos_fea6))
fea7_entity = (W(pos=pos_fea7))


#Rule(condition_num=5,condition=fea1_entity + Star(Any(),greedy=False) + fea2_entity + Star(Any(),greedy=False) + fea3_entity + Star(Any(),greedy=False) + fea4_entity + Star(Any(),greedy=False) + fea5_entity + Star(Any(),greedy=False),action=QuestionSet.has_solution_question),

rules = [
Rule(condition_num=7,condition=Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False)+ fea6_entity+Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea7_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution7),
Rule(condition_num=6,condition=Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False)+ fea6_entity+Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution6),
Rule(condition_num=5,condition=Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False)+ fea6_entity+Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution5),
Rule(condition_num=4,condition=Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False)+ fea6_entity+Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution4),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False)+ fea6_entity+Star(Any(),greedy=False) ,action=QuestionSet.has_solution3),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False)+fea6_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution2),
Rule(condition_num=1,condition=Star(Any(),greedy=False)+fea6_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution1),

Rule(condition_num=6,condition=Star(Any(),greedy=False)+fea7_entity + Star(Any(),greedy=False)+fea7_entity+ Star(Any(),greedy=False)+ fea7_entity+Star(Any(),greedy=False)+fea7_entity+ Star(Any(),greedy=False)+fea7_entity + Star(Any(),greedy=False)+fea7_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_fea6),
Rule(condition_num=5,condition=Star(Any(),greedy=False)+fea7_entity + Star(Any(),greedy=False)+fea7_entity+ Star(Any(),greedy=False)+ fea7_entity+Star(Any(),greedy=False)+fea7_entity+ Star(Any(),greedy=False)+fea7_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_fea5),
Rule(condition_num=4,condition=Star(Any(),greedy=False)+fea7_entity + Star(Any(),greedy=False)+fea7_entity+ Star(Any(),greedy=False)+ fea7_entity+Star(Any(),greedy=False)+fea7_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_fea4),


]

