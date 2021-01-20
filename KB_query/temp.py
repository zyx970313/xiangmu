# encoding=utf-8
"""
@desc:
设置问题模板，为每个模板设置对应的SPARQL语句。demo提供如下模板：
"""
from refo import finditer, Predicate, Star, Any, Disjunction
import re

# TODO SPARQL前缀和模板
SPARQL_PREXIX = u"""
PREFIX : <http://www.faultdemo.com#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
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
    def has_solution5(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a=None
        b=None
        c = None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea1 :
                a=w.token
            if w.pos == pos_fea2:
                b=w.token
            if w.pos == pos_fea3 :
                c=w.token
            if w.pos == pos_fea4:
                d=w.token
            if w.pos == pos_fea5 :
                e=w.token
        if a is not None and b is not None and c is not None and d is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1."\
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),fea2=b.decode('utf-8'),fea3=c.decode('utf-8'),fea4=d.decode('utf-8'),fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None



    @staticmethod
    def has_solution41(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        b = None
        c = None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea2:
                b = w.token
            if w.pos == pos_fea3:
                c = w.token
            if w.pos == pos_fea4:
                d = w.token
            if w.pos == pos_fea5:
                e = w.token
        if b is not None and c is not None and d is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea2=b.decode('utf-8'), fea3=c.decode('utf-8'),
                                          fea4=d.decode('utf-8'), fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution42(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a = None
        c = None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea1:
                a = w.token
            if w.pos == pos_fea3:
                c = w.token
            if w.pos == pos_fea4:
                d = w.token
            if w.pos == pos_fea5:
                e = w.token
        if a is not None and c is not None and d is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),
                                          fea3=c.decode('utf-8'), fea4=d.decode('utf-8'),
                                          fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution43(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a = None
        b = None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea1:
                a = w.token
            if w.pos == pos_fea2:
                b = w.token
            if w.pos == pos_fea4:
                d = w.token
            if w.pos == pos_fea5:
                e = w.token
        if a is not None and b is not None and d is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'),
                                          fea4=d.decode('utf-8'),
                                          fea5=e.decode('utf-8'))
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution44(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a = None
        b = None
        c = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea1:
                a = w.token
            if w.pos == pos_fea2:
                b = w.token
            if w.pos == pos_fea3:
                c = w.token
            if w.pos == pos_fea5:
                e = w.token
        if a is not None and b is not None and c is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'),
                                          fea3=c.decode('utf-8'),
                                          fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)

            return sparql
        else:
            return None

    @staticmethod
    def has_solution45(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a = None
        b = None
        c = None
        d = None
        for w in word_objects:
            if w.pos == pos_fea1:
                a = w.token
            if w.pos == pos_fea2:
                b = w.token
            if w.pos == pos_fea3:
                c = w.token
            if w.pos == pos_fea4:
                d = w.token
        if a is not None and b is not None and c is not None and d is not None :
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8'),
                                          fea3=c.decode('utf-8'),
                                          fea4=d.decode('utf-8'))
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3123(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a=None
        b=None
        c = None
        for w in word_objects:
            if w.pos == pos_fea1 :
                a=w.token
            if w.pos == pos_fea2:
                b=w.token
            if w.pos == pos_fea3 :
                c=w.token
        if a is not None and b is not None and c is not None :
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),fea2=b.decode('utf-8'),fea3=c.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3124(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a=None
        b=None
        d = None
        for w in word_objects:
            if w.pos == pos_fea1 :
                a=w.token
            if w.pos == pos_fea2:
                b=w.token
            if w.pos == pos_fea4:
                d=w.token
        if a is not None and b is not None and d is not None :
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),fea2=b.decode('utf-8'),fea4=d.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3125(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a=None
        b=None
        e = None
        for w in word_objects:
            if w.pos == pos_fea1 :
                a=w.token
            if w.pos == pos_fea2:
                b=w.token
            if w.pos == pos_fea5 :
                e=w.token
        if a is not None and b is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),fea2=b.decode('utf-8'),fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3134(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a=None
        c = None
        d = None
        for w in word_objects:
            if w.pos == pos_fea1 :
                a=w.token
            if w.pos == pos_fea3 :
                c=w.token
            if w.pos == pos_fea4:
                d=w.token
        if a is not None and c is not None and d is not None :
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),fea3=c.decode('utf-8'),fea4=d.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3135(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a=None
        c = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea1 :
                a=w.token
            if w.pos == pos_fea3 :
                c=w.token
            if w.pos == pos_fea5 :
                e=w.token
        if a is not None and  c is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),fea3=c.decode('utf-8'),fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3145(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a=None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea1 :
                a=w.token
            if w.pos == pos_fea4:
                d=w.token
            if w.pos == pos_fea5 :
                e=w.token
        if a is not None and d is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),fea4=d.decode('utf-8'),fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3234(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        b=None
        c = None
        d = None
        for w in word_objects:
            if w.pos == pos_fea2:
                b=w.token
            if w.pos == pos_fea3 :
                c=w.token
            if w.pos == pos_fea4:
                d=w.token
        if b is not None and c is not None and d is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea2=b.decode('utf-8'),fea3=c.decode('utf-8'),fea4=d.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3235(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        b=None
        c = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea2:
                b=w.token
            if w.pos == pos_fea3 :
                c=w.token
            if w.pos == pos_fea5 :
                e=w.token
        if b is not None and c is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea2=b.decode('utf-8'),fea3=c.decode('utf-8'),fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3245(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        b=None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea2:
                b=w.token
            if w.pos == pos_fea4:
                d=w.token
            if w.pos == pos_fea5 :
                e=w.token
        if b is not None and d is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea2=b.decode('utf-8'),fea4=d.decode('utf-8'),fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution3345(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        c = None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea3 :
                c=w.token
            if w.pos == pos_fea4:
                d=w.token
            if w.pos == pos_fea5 :
                e=w.token
        if c is not None and d is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea3=c.decode('utf-8'),fea4=d.decode('utf-8'),fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution212(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a = None
        b = None
        for w in word_objects:
            if w.pos == pos_fea1:
                a = w.token
            if w.pos == pos_fea2:
                b = w.token
        if a is not None and b is not None :
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'), fea2=b.decode('utf-8')
                                         )
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution213(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a = None
        c = None
        for w in word_objects:
            if w.pos == pos_fea1:
                a = w.token
            if w.pos == pos_fea3:
                c = w.token
        if a is not None and c is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'), fea3=c.decode('utf-8'),
                                          )
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution214(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a = None
        d = None
        for w in word_objects:
            if w.pos == pos_fea1:
                a = w.token
            if w.pos == pos_fea4:
                d = w.token
        if a is not None and d is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),
                                          fea4=d.decode('utf-8'))
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution215(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        a = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea1:
                a = w.token
            if w.pos == pos_fea5:
                e = w.token
        if a is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=a.decode('utf-8'),
                                           fea5=e.decode('utf-8'))
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution223(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        b = None
        c = None
        for w in word_objects:
            if w.pos == pos_fea2:
                b = w.token
            if w.pos == pos_fea3:
                c = w.token
        if b is not None and c is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format( fea2=b.decode('utf-8'), fea3=c.decode('utf-8'),
                                          )
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution224(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        b = None
        d = None
        for w in word_objects:
            if w.pos == pos_fea2:
                b = w.token
            if w.pos == pos_fea4:
                d = w.token
        if b is not None and d is not None :
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea2=b.decode('utf-8'),
                                          fea4=d.decode('utf-8'))
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None
    @staticmethod
    def has_solution225(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        b = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea2:
                b = w.token
            if w.pos == pos_fea5:
                e = w.token
        if b is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea2=b.decode('utf-8')
                                          , fea5=e.decode('utf-8'))
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution234(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        c = None
        d = None
        for w in word_objects:
            if w.pos == pos_fea3:
                c = w.token
            if w.pos == pos_fea4:
                d = w.token
        if  c is not None and d is not None :
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea3=c.decode('utf-8'),
                                          fea4=d.decode('utf-8'))
            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)
            return sparql
        else:
            return None

    @staticmethod
    def has_solution235(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        c = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea3:
                c = w.token
            if w.pos == pos_fea5:
                e = w.token
        if c is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea3=c.decode('utf-8'),
                                           fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)

            return sparql
        else:
            return None

    @staticmethod
    def has_solution245(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        d = None
        e = None
        for w in word_objects:
            if w.pos == pos_fea4:
                d = w.token
            if w.pos == pos_fea5:
                e = w.token
        if d is not None and e is not None:
            e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(
                                          fea4=d.decode('utf-8'), fea5=e.decode('utf-8'))

            sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                              select=select,
                                              expression=e)

            return sparql
        else:
            return None


    @staticmethod
    def has_solution11(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        for w in word_objects:
            if w.pos == pos_fea1:
                e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature1> '{fea1}'." \
                    u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                    u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea1=w.token.decode('utf-8'))

                sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                                  select=select,
                                                  expression=e)
                break
        return sparql

    @staticmethod
    def has_solution12(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        for w in word_objects:
            if w.pos == pos_fea1:
                e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature2> '{fea2}'." \
                    u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                    u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea2=w.token.decode('utf-8'))

                sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                                  select=select,
                                                  expression=e)
                break
        return sparql

    @staticmethod
    def has_solution13(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        for w in word_objects:
            if w.pos == pos_fea1:
                e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature3> '{fea3}''." \
                    u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                    u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea3=w.token.decode('utf-8'))

                sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                                  select=select,
                                                  expression=e)
                break
        return sparql

    @staticmethod
    def has_solution14(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        for w in word_objects:
            if w.pos == pos_fea1:
                e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature4> '{fea4}'." \
                    u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                    u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea4=w.token.decode('utf-8'))

                sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                                  select=select,
                                                  expression=e)
                break
        return sparql

    @staticmethod
    def has_solution15(word_objects):
        select = u"?x1 ?x2 ?x3 ?x4 ?x5"
        sparql = None
        for w in word_objects:
            if w.pos == pos_fea1:
                e = u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/features_feature5> '{fea5}'." \
                    u"?s <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/fea2sol> ?m." \
                    u"?m <file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature1> ?x1." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature2> ?x2." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature3> ?x3." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature4> ?x4." \
                    u"?m<file:///D:/pycode/library/d2rq/d2rq-0.8.1/vocab/sol_features_sol_feature5> ?x5".format(fea5=w.token.decode('utf-8'))

                sparql = SPARQL_SELECT_TEM.format(prefix=SPARQL_PREXIX,
                                                  select=select,
                                                  expression=e)
                break
        return sparql


# TODO 定义关键词
pos_fea1 = "one"
pos_fea2 = "two"
pos_fea3 = "three"
pos_fea4 = "four"
pos_fea5 = "five"

fea1_entity = (W(pos=pos_fea1))
fea2_entity = (W(pos=pos_fea2))
fea3_entity = (W(pos=pos_fea3))
fea4_entity = (W(pos=pos_fea4))
fea5_entity = (W(pos=pos_fea5))


#Rule(condition_num=5,condition=fea1_entity + Star(Any(),greedy=False) + fea2_entity + Star(Any(),greedy=False) + fea3_entity + Star(Any(),greedy=False) + fea4_entity + Star(Any(),greedy=False) + fea5_entity + Star(Any(),greedy=False),action=QuestionSet.has_solution_question),

rules = [Rule(condition_num=5,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution5),
Rule(condition_num=4,condition=Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution41),
Rule(condition_num=4,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution42),
Rule(condition_num=4,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea2_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution43),
Rule(condition_num=4,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution44),
Rule(condition_num=4,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution45),
Rule(condition_num=1,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution11),
Rule(condition_num=1,condition=Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False),action=QuestionSet.has_solution12),
Rule(condition_num=1,condition=Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False) ,action=QuestionSet.has_solution13),
Rule(condition_num=1,condition=Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False),action=QuestionSet.has_solution14),
Rule(condition_num=1,condition=Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution15),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution212),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea1_entity +Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False) ,action=QuestionSet.has_solution213),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution214),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution215),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False) ,action=QuestionSet.has_solution223),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution224),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution225),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution234),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution235),
Rule(condition_num=2,condition=Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution245),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False) ,action=QuestionSet.has_solution3123),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False),action=QuestionSet.has_solution3124),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution3125),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution3134),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution3135),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea1_entity + Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution3145),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False) ,action=QuestionSet.has_solution3234),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution3235),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+fea2_entity+ Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution3245),
Rule(condition_num=3,condition=Star(Any(),greedy=False)+ fea3_entity+Star(Any(),greedy=False)+fea4_entity+ Star(Any(),greedy=False)+fea5_entity + Star(Any(),greedy=False) ,action=QuestionSet.has_solution3345),


]

