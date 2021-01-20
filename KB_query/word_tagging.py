# encoding=utf-8

"""

@author: SimmerChan

@contact: hsl7698590@gmail.com

@file: word_tagging.py

@time: 2017/12/20 15:31

@desc: 定义Word类的结构；定义Tagger类，实现自然语言转为Word对象的方法。

"""
import jieba
import jieba.posseg as pseg
from refo import finditer

class Word(object):
    def __init__(self, token, pos):
        self.token = token
        self.pos = pos


class Tagger:
    def __init__(self, dict_paths):
        # TODO 加载外部词典
        for p in dict_paths:
            jieba.load_userdict(p)

        jieba.suggest_freq(('检查', '更换'), True)



    @staticmethod
    def get_word_objects(sentence):
        # type: (str) -> list
        """
        把自然语言转为Word对象
        :param sentence:
        :return:
        """
        return [Word(word.encode('utf-8'), tag) for word, tag in pseg.cut(sentence)]

# TODO 用于测试
if __name__ == '__main__':
    tagger = Tagger(['./external_dict/features.txt', './external_dict/solutions.txt','./external_dict/car','./external_dict/type'])
    while True:
        s = input()
        p = tagger.get_word_objects(s)
        print(p)

        for i in tagger.get_word_objects(s):
            print(i.token.decode('utf8'), i.pos)

