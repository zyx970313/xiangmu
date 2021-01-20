# encoding=utf-8



from KB_query.word_tagging import Tagger
import KB_query.question_temp
from KB_query.question_temp import rules



class Question2Sparql:
    def __init__(self, dict_paths):

        self.tw = Tagger(dict_paths)
        self.rules = rules

    def get_sparql(self, question):
        """
        进行语义解析，找到匹配的模板，返回对应的SPARQL查询语句
        :param question:
        :return:
        """
        word_objects = self.tw.get_word_objects(question)
        queries_dict = dict()

        for rule in self.rules:
            query, num = rule.apply(word_objects)

            if query is not None:
                queries_dict[num] = query
        # print("匹配到的模板的个数"+str(len(queries_dict)))
        if len(queries_dict) == 0:
            return None
        elif len(queries_dict) == 1:
            # print(list(queries_dict.values())[0])   #打印查询语句
            return list(queries_dict.values())[0]
        else:
            # TODO 匹配多个语句，以匹配关键词最多的句子作为返回结果
            sorted_dict = sorted(queries_dict.items(), key=lambda item: item[0], reverse=True)
            # return list(sorted_dict.values())[0]
            # print(sorted_dict[0][1])    #将查询语句打印出来
            return sorted_dict[0][1]

if __name__ == '__main__':
    q2s = Question2Sparql(['./external_dict/features.txt','./external_dict/solutions.txt'])
    #question = '喉插管损伤有什么症状？'
    #question = '马来酸罗格列酮片的批准文号是什么?'
    #question = '怎么预防不完全性肠梗阻?'
    question = '吊机吊装操作红色报警灯故障吊装过程'
    my_query = q2s.get_sparql(question.encode('utf-8'))
    print(my_query)
