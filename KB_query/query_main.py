# encoding=utf-8

"""

@desc:main函数，整合整个处理流程。

"""
import jena_sparql_endpoint
import question2sparql
#from KB_query import jena_sparql_endpoint
#from KB_query import question2sparql

def query_function(question):
    # TODO 连接Fuseki服务器。
    fuseki = jena_sparql_endpoint.JenaFuseki()
    # TODO 初始化自然语言到SPARQL查询的模块，参数是外部词典列表。
    q2s = question2sparql.Question2Sparql(
        ['./external_dict/f1.txt', './external_dict/f2.txt', './external_dict/f3.txt', './external_dict/f4.txt',
         './external_dict/f5.txt'])

    while True:
        question = question
        # print(question.encode('utf-8'))
        # isinstance(question.encode('utf-8'))
        my_query = q2s.get_sparql(question.encode('utf-8'))
        # print(my_query)
        if my_query is not None:
            result = fuseki.get_sparql_result(my_query)
            value = fuseki.get_sparql_result_value(result)

            # TODO 查询结果为空，根据OWA，回答“不知道”
            if len(value) == 0:
                return '胖子哥也不是扁鹊啊，知识库中并没有该问题的答案！！！'
            elif len(value) == 1:
                print(len(value[0]))
                if len(value[0]) != 1:
                    return value[0]
                else:
                    return value[0]
            else:
                output = ''
                for v in value:
                    output += v + u'、'
                return output

        else:
            # TODO 自然语言问题无法匹配到已有的正则模板上，回答“无法理解”
            return '胖子哥也不是扁鹊啊，无法理解你的问题！！！'

            # print('#' * 100)

if __name__ == '__main__':
    # TODO 连接Fuseki服务器。
    fuseki = jena_sparql_endpoint.JenaFuseki()
    # TODO 初始化自然语言到SPARQL查询的模块，参数是外部词典列表。
    q2s = question2sparql.Question2Sparql(['./external_dict/f1.txt', './external_dict/f2.txt','./external_dict/f3.txt','./external_dict/f4.txt','./external_dict/f5.txt'])

    while True:
        question = input()
        # my_query = q2s.get_sparql(question.decode('utf-8'))
        my_query = q2s.get_sparql(question.encode('utf-8'))
        print(my_query)
        if my_query is not None:
            result = fuseki.get_sparql_result(my_query)
            value = fuseki.get_sparql_result_value(result)

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
                elif len(value) == 1:
                    print(value[0])
                else:
                    output = ''
                    for v in value:
                        output += v + u'、'
                    print(output[0:-1])

        else:
            # TODO 自然语言问题无法匹配到已有的正则模板上，回答“无法理解”
            print('I can\'t understand. :(')

        print('#' * 100)
