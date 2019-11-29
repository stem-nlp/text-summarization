# 非监督文本自动摘要模型的构建
import random
import textsummary

class AutoSummary():
    def __init__(self, title, body,percent):
        self.title = title
        self.body = body
        self.percent=percent
    def train(self):
        pass
    def getResult(self):
        summary=textsummary.get_summary(self.title,self.body,self.percent)
        return summary
       # return "测试结果。测试结果。测试结果。测试结果。"
    def getDetail(self):
        res = []
        for i in range(4):
            res.append({
                "content":"测试结果",
                "score": "%.2f" % random.random(),
                "reserved": True
            })
        return res