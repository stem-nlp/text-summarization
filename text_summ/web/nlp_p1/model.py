# 非监督文本自动摘要模型的构建
import random

class AutoSummary():
    def __init__(self, title, body):
        self.title = title
        self.body = body
    def train(self):
        pass
    def getResult(self):
        return "测试结果。测试结果。测试结果。测试结果。"
    def getDetail(self):
        res = []
        for i in range(4):
            res.append({
                "content":"测试结果",
                "score": "%.2f" % random.random(),
                "reserved": True
            })
        return res