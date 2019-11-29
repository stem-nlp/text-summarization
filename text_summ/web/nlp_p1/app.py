from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin
import json

from model import AutoSummary

app = Flask(__name__)

@app.route("/", methods=["GET"])
def main():
    return render_template("main.html")

@app.route('/api/model', methods=["POST"])
@cross_origin()
def model():
    try:
        input_title = request.form.get('title','')
        input_body = request.form.get('body','')
        input_percent = request.form.get('percent', '')

        model = AutoSummary(input_title, input_body,input_percent)
        result = {
            "code": 0,
            "data":{
                "content": model.getResult(),
                "detail": model.getDetail()
            }
        }
        return jsonify(result)
    except Exception as e:
        print(str(e.__traceback__))
        return jsonify({"code":-1, "data":str(e)})


if __name__ == '__main__':
    app.run()

application = app