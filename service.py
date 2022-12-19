from flask import Flask
from flask import request
from flask import jsonify
from predictions import classify

app = Flask("classify")

@app.route("/classify", methods=["POST"])
def classified():

    image_file = request.files['img']
    # image_url = request.get_json()
    output = classify(image_file)
    result = {"prediction": output}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)
