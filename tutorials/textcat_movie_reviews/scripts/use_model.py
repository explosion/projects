import operator
import spacy

from flask import Flask
from flask import request, jsonify

app = Flask(__name__)


class TextClassificationModel(object):
    def __init__(self, path_to_model):
        # Load the best trained Text Classification model
        self.nlp_model = spacy.load(path_to_model)

    def predict(self, sentence):
        # Transform the input sentence to SpaCy's Doc with predicted categories
        doc = self.nlp_model(sentence)
        return doc.cats


nlp_model = TextClassificationModel('./training/model-best/')

@app.route('/predict', methods=['GET'])
def predict_category_for_input_sentence():
    sentence = request.args.get('sentence')
    cats = nlp_model.predict(sentence=sentence)

    best_category = max(cats.items(), key=operator.itemgetter(1))[0]
    best_score = [value for key, value in cats.items() if key == best_category][0]

    return jsonify({'code': 200, 'category': best_category, 'score': best_score})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5000')