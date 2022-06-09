from flask import Flask, render_template, request, jsonify
from api_service import *

if os.path.exists("Scraped reviews.xlsx"):
    df = pd.read_excel(r'Scraped reviews.xlsx')
    if os.path.isdir("cnn_model"):
        model = load_model("cnn_model")
        with open('tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        application = Flask(__name__)


@application.route('/', methods=['GET'])
def home():
    return render_template("Homepage.html")


@application.route('/classify_review', methods=['POST'])
def classify_review():
    review = request.form['review-area']
    if len(review) == 0:
        return jsonify({'error': 'Please input a review'})
    rating = predict_rating(review, tokenizer, model) + 1
    return jsonify({'rating': int(rating)})

@application.route('/generate_review', methods=['GET'])
def generate_review():
    review = generate_random_review(df)
    return jsonify({'review': review})


if __name__ == "__main__": application.run()
