from flask import Flask, render_template, request, jsonify
from api_service import *

if os.path.exists("Scraped reviews.xlsx"):
    df = pd.read_excel(r'Scraped reviews.xlsx')
    if os.path.isdir("cnn_model"):
        model = load_model("cnn_model")
        with open('tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template("Homepage.html")


@app.route('/classify_review', methods=['POST'])
def classify_review():
    review = request.form['review-area']
    if len(review) == 0:
        return jsonify({'error': 'Please input a review'})
    rating = predict_rating(review, tokenizer, model) + 1
    return jsonify({'rating': int(rating)})

@app.route('/generate_review', methods=['GET'])
def generate_review():
    review = generate_random_review(df)
    return jsonify({'review': review})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
