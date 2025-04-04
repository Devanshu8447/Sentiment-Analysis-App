from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import tensorflow as tf
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences


# Load the trained model and tokenizer
model_path = "model.pk1"
tokenizer_path = "tokenizer (1).pkl"

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define input length
max_length = 200

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    review = request.form['review']
    
    # Preprocess the input
    review_seq = tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_seq, maxlen=max_length, padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(review_padded)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    
    return render_template('index.html', prediction_text=f'Predicted Sentiment: {sentiment}')

if __name__ == "__main__":
    app.run(debug=True)

