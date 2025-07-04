import flask
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Custom objects to handle model loading compatibility
custom_objects = {
    'InputLayer': tf.keras.layers.InputLayer,
    'Embedding': tf.keras.layers.Embedding,
    'LSTM': tf.keras.layers.LSTM,
    'Dense': tf.keras.layers.Dense
}

# Load model with custom_objects to handle compatibility
model = load_model('next_word_model.h5', custom_objects=custom_objects, compile=False)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1] + 1

app = flask.Flask(__name__)

@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = flask.request.json.get('text', '')
    token_text = tokenizer.texts_to_sequences([text])[0]
    
    if not token_text:  
        return flask.jsonify({'predictions': []})

    padded_token_text = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')

    predictions = model.predict(padded_token_text, verbose=0)[0]

    num_options = 5
    top_indices = np.argsort(predictions)[-num_options:][::-1]



    predicted_words = [tokenizer.index_word.get(index, '') for index in top_indices if tokenizer.index_word.get(index, '')]

    return flask.jsonify({'predictions': predicted_words})
if __name__ == '__main__':
    app.run()

