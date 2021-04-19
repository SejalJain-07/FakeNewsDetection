from flask import Flask, render_template, request
from keras.models import  load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

saved_covid_lstm = load_model('bidirectiona-lstm-model.h5')
max_features=33000
tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    text = request.form['Message']
    news = str(text)
    tokenizer.fit_on_texts(texts = news)
    test_text = tokenizer.texts_to_sequences(texts = [news])
    test_text = pad_sequences(sequences = test_text, maxlen = max_features, padding = 'pre')
    lstm_prediction = saved_covid_lstm.predict_classes(test_text)
    if lstm_prediction == 1:
        result = "News is true"
    else:
        result = "News is false"
    return render_template('index.html', text_data=result)


app.run(port=6060, debug=True)