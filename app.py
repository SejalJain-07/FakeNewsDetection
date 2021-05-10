from flask import Flask, render_template, request
from tensorflow.keras.models import  load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys

app = Flask(__name__, template_folder='templates')

saved_covid_lstm = load_model('bidirectiona-lstm-model.h5')
max_features=33000
tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
result=""
lstm_prediction=None
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
    print(test_text) 
    lstm_prediction = saved_covid_lstm.predict_classes(test_text)
    """if lstm_prediction == 1 :
        result = "News is fake"
    else:
        result = "News is true" """ 
    
    """pred=sum(lstm_prediction)"""
    
    """pred=pred/2"""
    
    """for i in range(len(lstm_prediction)):
        if lstm_prediction[i].item() >= 0.5:
            result.append(1)
        else:
            result.append(0)"""
    """if pred >= 0.5:
        result.append(1)
    else:
        result.append(0)"""
        
    return render_template('index.html', text_data=lstm_prediction)

@app.route('/services')
def services():
    return render_template('services.html')

app.config['TEMPLATES_AUTO_RELOAD']=True
app.run(port=6060, debug=True)