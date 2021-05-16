from flask import Flask, render_template, request
from tensorflow.keras.models import  load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='templates')

saved_covid_lstm = load_model('bidirectiona-lstm-model.h5')
max_features=33000
tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
result=""
lstm_prediction=None

stop_words = set(stopwords.words('english'))
to_remove = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
stop_words.update(to_remove)

"""def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub('\[[^]]*\]', '', text)
    text = (" ").join([word for word in text.split() if not word in stop_words])
    text = "".join([char for char in text if not char in to_remove])
    return text"""


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    title = request.form['Name']
    text = request.form['Message']
    author = request.form['Subject']
    total = title + '' + text + '' + author
    news = str(total)
    news = re.sub(r'http\S+', '', news)
    news = re.sub('\[[^]]*\]', '', news)
    news = (" ").join([word for word in news.split() if not word in stop_words])
    news = "".join([char for char in news if not char in to_remove])
    """news = news.apply(clean_text)"""
    tokenizer.fit_on_texts(texts = news)
    test_text = tokenizer.texts_to_sequences(texts = [news])
    test_text = pad_sequences(sequences = test_text, maxlen = max_features, padding = 'pre')
    print(test_text) 
    lstm_prediction = saved_covid_lstm.predict_classes(test_text)
    if lstm_prediction == 1 :
        result = "News is fake"
    else:
        result = "News is true"  
    
    """for i in range(len(lstm_prediction)):
        if lstm_prediction[i].item() >= 0.5:
            result.append(1)
        else:
            result.append(0)"""
    """if pred >= 0.5:
        result.append(1)
    else:
        result.append(0)"""   
    return render_template('index.html', text_data=result)

@app.route('/services')
def services():
    return render_template('services.html')

app.config['TEMPLATES_AUTO_RELOAD']=True
app.run(port=6060, debug=True)