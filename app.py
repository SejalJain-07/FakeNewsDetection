from flask import Flask, render_template, request
from tensorflow.keras.models import  load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='templates')

saved_covid_blstm = load_model('bidirectional-lstm-model.h5')
max_features=33000
tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
result=""
blstm_prediction=None

stop_words = set(stopwords.words('english'))
to_remove = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
stop_words.update(to_remove)


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
    blstm_prediction = saved_covid_blstm.predict_classes(test_text)
    if blstm_prediction == 1 :
        result = "News is fake"
    else:
        result = "News is true"   
    return render_template('result.html', text_data=result)

@app.route('/services')
def services():
    return render_template('services.html')

app.config['TEMPLATES_AUTO_RELOAD']=True
app.run(port=6060, debug=True)