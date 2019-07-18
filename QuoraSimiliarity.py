import numpy as np #untuk merepresentasikan array
import pandas as pd #untuk meload file csv
import re #regular expressin / regex
import nltk 
from string import punctuation #membaca tanda baca di string
from nltk.stem import SnowballStemmer #stemmer untuk stemming dalam kalimat
from nltk.corpus import stopwords #membagi kalimat menjadi per kata
from sklearn.model_selection import train_test_split #membagi array / matriks menjadi pelatihan acak dan bagian tes
from sklearn.feature_extraction.text import CountVectorizer #convert collection text menjadi jumlah token dari matriks
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#load file csv dan menampilkan 15 baris pertama
df = pd.read_csv('train.csv')
print(df.head(15))
print("\n")

#menampilakan nilai yang kosong pada kolom
print(df.isnull().sum())
print("\n")

#merepresentasikan dimensi dari dataset
print(df.shape)
print("\n")
    
#menghapus value yang kosong, kolom id,qid, dan qid2
df.dropna(axis=0, inplace=True)
df.drop(['id','qid1','qid2'], axis=1, inplace=True)
print(df.shape)
print("\n")
print(df.isnull().sum())
print("\n")

#menampilkan beberapa kata yang mempunyai kemungkinan mirip antara question1 dan question2
a=0
for i in range(a,a+15):
    print(df.question1[i])
    print(df.question2[i])
    print()


SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}

def clean(text, stem_words=True):
    
    def pad_str(s):
        return ' '+s+' '
    
    if pd.isnull(text):
        return ''

    stops = set(stopwords.words("english"))
    # Membersihkan text dengan stemming word
    
    # Empty question
    
    if type(text) != str or text=='':
        return ''

    # Membersihkan text / Stemming Word
    text = re.sub("\'s", " ", text) 
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    # menghapus koma
    
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
    
    # tambahkan padding ke tanda baca dan karakter khusus
    
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
        
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) # mereplace non asci menjadi asci
    
    # indian dollar
    
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    
    # membersihkan text rules yang didapat dari url : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text) 
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
    
    # mereplace float number menjadi random number
    
    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)
  
    
    # menghapus padding dari text
    text = ''.join([c for c in text if c not in punctuation]).lower()
       # mengembalikan semua list kata yang telat di stemming
    return text

#melakukan cleaning text pada kolom question1 dan question2
df['question1'] = df['question1'].apply(clean)
df['question2'] = df['question2'].apply(clean)

#menampilkan text setelah di cleaning
a=0
for i in range(a,a+15):
    print(df.question1[i])
    print(df.question2[i])
    print()


X = df.loc[:, df.columns != 'is_duplicate']
y = df.loc[:, df.columns == 'is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

transformer = FeatureUnion([
                ('question1_bow', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['question1'], 
                                                  validate=False)),
                            ('bow', 
                              CountVectorizer())])),
                ('question2_bow', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['question2'], 
                                                  validate=False)),
                            ('bow', 
                              CountVectorizer())]))])

X_train_count = transformer.fit_transform(X_train)
X_train_count.shape

tfidf_trfm = TfidfTransformer(norm=None)
X_train_count_tfidf = tfidf_trfm.fit_transform(X_train_count)
X_test_count_tfidf = tfidf_trfm.transform(X_test_count)
y_pred = rf.predict(X_test_count_tfidf)
print(metrics.classification_report(y_test, y_pred))