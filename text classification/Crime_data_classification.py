import pandas as pd
import numpy as np
import csv
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

np.random.seed(500)

# Model Training
whole_News = pd.read_csv(r"F:/FYP/text classification/Dawn8years.csv", encoding='latin-1')

labelled_News = pd.read_csv(r"F:/FYP/text classification/labelledData_200.csv", encoding='latin-1')

# Step - a : Remove blank rows if any.
labelled_News['Summary'].dropna(inplace=True)
whole_News['Summary'].dropna(inplace=True)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
labelled_News['Summary'] = [entry.lower() for entry in labelled_News['Summary']]
whole_News['Summary'] = [entry.lower() for entry in whole_News['Summary']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
labelled_News['Summary'] = [word_tokenize(entry) for entry in labelled_News['Summary']]
whole_News['Summary'] = [word_tokenize(entry) for entry in whole_News['Summary']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# for labelled News
for index, entry in enumerate(labelled_News['Summary']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    labelled_News.loc[index, 'text_final'] = str(Final_words)

# for unlabelled News

for index, entry in enumerate(whole_News['Summary']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    whole_News.loc[index, 'summary_final'] = str(Final_words)

# spliting labelled News for model training

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(labelled_News['text_final'],
                                                                    labelled_News['label'], test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(labelled_News['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

whole_News_Tfidf = Tfidf_vect.transform(whole_News['summary_final'])

# print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)

# Classifier-Algorithm - Naive Bayes
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)

# Using the Model To Find labels for the whole Document
whole_News_predictions = SVM.predict(whole_News_Tfidf)
print("whole News labels => ", whole_News_predictions)

# Now spliting Crime News from the Document on the bases of labels provided by SVM model

whole_News_Document = "F:/FYP/text classification/Dawn8years.csv"
fields = []
rows = []
i = 0  # for iteration through labels

with open(whole_News_Document, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    # extracting field names through first row

    # extracting each data row one by one
    for row in csvreader:
        if whole_News_predictions[i] == 0:
            rows.append(row)

        i = i + 1

Crimefile = "F:/FYP/text classification/CrimeNews.csv"
with open(Crimefile, 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)
