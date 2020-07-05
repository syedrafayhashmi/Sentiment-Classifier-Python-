import spacy

import pandas as pd

import csv

snlp = spacy.load('en_core_web_md')

entities = ['GPE', 'LOC', 'PERSON', 'ORG', 'DATE']

# creates pandas dataframe with your specified input file, using the first row as a header
df = pd.read_csv(r"CrimeNews.csv", encoding='latin-1')
# creates a new column, ner_text, with entities extracted from a column titled 'text'
df['ner_text'] = df['Summary'].astype(str).apply(lambda x: list(snlp(x).ents))

# this loop labels entities which are bulit in spacy
for i in range(len(entities)):
    df[entities[i]] = df['Summary'].astype(str).apply(
        lambda x: [t.text for t in snlp(x).ents if t.label_ == entities[i]])


# below lines call the Crime_Entity_Model and labels Crime Entities for the crime news
nlp = spacy.load('Crime_Entity_Model')

df['Type of Crime'] = df['Summary'].astype(str).apply(lambda x: list({t.label_ for t in nlp(x).ents}))

# saving to csv file and NER.csv file contains crime news with entities labeled after
# excuting the above code
 df.to_csv('NER.csv')
