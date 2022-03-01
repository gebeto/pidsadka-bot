import spacy
from spacy.matcher import Matcher

# nlp = spacy.load("xx_ent_wiki_sm")
nlp = spacy.load("./output/model-last")
# doc = nlp("Today I was in Lviv ant tomorrow I will go to Kyiv")
doc = nlp("Yesterday I steal F22")

for ent in doc.ents:
  print(
    "{} -> {}".format(ent.text, ent.label_)
  )
