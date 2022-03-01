import spacy
from spacy.matcher import Matcher

# nlp = spacy.load("xx_ent_wiki_sm")
nlp = spacy.load("./output/model-last")
# doc = nlp("Today I was in Lviv ant tomorrow I will go to Kyiv")
doc = nlp("Поїду з Яворова (левада, черкаси) завтра до Львова в 12:00, 0970067238")
# doc = nlp("Підвезу до Львова з Яворова о 7:00 (ц.Анни)")

for ent in doc.ents:
  print(
    "{} -> {}".format(ent.text, ent.label_)
  )
