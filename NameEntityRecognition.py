
from flair.models import SequenceTagger
from flair.data import Sentence
import flair
from pathlib import Path
import re

flair.cache_root = Path("D:\models\.flair")
print(f"Flair cache root directory: {flair.cache_root}")


class NameEntityRecognitionModel:

    def __init__(self, model_name="PooryaPiroozfar/Flair-Persian-NER"):

        self.tagger = SequenceTagger.load(model_name)

    def getEntities(self, text):

        sentence = Sentence(text)
        self.tagger.predict(sentence)

        entities = {}
        entity_texts = []

        for span in sentence.get_spans("ner"):
            ent_type = span.get_label("ner").value
            if ent_type not in entities:
                entities[ent_type] = []
            entities[ent_type].append(span.text)
            entity_texts.append(re.escape(span.text))  
            print(f"Entity: {span.text}, Type: {ent_type}, Confidence: {span.score}")

   
        if entity_texts:

            pattern = "|".join(entity_texts)
            unlabelled = [s.strip() for s in re.split(pattern, text) if s.strip()]
        else:

            unlabelled = [text.strip()]

        return {
            "entities": entities,
            "unlabelled": unlabelled
        }