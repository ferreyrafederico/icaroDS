import numpy as np
import pandas as pd
import spacy
import re

def fclean_text(text,nlp=spacy.load("en_core_web_sm")):
    text=re.sub(r"\n", "", text)
    clean_text = []
    for token in nlp(text):
        if not token.is_stop and not token.is_punct:
            clean_text.append(token.lemma_.lower())
    return " ".join(clean_text)