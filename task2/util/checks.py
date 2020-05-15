import re, os, sys
from common.nlg import read_nlg_file
from collections import Counter



"""
def repetition_check(sentence):
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', sentence)



output = repetition_check(text1)
print(output)

MEI = "Management Overview"
path_output_file = os.path.join("..", "data", "ready", MEI.replace(" ","_")+"_outputs.txt")

def words(text):
    return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open(path_output_file).read()))

def P(word, N=sum(WORDS.values())):
    return WORDS[word] / N

def correction(word):
    return max(candidates(word), key=P)

def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))




text = text.lower()
punctuation = ['.', ',', "'s"]

for p in punctuation:
    text = text.replace(p, "")
tokens = list(set(text.split()))
wrongs = []
for token in tokens:
    if WORDS[token]:
        continue
    else:
        wrongs.append(token)
print(wrongs)

word = 'withile'
new_word = correction(word)
print(new_word)


import pandas as pd
v2 = pd.read_excel('/work/nlg/outputs.xlsx')
paragraphs = v2.values

wrongs = []
for idx, paragraph in enumerate(paragraphs):
    for par in paragraph:
        text = par.lower()
        punctuation = ['.', ',', "'s", ";"]
        for p in punctuation:
            text = text.replace(p, "")
        tokens = list(set(text.split()))
        for token in tokens:
            if WORDS[token]:
                continue
            else:
                wrongs.append((idx, token))

correct = []
for w in wrongs:
    correct.append((w[0], correction(w[1])))
print(wrongs)
print(correct)
"""

class spelling:

    def __init__(self, path_output_file):
        self.path_output_file = path_output_file
    def words(self, text):
        return re.findall(r'\w+', text.lower())
    def repetition_check(self, sentence):
        return re.sub(r'\b(\w+)( \1\b)+', r'\1', sentence)
    def check_spelling(self, text):
        WORDS = Counter(self.words(open(path_output_file).read()))
        wrongs = []
        text = text.lower()
        punctuation = ['.', ',', "'s", ";"]
        for p in punctuation:
            text = text.replace(p, "")
        tokens = list(set(text.split()))
        for token in tokens:
            if WORDS[token]:
                continue
            else:
                wrongs.append(token)
        if len(wrongs) > 0:
            return True
        False


if __name__ == "__main__":
    MEI = "Management Overview"
    path_output_file = os.path.join("..", "data", "ready", MEI.replace(" ", "_") + "_outputs.txt")
    checks = spelling(path_output_file)
    clean = checks.repetition_check(text1)
    if checks.check_spelling(clean):
        print("Company has errors")
    else:
        print("Company is clean")


text = "The company's ESG reporting is assessed as very strong and a management committee oversees ESG issues. " \
       "Similarly, the environmental policy and the standards are very strong. Concertrasting this, " \
       "the standards on social supply chain issues are also address child labour as well as forced labour. " \
       "In contrast, the company has set up a strong whistleblower programme."

text1 = "The company lacks in ESG disclosure as it has not published relevant reports in recent years and a board " \
        "committee is in place to oversee governance issues only. Furthermore, evidence indicates the company " \
        "lacks an environmental policy and has has strong social supply chain standards. For instance, the standards " \
        "address child and forced labour. In contrast, the company has set up a strong whistleblower programme."









