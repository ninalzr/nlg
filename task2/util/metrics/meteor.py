from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score


def meteor_score(y_true, y_pred, lookup, preprocess = str.lower, stemmer = PorterStemmer(), wordnet=wordnet, alpha=0.9,
                 beta=3, gamma=0.5):
    total_score = 0

    for y_true_seq, y_pred_seq in zip(y_true, y_pred):
        #references = [lookup.convert_ids_to_tokens(index) for index in y_true_seq]
        #hypothesis = [lookup.convert_ids_to_tokens(index) for index in y_pred_seq]
        #references = [" ".join(references)]
        #hypothesis = " ".join(hypothesis)
        references = [lookup.decode(y_true_seq, skip_bos_eos_tokens = True)]
        hypothesis = lookup.decode(y_pred_seq, skip_bos_eos_tokens = True)

        total_score += nltk_meteor_score(references, hypothesis, preprocess, stemmer, wordnet, alpha, beta, gamma)

    return total_score / len(y_true)