import string
import collections
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize


def tokenizer(words: str) -> list:
    stops = set(stopwords.words('english') + list(string.punctuation))
    words = [word for word in wordpunct_tokenize(words.lower()) if word not in stops]
    return words


def remove_dup(words: list) -> list:
    """ Remove duplicated words, first remove front ones (for query only). """
    words_unique = []
    for word in words[::-1]:
        if word not in words_unique:
            words_unique.append(word)
    words_unique.reverse()
    return words_unique
    

def filter_words(document: list, min_num: int) -> list:
    """ Filter words in documents less than min_num. """
    cnt = collections.Counter()
    for sentence in document:
        cnt.update(sentence)

    s = set(word for word in cnt if cnt[word] < min_num)
    document = [[word for word in sentence if word not in s] for sentence in document]
    return document
