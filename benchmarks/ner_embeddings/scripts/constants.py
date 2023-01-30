# A mapping of datasets and their vectors
DATASET_VECTORS = {
    "archaeo": {"spacy": "nl_core_news_lg", "fasttext": "fasttext-nl", "lang": "nl"},
    "anem": {"spacy": "en_core_web_lg", "fasttext": "fasttext-en", "lang": "en"},
    "es-conll": {"spacy": "es_core_news_lg", "fasttext": "fasttext-es", "lang": "es"},
    "nl-conll": {"spacy": "nl_core_news_lg", "fasttext": "fasttext-nl", "lang": "nl"},
    "wnut17": {"spacy": "en_core_web_lg", "fasttext": "fasttext-en", "lang": "en"},
    "finer": {"spacy": "fi_core_news_lg", "fasttext": "fasttext-fi", "lang": "fi"},
    "ontonotes": {"spacy": "en_core_web_lg", "fasttext": "fasttext-en", "lang": "en"},
}

CONFIGS = ["multiembed", "multihashembed"]
