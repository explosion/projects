try:
    from spacy.kb import InMemoryLookupKB as KnowledgeBase
except ImportError:
    from spacy.kb import KnowledgeBase as KnowledgeBase
