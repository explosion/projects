"""Extract demo set from Wiki dumps."""
from utils import read_filter_terms
from wiki import wiki_dump_api

if __name__ == '__main__':
    wiki_dump_api.extract_demo_dump(read_filter_terms())
