""" Functionalities for processing Wikipedia dump.
Modified from https://github.com/explosion/projects/blob/master/nel-wikipedia/wikipedia_processor.py.
"""

import re
import bz2
import sqlite3

from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Set, Any

import tqdm

from wiki.namespaces import (
    WP_META_NAMESPACE,
    WP_FILE_NAMESPACE,
    WP_CATEGORY_NAMESPACE,
)

"""
Process a Wikipedia dump to calculate entity_title frequencies and prior probabilities in combination with certain mentions.
Write these results to file for downstream KB and training data generation.

Process Wikipedia interlinks to generate a training dataset for the EL algorithm.
"""

map_alias_to_link = dict()

title_regex = re.compile(r"(?<=<title>).*(?=</title>)")
id_regex = re.compile(r"(?<=<id>)\d*(?=</id>)")
text_tag_regex = re.compile(r"(?<=<text).*?(?=>)")
text_regex = re.compile(r"(?<=<text>).*(?=</text)")
info_regex = re.compile(r"{[^{]*?}")
html_regex = re.compile(r"&lt;!--[^-]*--&gt;")
ref_regex = re.compile(r"&lt;ref.*?&gt;")  # non-greedy
ref_2_regex = re.compile(r"&lt;/ref.*?&gt;")  # non-greedy

# find the links
link_regex = re.compile(r"\[\[[^\[\]]*\]\]")

# match on interwiki links, e.g. `en:` or `:fr:`
ns_regex = r":?" + "[a-z][a-z]" + ":"
# match on Namespace: optionally preceded by a :
for ns in WP_META_NAMESPACE:
    ns_regex += "|" + ":?" + ns + ":"
ns_regex = re.compile(ns_regex, re.IGNORECASE)

files = r""
for f in WP_FILE_NAMESPACE:
    files += "\[\[" + f + ":[^[\]]+]]" + "|"
files = files[0 : len(files) - 1]
file_regex = re.compile(files)

cats = r""
for c in WP_CATEGORY_NAMESPACE:
    cats += "\[\[" + c + ":[^\[]*]]" + "|"
cats = cats[0 : len(cats) - 1]
category_regex = re.compile(cats)


def read_prior_probs(
    wikidata_input_path: Union[str, Path],
    db_conn: sqlite3.Connection,
    batch_size: int = 5000,
    limit: Optional[int] = None,
) -> None:
    """
    Read the XML wikipedia data and parse out intra-wiki links to estimate prior probabilities.
    The full file takes about 2-3h to parse 1100M lines. Writes prior information to DB.
    It works relatively fast because it runs line by line, irrelevant of which article the intrawiki is from.
    wikidata_input_path (Union[str, Path]): Path to Wikipedia dump.
    batch_size (int): DB batch size.
    db_conn (sqlite3.Connection): Database connection.
    n_article_limit (Optional[int]): Number of articles/entities to process.
    """

    read_id = False
    current_article_id = None
    entity_title_to_id = {
        row["name"]: row["id"]
        for row in db_conn.execute("SELECT name, id FROM entities")
    }

    def write_to_db(_aliases_for_entities) -> None:
        """Writes record triples to DB.
        __aliases_for_entities (): alias-entity-frequency triples.
        """
        db_conn.cursor().executemany(
            "INSERT INTO aliases_for_entities (alias, entity_id, count) VALUES (?, ?, ?)",
            _aliases_for_entities,
        )
        db_conn.commit()

    with bz2.open(wikidata_input_path, mode="rb") as file:
        pbar_params = {"total": limit} if limit else {}
        with tqdm.tqdm(
            desc="Parsing alias-entity prior probabilities", **pbar_params
        ) as pbar:
            line = file.readline()
            while line and (not limit or pbar.n < limit):
                clean_line = line.strip().decode("utf-8")

                # we attempt at reading the article's ID (but not the revision or contributor ID)
                if "<revision>" in clean_line or "<contributor>" in clean_line:
                    read_id = False
                if "<page>" in clean_line:
                    read_id = True

                if read_id:
                    ids = id_regex.search(clean_line)
                    if ids:
                        current_article_id = ids[0]

                # only processing prior probabilities from true training (non-dev) articles
                if not is_dev(current_article_id):
                    aliases, entities, normalizations = _get_wp_links(clean_line)
                    for alias, entity_title, norm in zip(
                        aliases, entities, normalizations
                    ):
                        _store_alias(
                            alias,
                            entity_title,
                            normalize_alias=norm,
                            normalize_entity=True,
                        )

                line = file.readline()
                pbar.update(1)

    # write all aliases and their entities and count occurrences to file
    # len(map_alias_to_link) == 1323974520
    with tqdm.tqdm(
        desc="Persisting prior probabilities", total=len(map_alias_to_link)
    ) as pbar:
        aliases_for_entities: List[Tuple[str, str, int]] = []
        for alias, alias_dict in map_alias_to_link.items():
            for entity_title, count in alias_dict.items():
                if entity_title in entity_title_to_id:
                    aliases_for_entities.append(
                        (alias, entity_title_to_id[entity_title], count)
                    )
            if pbar.n % batch_size == 0:
                write_to_db(aliases_for_entities)
                aliases_for_entities = []

            pbar.update(1)

        if pbar.n % batch_size != 0:
            write_to_db(aliases_for_entities)


def _store_alias(
    alias: str,
    entity_title: str,
    normalize_alias: bool = False,
    normalize_entity: bool = True,
) -> None:
    """Stores (normalized) alias for (normalized) entity_title ID in mapping dictionaries.
    alias (str): Alias text.
    entity_title (str): Entity title.
    normalize_alias (bool): Whether to normalize the alias text, i.e. remove anchors.
    normalize_entity (bool): Whether to normalize the entity title.
    """
    alias = alias.strip()
    entity_title = entity_title.strip()

    # remove everything after # as this is not part of the title but refers to a specific paragraph
    if normalize_entity:
        # wikipedia titles are always capitalized
        entity_title = _capitalize_first(entity_title.split("#")[0])
    if normalize_alias:
        alias = alias.split("#")[0]

    if alias and entity_title:
        alias_dict = map_alias_to_link.get(alias, dict())
        entity_count = alias_dict.get(entity_title, 0)
        alias_dict[entity_title] = entity_count + 1
        map_alias_to_link[alias] = alias_dict


def _get_wp_links(text: str) -> Tuple[List[str], List[str], List[bool]]:
    """Retrieve interwiki links from text.
    text (str): Text to parse.
    RETURNS (Tuple[List[str], List[str], List[bool]]): List of aliases, entity titles, and whether normalization they
        were normalized.
    """
    aliases: List[str] = []
    entities: List[str] = []
    normalizations: List[bool] = []

    matches = link_regex.findall(text)
    for match in matches:
        match = match[2:][:-2].replace("_", " ").strip()

        if ns_regex.match(match):
            pass  # ignore the entity_title if it points to a "meta" page

        # this is a simple [[link]], with the alias the same as the mention
        elif "|" not in match:
            aliases.append(match)
            entities.append(match)
            normalizations.append(True)

        # in wiki format, the link is written as [[entity_title|alias]]
        else:
            splits = match.split("|")
            entity = splits[0].strip()
            alias = splits[1].strip()
            # specific wiki format  [[alias (specification)|]]
            if len(alias) == 0 and "(" in entity:
                alias = entity.split("(")[0]
                aliases.append(alias)
                entities.append(entity)
                normalizations.append(False)
            else:
                aliases.append(alias)
                entities.append(entity)
                normalizations.append(False)

    return aliases, entities, normalizations


def _capitalize_first(text: str) -> Optional[str]:
    """Capitalize first character.
    text (str): String in which to capitalize first character.
    RETURN (Optional[str]): Text with first character capitalized.
    """
    if not text:
        return None
    result = text[0].capitalize()
    if len(result) > 0:
        result += text[1:]
    return result


def read_texts(
    wikipedia_input_path: Union[str, Path],
    db_conn: sqlite3.Connection,
    batch_size: int = 5000,
    limit: Optional[int] = None,
    n_char_limit: int = 1000,
) -> None:
    """
    Read the XML Wikipedia data to parse out clean article texts. Texts are stored in file.
    wikipedia_input_path (Union[str, Path]): Path to Wikipedia dump.
    db_conn (sqlite3.Connection): DB connection.
    limit (Optional[int]): Max. number of articles to process. If None, all are processed.
    n_char_limit (Optional[int]): Max. number of characters to process per article.
    """
    read_ids: Set[str] = set()
    entity_title_to_id = {
        row["name"]: row["id"]
        for row in db_conn.execute("SELECT name, id FROM entities")
    }
    records: List[Tuple[str, str, str]] = []

    def write_to_db(_records: List[Tuple[str, str, str]]) -> None:
        """Writes records to list.
        _records (List[Tuple[str, str, str]]): Article triples with entity ID, title and text.
        """
        db_conn.cursor().executemany(
            "INSERT INTO articles (entity_id, title, text) VALUES (?, ?, ?)", records
        )
        db_conn.commit()

    with bz2.open(wikipedia_input_path, mode="rb") as file:
        pbar_params = {"total": limit} if limit else {}
        with tqdm.tqdm(desc="Parsing article texts", **pbar_params) as pbar:
            article_text = ""
            article_title: Optional[str] = None
            article_id: Optional[str] = None
            reading_text = False
            reading_revision = False

            for line in file:
                if limit and pbar.n >= limit:
                    break

                clean_line = line.strip().decode("utf-8")

                if clean_line == "<revision>":
                    reading_revision = True
                elif clean_line == "</revision>":
                    reading_revision = False

                # Start reading new page
                if clean_line == "<page>":
                    article_text = ""
                    article_title = None
                    article_id = None

                # finished reading this page
                elif clean_line == "</page>":
                    if article_id:
                        clean_text, entities = _process_wp_text(
                            article_title, article_text, entity_title_to_id
                        )
                        if clean_text is not None and entities is not None:
                            if article_title in entity_title_to_id:
                                records.append(
                                    (
                                        entity_title_to_id[article_title],
                                        article_title,
                                        " ".join(
                                            clean_text[:n_char_limit].split(" ")[:-1]
                                        ),
                                    )
                                )
                            pbar.update(1)

                            if pbar.n % batch_size == 0:
                                write_to_db(records)
                                records = []

                    article_text = ""
                    article_title = None
                    article_id = None
                    reading_text = False
                    reading_revision = False

                # start reading text within a page
                if "<text" in clean_line:
                    reading_text = True

                if reading_text:
                    article_text += " " + clean_line

                # stop reading text within a page (we assume a new page doesn't start on the same line)
                if "</text" in clean_line:
                    reading_text = False

                # read the ID of this article (outside the revision portion of the document)
                if not reading_revision:
                    ids = id_regex.search(clean_line)
                    if ids:
                        article_id = ids[0]
                        if article_id in read_ids:
                            # This should never happen ...
                            print("Found duplicate article ID", article_id, clean_line)
                        read_ids.add(article_id)

                # read the title of this article (outside the revision portion of the document)
                if not reading_revision:
                    titles = title_regex.search(clean_line)
                    if titles:
                        article_title = titles[0].strip()

    if pbar.n % batch_size != 0:
        write_to_db(records)


def _process_wp_text(
    article_title: str, article_text: str, entity_title_to_id: Dict[str, str]
) -> Tuple[Optional[str], Optional[List[Tuple[str, Any, int, int]]]]:
    """Process article text.
    article_title (str): Article title.
    article_text (str): Article text.
    entity_title_to_id (Dict[str, str]): Map for entity/article titles to their IDs.
    RETURNS (Tuple[Optional[str], Optional[List[Tuple[str, Any, int, int]]]]): Cleaned text and list of entities in
        article text.
    """
    # ignore meta Wikipedia pages
    if ns_regex.match(article_title):
        return None, None

    # remove the text tags
    text_search = text_tag_regex.sub("", article_text)
    text_search = text_regex.search(text_search)
    if text_search is None:
        return None, None
    text = text_search.group(0)

    # stop processing if this is a redirect page
    if text.startswith("#REDIRECT"):
        return None, None

    # get the raw text without markup etc, keeping only interwiki links
    return _remove_links(_get_clean_wp_text(text), entity_title_to_id)


def _get_clean_wp_text(article_text: str) -> str:
    """Cleans article text.
    article_text (str): Text to clean.
    RETURNS (str): Cleaned text.
    """
    clean_text = article_text.strip()

    # remove bolding & italic markup
    clean_text = clean_text.replace("'''", "")
    clean_text = clean_text.replace("''", "")

    # remove nested {{info}} statements by removing the inner/smallest ones first and iterating
    try_again = True
    previous_length = len(clean_text)
    while try_again:
        clean_text = info_regex.sub(
            "", clean_text
        )  # non-greedy match excluding a nested {
        if len(clean_text) < previous_length:
            try_again = True
        else:
            try_again = False
        previous_length = len(clean_text)

    # remove HTML comments
    clean_text = html_regex.sub("", clean_text)

    # remove Category and File statements
    clean_text = category_regex.sub("", clean_text)
    clean_text = file_regex.sub("", clean_text)

    # remove multiple =
    while "==" in clean_text:
        clean_text = clean_text.replace("==", "=")

    clean_text = clean_text.replace(". =", ".")
    clean_text = clean_text.replace(" = ", ". ")
    clean_text = clean_text.replace("= ", ".")
    clean_text = clean_text.replace(" =", "")

    # remove refs (non-greedy match)
    clean_text = ref_regex.sub("", clean_text)
    clean_text = ref_2_regex.sub("", clean_text)

    # remove additional wikiformatting
    clean_text = re.sub(r"&lt;blockquote&gt;", "", clean_text)
    clean_text = re.sub(r"&lt;/blockquote&gt;", "", clean_text)

    # change special characters back to normal ones
    clean_text = clean_text.replace(r"&lt;", "<")
    clean_text = clean_text.replace(r"&gt;", ">")
    clean_text = clean_text.replace(r"&quot;", '"')
    clean_text = clean_text.replace(r"&amp;nbsp;", " ")
    clean_text = clean_text.replace(r"&amp;", "&")

    # remove multiple spaces
    while "  " in clean_text:
        clean_text = clean_text.replace("  ", " ")

    return clean_text.strip()


def _remove_links(
    clean_text: str, entity_title_to_id: Dict[str, str]
) -> Tuple[Optional[str], Optional[List[Tuple[str, Any, int, int]]]]:
    """Remove links from clean text.
    clean_text (str): Cleaned article text.
    entity_title_to_id (Dict[str, str]): Map for entity/article titles to their IDs.
    RETURNS (Tuple[Optional[str], Optional[List[Tuple[str, Any, int, int]]]]): Cleaned text without links, information
        on entities in text.
    """
    # read the text char by char to get the right offsets for the interwiki links
    entities = []
    final_text = ""
    open_read = 0
    reading_text = True
    reading_entity = False
    reading_mention = False
    reading_special_case = False
    entity_buffer = ""
    mention_buffer = ""
    for index, letter in enumerate(clean_text):
        if letter == "[":
            open_read += 1
        elif letter == "]":
            open_read -= 1
        elif letter == "|":
            if reading_text:
                final_text += letter
            # switch from reading entity_title to mention in the [[entity_title|mention]] pattern
            elif reading_entity:
                reading_text = False
                reading_entity = False
                reading_mention = True
            else:
                reading_special_case = True
        else:
            if reading_entity:
                entity_buffer += letter
            elif reading_mention:
                mention_buffer += letter
            elif reading_text:
                final_text += letter
            else:
                raise ValueError("Not sure at point", clean_text[index - 2 : index + 2])

        if open_read > 2:
            reading_special_case = True

        if open_read == 2 and reading_text:
            reading_text = False
            reading_entity = True
            reading_mention = False

        # we just finished reading an entity_title
        if open_read == 0 and not reading_text:
            if "#" in entity_buffer or entity_buffer.startswith(":"):
                reading_special_case = True
            # Ignore cases with nested structures like File: handles etc
            if not reading_special_case:
                if not mention_buffer:
                    mention_buffer = entity_buffer
                start = len(final_text)
                end = start + len(mention_buffer)
                qid = entity_title_to_id.get(entity_buffer, None)
                if qid:
                    entities.append((mention_buffer, qid, start, end))
                final_text += mention_buffer

            entity_buffer = ""
            mention_buffer = ""

            reading_text = True
            reading_entity = False
            reading_mention = False
            reading_special_case = False

    return final_text, entities


def is_dev(article_id: str) -> bool:
    """Checks whether article is dev article.
    article_id (str): Article ID.
    RETURNS (bool): Whether article is dev article.
    """
    if not article_id:
        return False
    return article_id.endswith("3")


def is_valid_article(doc_text: str) -> bool:
    """Checks whether article is valid.
    doc_text (str): Article text to check.
    RETURNS (bool): Whether article text is valid.
    """
    # custom length cut-off
    return 10 < len(doc_text) < 30000


def is_valid_sentence(sent_text: str) -> bool:
    """Checks whether sentence is valid.
    sent_text (str): Sentence to check.
    RETURNS (bool): Whether sentence is valid.
    """
    if not 10 < len(sent_text) < 3000:
        # custom length cut-off
        return False

    if sent_text.strip().startswith("*") or sent_text.strip().startswith("#"):
        # remove 'enumeration' sentences (occurs often on Wikipedia)
        return False

    return True
