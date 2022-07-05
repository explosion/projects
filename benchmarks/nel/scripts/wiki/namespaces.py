""" Information on Wiki namespaces.
Source: https://github.com/explosion/projects/blob/master/nel-wikipedia/wiki_namespaces.py.
"""

# List of meta pages in Wikidata, should be kept out of the Knowledge base
WD_META_ITEMS = [
    "Q163875",
    "Q191780",
    "Q224414",
    "Q4167836",
    "Q4167410",
    "Q4663903",
    "Q11266439",
    "Q13406463",
    "Q15407973",
    "Q18616576",
    "Q19887878",
    "Q22808320",
    "Q23894233",
    "Q33120876",
    "Q42104522",
    "Q47460393",
    "Q64875536",
    "Q66480449",
]


# TODO: add more cases from non-English WP's

# List of prefixes that refer to Wikipedia "file" pages
WP_FILE_NAMESPACE = ["Bestand", "File"]

# List of prefixes that refer to Wikipedia "category" pages
WP_CATEGORY_NAMESPACE = ["Kategori", "Category", "Categorie"]

# List of prefixes that refer to Wikipedia "meta" pages
# these will/should be matched ignoring case
WP_META_NAMESPACE = (
    WP_FILE_NAMESPACE
    + WP_CATEGORY_NAMESPACE
    + [
        "b",
        "betawikiversity",
        "Book",
        "c",
        "Commons",
        "d",
        "dbdump",
        "download",
        "Draft",
        "Education",
        "Foundation",
        "Gadget",
        "Gadget definition",
        "Gebruiker",
        "gerrit",
        "Help",
        "Image",
        "Incubator",
        "m",
        "mail",
        "mailarchive",
        "media",
        "MediaWiki",
        "MediaWiki talk",
        "Mediawikiwiki",
        "MediaZilla",
        "Meta",
        "Metawikipedia",
        "Module",
        "mw",
        "n",
        "nost",
        "oldwikisource",
        "otrs",
        "OTRSwiki",
        "Overleg gebruiker",
        "outreach",
        "outreachwiki",
        "Portal",
        "phab",
        "Phabricator",
        "Project",
        "q",
        "quality",
        "rev",
        "s",
        "spcom",
        "Special",
        "species",
        "Strategy",
        "sulutil",
        "svn",
        "Talk",
        "Template",
        "Template talk",
        "Testwiki",
        "ticket",
        "TimedText",
        "Toollabs",
        "tools",
        "tswiki",
        "User",
        "User talk",
        "v",
        "voy",
        "w",
        "Wikibooks",
        "Wikidata",
        "wikiHow",
        "Wikinvest",
        "wikilivres",
        "Wikimedia",
        "Wikinews",
        "Wikipedia",
        "Wikipedia talk",
        "Wikiquote",
        "Wikisource",
        "Wikispecies",
        "Wikitech",
        "Wikiversity",
        "Wikivoyage",
        "wikt",
        "wiktionary",
        "wmf",
        "wmania",
        "WP",
    ]
)
