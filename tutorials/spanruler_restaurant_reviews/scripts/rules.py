from typing import Dict, List, Any
from spacy.util import registry


Rules = List[Dict[str, Any]]


@registry.misc("restaurant_rules.v1")
def restaurant_span_rules() -> Rules:
    rules = (
        pattern_star_ratings()
        + pattern_good_ratings()
        + pattern_poor_ratings()
        + pattern_price()
        + pattern_opening_hours()
        + pattern_attire_amenity()
        + pattern_payment_option_amenity()
        + pattern_occasion_amenity()
        + pattern_adjective_amenity()
        + pattern_landmark_location()
        + pattern_range_location()
        + pattern_drive_location()
        + pattern_common_states()
    )
    return rules


def pattern_star_ratings() -> Rules:
    """Define rules to find n-star ratings

    Set of rules that detect ratings that are based on stars.
    For example: '5 star', '4-stars', '3 stars or higher'
    """
    patterns = [
        {
            "label": "Rating",
            "pattern": [
                {"IS_DIGIT": True},
                {"LOWER": "star"},
                {"LOWER": "review", "OP": "?"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"IS_DIGIT": True},
                {"LOWER": {"REGEX": "star(s)?"}},
                {"LOWER": "or", "OP": "?"},
                {"LOWER": "higher", "OP": "?"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "more", "OP": "?"},
                {"LOWER": "than", "OP": "?"},
                {"IS_DIGIT": True},
                {"LOWER": {"REGEX": "star(s)?"}},
                {"LOWER": {"REGEX": "rat(ed|ing|ings)?"}, "OP": "?"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "at", "OP": "?"},
                {"LOWER": "least", "OP": "?"},
                {"IS_DIGIT": True},
                {"LOWER": {"REGEX": "star(s)?"}},
                {"LOWER": {"REGEX": "rat(ed|ing|ings)?"}, "OP": "?"},
            ],
        },
    ]

    return patterns


def pattern_good_ratings() -> Rules:
    """Define rules to detect high and favorable ratings"""
    patterns = [
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "best"},
                {"LOWER": "michelin"},
                {"LOWER": "rated", "OP": "?"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "highest"},
                {"LOWER": "approval"},
                {"LOWER": "ratings", "OP": "?"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "highest"},
                {"LOWER": "customer"},
                {"LOWER": "ratings"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": {"REGEX": "local(ly)?"}},
                {"LOWER": {"REGEX": "favo(u)?rite"}},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "best", "OP": "?"},
                {"LOWER": "michelin"},
                {"LOWER": "rated", "OP": "?"},
            ],
        },
    ]
    return patterns


def pattern_poor_ratings() -> Rules:
    """Define rules to detect poor ratings"""
    patterns = [
        {"label": "Rating", "pattern": [{"LOWER": "poorest"}]},
        {"label": "Rating", "pattern": [{"LOWER": "poorest"}, {"LOWER": "quality"}]},
    ]
    return patterns


def pattern_price() -> Rules:
    """Define rules that describe price"""
    patterns = [
        {
            "label": "Price",
            "pattern": [{"LOWER": "real", "OP": "?"}, {"LOWER": "cheap"}],
        },
        {
            "label": "Price",
            "pattern": [{"LOWER": "isnt", "OP": "?"}, {"LOWER": "cheap"}],
        },
        {
            "label": "Price",
            "pattern": [
                {"LOWER": "really", "OP": "?"},
                {"LOWER": "tight"},
                {"LOWER": "budget"},
            ],
        },
        {
            "label": "Price",
            "pattern": [
                {"LOWER": "meal", "OP": "?"},
                {"LOWER": "under"},
                {"IS_DIGIT": True},
            ],
        },
    ]
    return patterns


def pattern_opening_hours() -> Rules:
    """Define rules that describe opening hours

    Hours are often prefixed with phrases like 'open at' or
    'available at'. The common format is X XX am / pm.
    """
    patterns = [
        {
            "label": "Hours",
            "pattern": [
                {"LOWER": {"IN": ["open", "available"]}},
                {"LOWER": {"IN": ["at", "until", "till"]}},
                {"IS_DIGIT": True, "OP": "+"},
                {"LOWER": {"REGEX": "(a|p)?m"}},
            ],
        },
        {
            "label": "Hours",
            "pattern": [{"LOWER": {"IN": ["dinner", "lunch", "breakfast", "brunch"]}}],
        },
    ]
    return patterns


def pattern_attire_amenity() -> Rules:
    """Define rules for the type of attire often described in reviews"""
    patterns = [
        {"label": "Amenity", "pattern": [{"LOWER": "formal"}, {"LOWER": "attire"}]},
        {"label": "Amenity", "pattern": [{"LOWER": "casual"}, {"LOWER": "attire"}]},
    ]
    return patterns


def pattern_payment_option_amenity() -> Rules:
    """Define rules that detect the type of payment option and discounts"""
    patterns = [
        {"label": "Amenity", "pattern": [{"LOWER": "take"}, {"LOWER": "visa"}]},
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "take", "OP": "?"},
                {"LOWER": "master"},
                {"LOWER": "card"},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "take", "OP": "?"},
                {"LOWER": "credit"},
                {"LOWER": "card"},
            ],
        },
        {"label": "Amenity", "pattern": [{"LOWER": "senior"}, {"LOWER": "discount"}]},
        {"label": "Amenity", "pattern": [{"LOWER": "senior"}, {"LOWER": "special"}]},
    ]
    return patterns


def pattern_occasion_amenity() -> Rules:
    """Define rules that detect usual occasions found in reviews"""
    patterns = [
        {
            "label": "Amenity",
            "pattern": [{"LOWER": "date"}, {"LOWER": "night", "OP": "?"}],
        }
    ]
    return patterns


def pattern_adjective_amenity() -> Rules:
    """Define rules that detect usual adjectives found in reviews"""
    patterns = [
        {"label": "Amenity", "pattern": [{"LOWER": "classy"}]},
        {"label": "Amenity", "pattern": [{"LOWER": "clean"}]},
    ]
    return patterns


def pattern_landmark_location() -> Rules:
    """Define rules that detect common landmarks"""
    patterns = [
        {"label": "Location", "pattern": [{"LOWER": "in"}, {}, {"LOWER": "square"}]},
        {"label": "Location", "pattern": [{"LOWER": "airport"}]},
        {
            "label": "Location",
            "pattern": [{"LOWER": "street"}, {"LOWER": "address"}],
        },
        {
            "label": "Location",
            "pattern": [{"LOWER": "downtown"}, {"LOWER": "area"}],
        },
        {
            "label": "Location",
            "pattern": [{"LOWER": "in", "OP": "?"}, {"LOWER": "chinatown"}],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "off"},
                {"LOWER": "the"},
                {"LOWER": "beaten"},
                {"LOWER": "path"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "close"},
                {"LOWER": "location"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": {"REGEX": "close(st)?"}},
                {"LOWER": "to"},
                {"LOWER": "me"},
            ],
        },
    ]
    return patterns


def pattern_range_location() -> Rules:
    """Define rules that tell the range of a place."""
    patterns = [
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "within"},
                {"IS_DIGIT": True},
                {"LOWER": "minutes"},
                {"LOWER": "driving", "OP": "?"},
                {"LOWER": "distance", "OP": "?"},
            ],
        },
    ]
    return patterns


def pattern_drive_location() -> Rules:
    """Define rules based on drive distance"""
    patterns = [
        {
            "label": "Location",
            "pattern": [{"LOWER": "driving"}, {"LOWER": "distance"}],
        },
        {
            "label": "Location",
            "pattern": [{"LOWER": "fastest"}, {"LOWER": "route"}],
        },
        {
            "label": "Location",
            "pattern": [{"LOWER": "shortest"}, {"LOWER": "route"}],
        },
    ]
    return patterns


def pattern_common_states() -> Rules:
    """Define rules to look for common states in the training data"""
    patterns = [
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "in", "OP": "?"},
                {"LOWER": "roxbury"},
                {"LOWER": "crossing", "OP": "?"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "my", "OP": "?"},
                {"LOWER": "zip"},
                {"LOWER": "code"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "in", "OP": "?"},
                {"LOWER": "san"},
                {"LOWER": "jose"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "in", "OP": "?"},
                {"LOWER": "los"},
                {"LOWER": "angeles"},
            ],
        },
        {
            "label": "Location",
            "pattern": [{"LOWER": "near", "OP": "?"}, {"LOWER": "miami"}],
        },
        {
            "label": "Location",
            "pattern": [{"LOWER": "stockton"}, {"LOWER": "ca"}],
        },
        {
            "label": "Location",
            "pattern": [{"LOWER": "portland"}, {"LOWER": "or"}],
        },
    ]
    return patterns
