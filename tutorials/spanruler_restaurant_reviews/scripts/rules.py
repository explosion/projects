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
        + pattern_time_hours()
        + pattern_day_hours()
        + pattern_opening_hours()
        + pattern_attire_amenity()
        + pattern_payment_option_amenity()
        + pattern_occasion_amenity()
        + pattern_adjective_amenity()
        + pattern_liquor_amenity()
        + pattern_atmosphere_amenity()
        + pattern_landmark_location()
        + pattern_range_location()
        + pattern_drive_location()
        + pattern_common_states()
        + pattern_restaurant_names()
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
        {
            "label": "Price",
            "pattern": [
                {"LOWER": "cheap"},
                {"LOWER": "to"},
                {"LOWER": "moderate"},
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
        {
            "label": "Hours",
            "pattern": [
                {"LOWER": "open", "OP": "?"},
                {"LOWER": "past"},
                {"IS_DIGIT": True},
                {"LOWER": {"REGEX": "(a|p)?m"}},
            ],
        },
        {
            "label": "Hours",
            "pattern": [
                {"LOWER": {"IN": ["breakfast", "business"]}},
                {"LOWER": "hours"},
            ],
        },
    ]
    return patterns


def pattern_time_hours() -> Rules:
    """Define rules that describe time"""
    patterns = [
        {
            "label": "Hours",
            "pattern": [
                {"LOWER": "open"},
                {"LOWER": "for"},
                {"LOWER": "breakfast"},
            ],
        },
        {
            "label": "Hours",
            "pattern": [
                {"LOWER": "tonight"},
                {"LOWER": "at"},
                {"IS_DIGIT": True},
                {"IS_DIGIT": True, "OP": "?"},
            ],
        },
    ]
    return patterns


def pattern_day_hours() -> Rules:
    """Define rules that describe days of the week"""
    patterns = [
        {
            "label": "Hours",
            "pattern": [
                {"LOWER": "next", "OP": "?"},
                {
                    "LOWER": {
                        "IN": ["monday", "tuesday", "wednesday", "thursday", "friday"]
                    }
                },
                {"LOWER": {"REGEX": "night(s)?"}},
            ],
        },
        {
            "label": "Hours",
            "pattern": [
                {"LOWER": "open"},
                {"LOWER": {"IN": ["7", "seven"]}},
                {"LOWER": "days"},
                {"LOWER": "a"},
                {"LOWER": "week"},
            ],
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
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": {"REGEX": "accept(s)?"}},
                {"LOWER": "credit"},
                {"LOWER": "cards"},
            ],
        },
    ]
    return patterns


def pattern_occasion_amenity() -> Rules:
    """Define rules that detect usual occasions found in reviews"""
    patterns = [
        {
            "label": "Amenity",
            "pattern": [{"LOWER": "date"}, {"LOWER": "night", "OP": "?"}],
        },
    ]
    return patterns


def pattern_liquor_amenity() -> Rules:
    """Define rules that detect liquor related entities"""
    patterns = [
        # {
        #     "label": "Amenity",
        #     "pattern": [{"LOWER": "wine"}, {"LOWER": {"REGEX": "list(s)?"}, "OP": "?"}],
        # },
        # {
        #     "label": "Amenity",
        #     "pattern": [{"LOWER": "with"}, {"LOWER": "a"}, {"LOWER": "bar"}],
        # },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": {"IN": ["notable", "good"]}},
                {"LOWER": "beer"},
                {"LOWER": {"IN": ["list", "selection"]}},
            ],
        },
    ]
    return patterns


def pattern_adjective_amenity() -> Rules:
    """Define rules that detect usual adjectives found in reviews"""
    patterns = [
        {"label": "Amenity", "pattern": [{"LOWER": "classy"}]},
        {"label": "Amenity", "pattern": [{"LOWER": "clean"}]},
    ]
    return patterns


def pattern_atmosphere_amenity() -> Rules:
    """Define rules that detect atmosphere-related phrases found in reviews"""
    patterns = [
        {"label": "Amenity", "pattern": [{"LOWER": "allows"}, {"LOWER": "smoking"}]},
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


def pattern_restaurant_names() -> Rules:
    """Add patterns that check 'restaurant' or 'hotel' in proper names"""
    patterns = [
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "asia"},
                {"LOWER": "express"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "hard"},
                {"LOWER": "rock"},
                {"LOWER": "hotel", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "fatty"},
                {"LOWER": "fish"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "donut"},
                {"LOWER": "and"},
                {"LOWER": "donuts"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "caranova"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "fuji"},
                {"LOWER": "ya", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "lulus"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "tim"},
                {"LOWER": "hortons"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "evergreen"},
                {"LOWER": "taiwanese"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "tea"},
                {"LOWER": "garden"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "sebastians"},
                {"LOWER": "restaurant", "OP": "?"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "english"},
                {"LOWER": "pub"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "tgi"},
                {"LOWER": "fridays"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "hooters"},
            ],
        },
    ]
    return patterns
