from typing import Dict, List, Any
from spacy.util import registry


Rules = List[Dict[str, Any]]


@registry.misc("restaurant_rules.v2")
def restaurant_span_rules() -> Rules:
    rules = (
        pattern_restuarant_names()
        + pattern_money_amenity()
        + pattern_atmosphere_amenity()
        + pattern_other_amenity()
        + pattern_cuisine()
        + pattern_dish()
        + pattern_common_cities()
        + pattern_range_location()
        + pattern_good_ratings()
    )
    return rules


def pattern_restuarant_names() -> Rules:
    """Define rules to detect popular restaurant names"""
    patterns = [
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "mister"},
                {"LOWER": "foos"},
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
                {"LOWER": "grasons"},
                {"LOWER": "barbeque"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "chipotle"},
                {"LOWER": "mexican"},
                {"LOWER": "grill"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "chinese"},
                {"LOWER": "express"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "true"},
                {"LOWER": "thai"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "heart"},
                {"LOWER": "attack"},
                {"LOWER": "grill"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "sudbury"},
                {"LOWER": "pizza"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "pfchangs"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "butcher"},
                {"LOWER": "and"},
                {"LOWER": "the"},
                {"LOWER": "boar"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "italys"},
                {"LOWER": "little"},
                {"LOWER": "kitchen"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "asia"},
                {"LOWER": "express"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "waffle"},
                {"LOWER": "house"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "chik"},
                {"LOWER": "fa"},
                {"LOWER": "lay"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "hard"},
                {"LOWER": "rock"},
                {"LOWER": "hotel"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "donut"},
                {"LOWER": "and"},
                {"LOWER": "donuts"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "rubios"},
                {"LOWER": "fresh"},
                {"LOWER": {"REGEX": "mex(ican)?"}},
                {"LOWER": "grill"},
            ],
        },
        {
            "label": "Restaurant_Name",
            "pattern": [
                {"LOWER": "international"},
                {"LOWER": "brownie"},
            ],
        },
    ]

    return patterns

def pattern_money_amenity() -> Rules:
    """Define rules to find amenities related to money

    Set of rules that detect amentities that have to do with
    payment or discounts.
    """
    patterns = [
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "accept"},
                {"LOWER": "out"},
                {"LOWER": "of"},
                {"LOWER": "town"},
                {"LOWER": "checks"},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "accept"},
                {"LOWER": "a", "OP": "?"},
                {"LOWER": "prepaid"},
                {"LOWER": "visa"},
                {"LOWER": "gift"},
                {"LOWER": {"REGEX": "card(s)?"}},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "senior"},
                {"LOWER": "discount"},
            ],
        },
    ]
    return patterns

def pattern_atmosphere_amenity() -> Rules:
    """Define rules to find amenities related to atmosphere"""
    patterns = [
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "stars"},
                {"LOWER": "hang"},
                {"LOWER": "out"},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "jazz"},
                {"LOWER": {"REGEX": "club(s)?"}},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "nicely"},
                {"LOWER": "decorated"},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "date"},
                {"LOWER": "night"},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "dine"},
                {"LOWER": "at"},
                {"LOWER": "the", "OP": "?"},
                {"LOWER": "bar"},
            ],
        },
    ]
    return patterns

def pattern_other_amenity() -> Rules:
    """Define rules to find other amenities"""
    patterns = [
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "beer"},
                {"LOWER": "selection"},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "sitting"},
                {"LOWER": "area"},
            ],
        },
        {
            "label": "Amenity",
            "pattern": [
                {"LOWER": "made"},
                {"LOWER": "to"},
                {"LOWER": "order"},
            ],
        },   
    ]

    return patterns

def pattern_cuisine() -> Rules:
    """Define rules to find cuisine-related items"""
    patterns = [
        {
            "label": "Cuisine",
            "pattern": [
                {"LOWER": "comfort"},
                {"LOWER": "food"},
            ],
        },
        {
            "label": "Cuisine",
            "pattern": [
                {"LOWER": "asain"},
            ],
        },
        {
            "label": "Cuisine",
            "pattern": [
                {"LOWER": {"REGEX": "portug(ues|eese)"}},
            ],
        },
    ]

    return patterns

def pattern_dish() -> Rules:
    """Define rules to find dishes in reviews"""
    patterns = [
        {
            "label": "Dish",
            "pattern": [
                {"LOWER": "twice"},
                {"LOWER": "baked"},
                {"LOWER": "potatoes"},
            ],
        },
        {
            "label": "Dish",
            "pattern": [
                {"LOWER": "chips"},
                {"LOWER": "and"},
                {"LOWER": "salsa"},
            ],
        },
        {
            "label": "Dish",
            "pattern": [
                {"LOWER": "grilled"},
                {"LOWER": "cheese"},
            ],
        },
        {
            "label": "Dish",
            "pattern": [
                {"LOWER": "gyros"},
            ],
        },
    ]

    return patterns

def pattern_common_cities() -> Rules:
    """Define rules to find common cities"""
    patterns = [
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "sd"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "charlotte"},
                {"LOWER": "nc"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "patchogue"},
                {"LOWER": "new"},
                {"LOWER": "york"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "cincinnatti"},
                {"LOWER": "ohio"},
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
                {"LOWER": "less"},
                {"LOWER": "than"},
                {"IS_DIGIT": True},
                {"LOWER": {"REGEX": "mile(s)?"}},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "on"},
                {"LOWER": "the"},
                {"LOWER": "way"},
                {"LOWER": "to", "OP": "?"},
                {"LOWER": "my", "OP": "?"},
                {"LOWER": "destination", "OP": "?"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "to"},
                {"LOWER": "my"},
                {"LOWER": "zip"},
                {"LOWER": "code"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "no"},
                {"LOWER": "more"},
                {"LOWER": "than"},
                {"OP": "?"},
                {"LOWER": "minutes"},
            ],
        },
        {
            "label": "Location",
            "pattern": [
                {"LOWER": "with"},
                {"LOWER": "in"},
                {"LOWER": {"IN": ["a", "one"]}, "OP": "?"},
                {"IS_DIGIT": True, "OP": "?"},
                {"LOWER": "mile"},
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
                {"LOWER": "nationally"},
                {"LOWER": "known"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "best"},
                {"LOWER": "selection"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "best"},
                {"LOWER": "service"},
                {"LOWER": "and"},
                {"LOWER": "food"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "more"},
                {"LOWER": "than"},
                {"IS_DIGIT": True},
                {"LOWER": {"REGEX": "star(s)?"}},
                {"LOWER": "review", "OP": "?"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "top"},
                {"LOWER": "of"},
                {"LOWER": "the"},
                {"LOWER": "line"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "all"},
                {"LOWER": "the"},
                {"LOWER": "raves"},
            ],
        },
        {
            "label": "Rating",
            "pattern": [
                {"LOWER": "high"},
                {"LOWER": "ratings"},
                {"LOWER": "for"},
                {"LOWER": "its", "OP": "?"},
                {"LOWER": "service"},
            ],
        },
    ]

    return patterns