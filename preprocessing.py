from sklearn.preprocessing import OneHotEncoder
import numpy as np

# === One-hot mappings for categorical variables ===

REGISTER_OPTIONS = [
    "formal", "neutral", "informal", "slang", "technical",
    "childish", "rare", "obsolete", "poetic", "offensive"
]

TYPE_OPTIONS = [
    "literal", "idiomatic", "metaphorical", "euphemistic",
    "sarcastic", "loanword", "compound", "archaism"
]

POS_OPTIONS = ["noun", "verb", "adjective", "adverb", "other"]

GENDER_OPTIONS = ["masculine", "feminine", "neuter", "unknown"]

FREQ_MAP = {
    "extremely common": 0,
    "common": 1,
    "rare": 2,
    "extremely rare": 3
}

# === Utility functions ===

def one_hot_encode(value, options):
    """One-hot encode a value given an ordered list of options."""
    vec = [0] * len(options)
    if value in options:
        vec[options.index(value)] = 1
    return vec

def encode_metadata(register, mtype, pos, gender, frequency):
    """Encode all metadata fields into one feature vector."""
    return (
        one_hot_encode(register, REGISTER_OPTIONS) +
        one_hot_encode(mtype, TYPE_OPTIONS) +
        one_hot_encode(pos, POS_OPTIONS) +
        one_hot_encode(gender, GENDER_OPTIONS) +
        [FREQ_MAP.get(frequency, 1)]  # default to 'common'
    )
