"""Vocabulary and filtering policy for catalog-backed synonym bridges."""

ENGLISH_STOP_WORDS = frozenset({
    "in", "of", "for", "a", "an", "the", "to", "at", "on", "by", "with",
    "and", "or", "from", "as", "about", "into", "over", "after", "one's",
    "sb", "sth", "someone", "something", "etc", "esp", "chiefly", "especially",
})
ENGLISH_PREPOSITIONS = frozenset({"in", "of", "for", "with", "at", "from", "on", "by", "about"})
ENGLISH_NOUN_ONLY_COLLISIONS = frozenset({"train", "weather"})

BILINGUAL_ACTION_TARGETS = frozenset({
    "手伝う",
    "協力",
    "走る",
    "全力疾走",
    "徒競走",
    "ランニング",
    "歩く",
    "散歩",
    "泳ぐ",
    "水泳",
    "歌う",
    "寝る",
    "睡眠",
    "起きる",
    "飲む",
    "食べる",
    "怒る",
    "休む",
    "休憩",
    "働く",
    "仕事",
    "勉強",
    "学ぶ",
    "旅行",
    "買い物",
    "読書",
    "本を読む",
    "ダンス",
    "踊る",
    "料理",
    "聞く",
    "死ぬ",
    "亡くなる",
    "葬式",
    "お葬式",
    "墓",
})

LOCOMOTION_ACTION_TARGETS = frozenset({
    "走る",
    "全力疾走",
    "徒競走",
    "ランニング",
    "歩く",
    "散歩",
    "泳ぐ",
    "水泳",
})

NEGATION_OR_IDIOM_TOKENS = frozenset({
    "cannot", "can't", "unable", "not", "never", "without", "no", "please",
})

PASS2_BLOCKED_PREFIXES = (
    "the rest",
    "rest of",
    "work of",
    "head of",
    "school of",
    "early bird",
)

LOCOMOTION_ACTION_WORDS = frozenset({"run", "walk", "swim", "fly"})

# Limits and disambiguation vocabularies for bilingual gloss fallback.
MAX_BILINGUAL_TARGETS = 3
MIN_GLOSS_TOKEN_LENGTH = 3
SPATIAL_MARGIN_GLOSSES = frozenset({"gap", "space", "opening", "margin", "clearance"})
STYLE_SCHOOL_GLOSSES = frozenset({"style", "manner", "doctrine", "method", "tradition"})
