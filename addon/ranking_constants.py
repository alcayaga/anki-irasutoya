"""Vocabulary, boundary patterns, and scoring policy for illustration ranking."""
import re
from typing import Tuple

KANJI_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf々〆〇]")
HIRAGANA_CHAR_RE = re.compile(r"[\u3040-\u309F]")
# Exclude ゠ (U+30A0) and ・ (U+30FB); they are word separators, not Katakana letters.
KATAKANA_CHAR_RE = re.compile(r"[\u30A1-\u30FA\u30FC-\u30FF]")

DECORATIVE_NOISE_KEYWORDS = ("ライン素材", "フレーム", "枠", "テンプレート", "壁紙")
TEXT_CALLOUT_KEYWORDS = ("イラスト文字",)
SITUATIONAL_NOISE_KEYWORDS = ("事故", "トラブル", "失敗", "注意", "禁止", "マーク")
VERB_TERMINAL_ENDINGS = ("う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む", "る", "た", "だ")
SINGLE_CHAR_PARTICLES = ("の", "が", "を", "に", "で", "と", "は", "も", "へ", "や", "な", "か")
MULTI_CHAR_PARTICLES = ("から", "より", "ながら", "など", "だけ", "ほど", "ばかり", "まで")
ATTRIBUTIVE_ENDINGS = ("い",) + VERB_TERMINAL_ENDINGS
COLLECTION_PREFIXES = ("いろいろな", "色々な", "様々な", "さまざまな")
COMPOUND_HEADWORD_SYNONYMS = {
    "仮面": ("マスク", "お面", "面"),
}
BYPRODUCT_MODIFIER_SEPARATORS = ("のついた", "がついた", "の付いた", "が付いた", "の入った", "が入った")
NATURAL_DISASTER_QUERIES = frozenset({
    "台風", "地震", "津波", "洪水", "噴火", "竜巻", "落雷", "土砂崩れ", "雪崩",
    "豪雨", "暴風", "暴風雨", "大雨", "大雪", "嵐", "ハリケーン", "サイクロン",
    "山火事", "火災", "火事", "災害", "自然災害", "避難", "防災",
})
SUBTYPE_PREFIXES = frozenset({
    # Colors
    "黒", "白", "赤", "青", "黄", "茶", "灰", "緑", "紫", "金", "銀",
    # Size & age
    "大", "小", "子", "巨", "微", "老", "若", "新", "古",
    # Species & attributes
    "柴", "野",
})
# Single-Kanji queries eligible for fallback subtype matching via SUBTYPE_PREFIXES when categories are absent
SUBTYPE_ENTITY_QUERIES = frozenset({
    "犬", "猫", "鳥", "魚", "虫", "馬", "牛", "豚", "羊", "熊", "鹿",
    "車", "船", "花", "木", "草", "葉", "雨", "雪", "雲", "茶", "本",
})
COMMON_OKURIGANA_NOUNS = frozenset({
    "祭り", "祭", "周り", "香り", "祈り", "便り", "望み", "恵み", "答え", "知らせ",
    "怒り", "踊り", "誓い", "願い", "想い", "思い", "集まり", "始まり", "終わり",
    "こたつ", "くつ", "ねずみ", "さる",
    "一つ", "二つ", "三つ", "四つ", "五つ", "六つ", "七つ", "八つ", "九つ",
    "近く", "遠く", "多く",
})

COMMON_KANA_VERBS = frozenset({
    "はしる", "たべる", "あるく", "およぐ", "のむ", "あそぶ",
    "みる", "きく", "ねる", "おきる", "たつ", "すわる",
    "いう", "いく", "くる", "かう", "うる", "かく", "よむ",
    "わらう", "なく", "おこる", "おどる", "うたう", "とぶ",
    "はなす", "まつ", "もつ", "しぬ", "よぶ", "あらう",
})

COMMON_ABSTRACT_NOUNS = frozenset({
    "禁止", "注意", "危険", "安全", "成功", "失敗", "平和", "関係", "状態",
    "方法", "原因", "結果", "自由", "練習", "勉強", "努力", "希望", "夢",
    "愛", "心", "気持ち", "思い出", "時間", "約束", "秘密", "ルール", "マナー",
    "名前",
})

# Anti-archetype categories (callout typography, symbols, frames, decorative lines, templates)
UNIVERSAL_ANTI_ARCHETYPE_CATEGORIES = {
    "文字": -50.0,
    "書体": -50.0,
    "ペンキ書体": -50.0,
    "白抜き書体": -50.0,
    "マーク": -40.0,
    "旗": -35.0,
    "フレーム": -50.0,
    "ライン": -50.0,
    "パターン": -45.0,
    "テンプレート": -40.0,
    "飾り": -40.0,
    "一筆箋": -40.0,
    "伝言メモ": -40.0,
    "メッセージカード": -40.0,
    "POP": -35.0,
}

# Situational hazard noise categories (penalized unless query itself is a hazard/violation term)
SITUATIONAL_HAZARD_CATEGORIES = {
    "事故": -25.0,
    "違反": -25.0,
    "怪我": -25.0,
    "災害": -25.0,
    "トラブル": -25.0,
}

# Action / motion / pose archetype categories (boosted for verbs / actions)
ACTION_ARCHETYPE_CATEGORIES = {
    "ポーズ": 30.0,
    "運動": 25.0,
    "スポーツ": 20.0,
    "表情": 20.0,
    "感情": 20.0,
    "食事": 20.0,
    "掃除": 20.0,
    "洗濯": 20.0,
    "料理": 15.0,
    "仕事": 15.0,
    "職業": 15.0,
    "ビジネス": 10.0,
    "棒人間": 15.0,
}

# Concrete physical object / entity archetype categories (boosted for nouns)
OBJECT_ARCHETYPE_CATEGORIES = {
    "食べ物": 25.0,
    "料理": 20.0,
    "食材": 20.0,
    "フルーツ": 20.0,
    "野菜": 20.0,
    "スイーツ": 20.0,
    "パン": 20.0,
    "動物": 25.0,
    "鳥": 20.0,
    "魚": 20.0,
    "犬": 20.0,
    "猫": 20.0,
    "虫": 20.0,
    "植物": 20.0,
    "花": 20.0,
    "乗り物": 20.0,
    "車": 20.0,
    "電車": 20.0,
    "自転車": 20.0,
    "道具": 20.0,
    "文房具": 20.0,
    "本": 20.0,
    "家具": 15.0,
    "家電": 15.0,
    "おもちゃ": 15.0,
    "楽器": 15.0,
}

# State / attribute / emotion archetype categories (boosted for adjectives)
ADJECTIVE_ARCHETYPE_CATEGORIES = {
    "感情": 30.0,
    "表情": 30.0,
    "ポーズ": 25.0,
    "天気": 20.0,
    "美容": 20.0,
}

COLLAGE_URL_RE = re.compile(
    r"/(?:thumb(?:n?m?a?i?l?|nai)?_|banner_|.*_thumbnail\.)",
    re.IGNORECASE,
)

STANDALONE_MATCH_BOUNDARIES: Tuple[str, ...] = (
    "exact_quoted",
    "standalone",
    "polite_prefix",
    "inflected",
)

CLOSING_BRACKETS = ("」", "』", "）", ")", "】", "］", "]", "〉", "》")

# Category adjustments: the anti-archetype cap is a signed lower bound.
ANTI_ARCHETYPE_PENALTY_CAP = -70.0
EXACT_CATEGORY_BOOST = 25.0
CATEGORY_BOOST_CAP = 35.0

# Bonuses and penalties follow the scoring branches; equal values remain independent.
CANONICAL_TITLE_BONUS = 100.0
CANONICAL_MARK_BONUS = 95.0
POLITE_CANONICAL_TITLE_BONUS = 95.0
CUTE_CANONICAL_TITLE_BONUS = 90.0
EXACT_TITLE_BONUS = 85.0
TITLE_PREFIX_BONUS = 60.0
CANONICAL_POST_TITLE_BONUS = 50.0
POST_TITLE_PREFIX_BONUS = 30.0
COLLECTION_TITLE_BONUS = 15.0
QUOTED_BOUNDARY_BONUS = 55.0
POLITE_BOUNDARY_BONUS = 45.0
STANDALONE_BOUNDARY_BONUS = 35.0
INFLECTED_BOUNDARY_BONUS = 30.0
COMPOUND_SYNONYM_BONUS = 35.0
INVALID_COMPOUND_PENALTY = 35.0
COMPOUND_BOUNDARY_BONUS = 20.0
POST_QUOTED_BOUNDARY_BONUS = 50.0
POST_POLITE_BOUNDARY_BONUS = 40.0
POST_STANDALONE_BOUNDARY_BONUS = 30.0
POST_INFLECTED_BOUNDARY_BONUS = 25.0
POST_COMPOUND_BOUNDARY_BONUS = 20.0
POST_RELEVANCE_BONUS = 15.0
DESCRIPTION_RELEVANCE_BONUS = 15.0
IRRELEVANT_IMAGE_PENALTY = 50.0
THUMBNAIL_COLLAGE_PENALTY = 80.0
SITUATIONAL_NOISE_PENALTY = 25.0
BYPRODUCT_MODIFIER_PENALTY = 20.0
TEXT_CALLOUT_PENALTY = 65.0
DECORATIVE_ASSET_PENALTY = 60.0
BACKGROUND_ASSET_PENALTY = 60.0
BACKGROUND_MENTION_PENALTY = 35.0
TITLE_LENGTH_PENALTY_PER_CHAR = 0.8
