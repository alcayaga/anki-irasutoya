"""Unit tests for Yomitan HTML metadata extraction."""
import unittest

from addon.yomitan_parser import (
    extract_english_glosses_from_note,
    extract_yomitan_metadata_from_note,
    normalize_english_gloss,
    parse_yomitan_html,
)


class TestYomitanParser(unittest.TestCase):
    """Test suite for parsing structured JMdict metadata from Yomitan HTML fields."""

    def test_uk_orthography(self):
        """Test detecting 'uk' orthography tag."""
        html_uk = (
            '<div class="yomitan-glossary"><ol><li data-dictionary="JMdict [2025-11-01]">'
            '<i>(1, adj-no, adv, n, uk, JMdict [2025-11-01])</i> '
            '<span><ul data-sc-content="glossary"><li>as is</li></ul></span></li></ol></div>'
        )
        meta = parse_yomitan_html(html_uk)
        self.assertIsNotNone(meta)
        self.assertTrue(meta.is_uk)
        self.assertIn("uk", meta.pos_tags)
        self.assertIn("adv", meta.pos_tags)
        self.assertFalse(meta.is_particle_or_aux)

    def test_secondary_uk_sense_does_not_set_is_uk(self):
        """Test that uk on a secondary sense does not mark the entire word as is_uk."""
        html_secondary_uk = (
            '<div class="yomitan-glossary"><ol>'
            '<li data-dictionary="JMdict [2025-11-01]"><i>(1, n, JMdict [2025-11-01])</i> <span>primary sense</span></li>'
            '<li data-dictionary="JMdict [2025-11-01]"><i>(2, n, uk, JMdict [2025-11-01])</i> <span>rare/slang sense</span></li>'
            '</ol></div>'
        )
        meta = parse_yomitan_html(html_secondary_uk)
        self.assertIsNotNone(meta)
        self.assertIn("uk", meta.pos_tags)
        self.assertFalse(meta.is_uk)

    def test_verb_pos_classes(self):
        """Test extracting exact verb classes (v1, v5r, v5k, vs-i, vk)."""
        html_v1 = '<div><ol><li><i>(1, v1, vt, JMdict [2025-11-01])</i> <span>to stop</span></li></ol></div>'
        meta_v1 = parse_yomitan_html(html_v1)
        self.assertEqual(meta_v1.verb_class, "v1")

        html_v5r = '<div><ol><li><i>(1, v5r, vt, JMdict [2025-11-01])</i> <span>to distribute</span></li></ol></div>'
        meta_v5r = parse_yomitan_html(html_v5r)
        self.assertEqual(meta_v5r.verb_class, "v5r")

        html_v5k = '<div><ol><li><i>(1, v5k, vt, JMdict [2025-11-01])</i> <span>to guide</span></li></ol></div>'
        meta_v5k = parse_yomitan_html(html_v5k)
        self.assertEqual(meta_v5k.verb_class, "v5k")

        html_vsi = '<div><ol><li><i>(exp, vs-i, JMdict [2025-11-01])</i> <span>to care</span></li></ol></div>'
        meta_vsi = parse_yomitan_html(html_vsi)
        self.assertEqual(meta_vsi.verb_class, "vs-i")

    def test_grammar_particles_and_auxiliaries(self):
        """Test identifying grammar function words (prt, conj, aux)."""
        html_prt = '<div><ol><li><i>(1, prt, uk, JMdict [2025-11-01])</i> <span>until</span></li></ol></div>'
        meta_prt = parse_yomitan_html(html_prt)
        self.assertTrue(meta_prt.is_particle_or_aux)
        self.assertTrue(meta_prt.is_uk)
        self.assertIn("prt", meta_prt.pos_tags)

        html_conj = '<div><ol><li><i>(1, conj, prt, JMdict [2025-11-01])</i> <span>because</span></li></ol></div>'
        meta_conj = parse_yomitan_html(html_conj)
        self.assertTrue(meta_conj.is_particle_or_aux)

    def test_verb_with_auxiliary_sense(self):
        """Test that a verb with a secondary aux-v sense (e.g. 見る) is NOT marked as a particle/auxiliary."""
        html_miru = (
            '<div><ol>'
            '<li><i>(1, v1, vt, JMdict [2025-11-01])</i> <span>to see</span></li>'
            '<li><i>(2, aux-v, v1, JMdict [2025-11-01])</i> <span>to try</span></li>'
            '</ol></div>'
        )
        meta_miru = parse_yomitan_html(html_miru)
        self.assertIsNotNone(meta_miru)
        self.assertEqual(meta_miru.verb_class, "v1")
        self.assertFalse(meta_miru.is_particle_or_aux)

    def test_mixed_noun_and_affix_sense_not_marked_as_particle_or_aux(self):
        """Test that a noun with a secondary or concurrent affix tag (e.g. n, suf) is NOT marked grammar."""
        html_hito = (
            '<div><ol>'
            '<li><i>(1, n, suf, JMdict [2025-11-01])</i> <span>person</span></li>'
            '</ol></div>'
        )
        meta_hito = parse_yomitan_html(html_hito)
        self.assertIsNotNone(meta_hito)
        self.assertFalse(meta_hito.is_particle_or_aux)

        html_pure_suf = (
            '<div><ol>'
            '<li><i>(1, suf, uk, JMdict [2025-11-01])</i> <span>pluralizing suffix</span></li>'
            '</ol></div>'
        )
        meta_pure_suf = parse_yomitan_html(html_pure_suf)
        self.assertIsNotNone(meta_pure_suf)
        self.assertTrue(meta_pure_suf.is_particle_or_aux)

    def test_forms_table(self):
        """Test extracting canonical ★ Kanji forms from formsTable and ignoring rare ones."""
        html_table = (
            '<div class="yomitan-glossary">'
            '<table data-sc-content="formsTable">'
            '<tbody>'
            '<tr><th></th><th>交代</th><th>交替</th><th>〈田圃〉</th></tr>'
            '<tr><th>こうたい</th><td>★</td><td>★</td><td>㊒</td></tr>'
            '<tr><th>こうだい</th><td>⛬</td><td>⛬</td><td>🅁</td></tr>'
            '</tbody></table>'
            '</div>'
        )
        meta = parse_yomitan_html(html_table)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.forms_star, ["交代", "交替"])
        # Ensure non-star forms like 田圃 are not included in forms_star
        self.assertNotIn("田圃", meta.forms_star)
        self.assertEqual(meta.forms_all, ["交代", "交替"])

    def test_forms_table_column_status_filtering(self):
        """Test that formsTable columns with all-rare cells, rare-starred cells, or marked headers are excluded."""
        html_table = (
            '<div class="yomitan-glossary">'
            '<table data-sc-content="formsTable">'
            '<tbody>'
            '<tr><th></th><th>標準</th><th>並記</th><th>〈田圃〉</th><th>古語</th><th>旧記</th></tr>'
            '<tr><th>ひょうじゅん</th><td>★</td><td>⛬</td><td>★</td><td>★⛬</td><td class="rare">★</td></tr>'
            '<tr><th>ならび</th><td></td><td>🅁</td><td>🅁</td><td>🅁</td><td>×</td></tr>'
            '</tbody></table>'
            '</div>'
        )
        meta = parse_yomitan_html(html_table)
        self.assertIsNotNone(meta)
        self.assertIn("標準", meta.forms_star)
        self.assertIn("標準", meta.forms_all)
        self.assertNotIn("並記", meta.forms_all)
        self.assertNotIn("並記", meta.forms_star)
        self.assertNotIn("田圃", meta.forms_star)
        self.assertNotIn("田圃", meta.forms_all)
        self.assertNotIn("古語", meta.forms_star)
        self.assertNotIn("古語", meta.forms_all)
        self.assertNotIn("旧記", meta.forms_star)
        self.assertNotIn("旧記", meta.forms_all)

    def test_forms_list(self):
        """Test extracting ★ forms from unordered list under (forms...) tag."""
        html_list = (
            '<ol>'
            '<li data-dictionary="JMdict [2025-11-01]"><i>(forms, JMdict [2025-11-01])</i> '
            '<ul><li>手袋（★）</li><li>手ぶくろ</li><li>缶蹴り★</li></ul></li>'
            '</ol>'
        )
        meta = parse_yomitan_html(html_list)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.forms_star, ["手袋", "缶蹴り"])
        self.assertEqual(meta.forms_all, ["手袋", "手ぶくろ", "缶蹴り"])

    def test_forms_all_extracts_mixed_kana_orthography(self):
        """Test extracting non-starred orthographic variants from JMdict (forms...) entry."""
        html_kankeri = (
            '<ol>'
            '<li data-dictionary="JMdict [2025-11-01]"><i>(forms, JMdict [2025-11-01])</i> '
            '<ul><li>缶蹴り</li><li>缶けり</li></ul></li>'
            '</ol>'
        )
        meta = parse_yomitan_html(html_kankeri)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.forms_star, [])
        self.assertEqual(meta.forms_all, ["缶蹴り", "缶けり"])

    def test_forms_all_skips_outdated_markers(self):
        """Test that forms marked with rare/outdated symbols or tags are excluded from forms_all."""
        html_rare = (
            '<ol>'
            '<li data-dictionary="JMdict [2025-11-01]"><i>(forms, JMdict [2025-11-01])</i> '
            '<ul>'
            '<li>標準形</li>'
            '<li>注意形（⚠️）</li>'
            '<li>素警告形（⚠）</li>'
            '<li>廃用形（rK）</li>'
            '<li>稀少形（rare）</li>'
            '<li>〈旧字〉</li>'
            '<li>棄却形（×）</li>'
            '<li>記号形（🅁）</li>'
            '<li>星付稀少（★⛬）</li>'
            '</ul></li>'
            '</ol>'
        )
        meta = parse_yomitan_html(html_rare)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.forms_all, ["標準形"])
        self.assertNotIn("星付稀少", meta.forms_star)

    def test_see_references(self):
        """Test extracting 'see:' references while ignoring antonyms."""
        html_refs = (
            '<div class="yomitan-glossary">'
            '<ul data-sc-content="references"><li>see: <a lang="ja" href="#"><span>軽自動車</span></a>'
            '<span data-sc-content="refGlosses"> kei car</span></li></ul>'
            '<ul data-sc-content="references"><li>see: <span lang="ja"><a href="#"><span>博士</span></a>（<a href="#"><span>はくし</span></a>）</span>'
            '<span data-sc-content="refGlosses"> doctor</span></li></ul>'
            '<ul data-sc-content="antonyms"><li>antonym: <a lang="ja" href="#"><span>重自動車</span></a></li></ul>'
            '</div>'
        )
        meta = parse_yomitan_html(html_refs)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.references, ["軽自動車", "博士"])
        # Antonyms must NOT be in references
        self.assertNotIn("重自動車", meta.references)

    def test_sense_kanji_notes(self):
        """Test extracting sense-specific kanji from notes while discarding English notes."""
        html_notes = (
            '<div class="yomitan-glossary">'
            '<ul data-sc-content="notes"><li>esp. 止める, 停める</li></ul>'
            '<ul data-sc-content="notes"><li>esp. 留める</li></ul>'
            '<ul data-sc-content="notes"><li>used vocatively; esp. of an actor</li></ul>'
            '</div>'
        )
        meta = parse_yomitan_html(html_notes)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.sense_kanji, ["止める", "停める", "留める"])
        self.assertNotIn("actor", meta.sense_kanji)

    def test_extract_yomitan_metadata_from_note(self):
        """Test extracting metadata from note dictionaries with different field names."""
        note_main_def = {
            "Expression": "其の儘",
            "MainDefinition": '<ol><li><i>(uk, JMdict)</i> as is</li></ol>'
        }
        meta1 = extract_yomitan_metadata_from_note(note_main_def)
        self.assertIsNotNone(meta1)
        self.assertTrue(meta1.is_uk)

        note_glossary = {
            "Expression": "田んぼ",
            "Glossary": '<table data-sc-content="formsTable"><tr><th></th><th>田んぼ</th></tr><tr><th>たんぼ</th><td>★</td></tr></table>'
        }
        meta2 = extract_yomitan_metadata_from_note(note_glossary)
        self.assertIsNotNone(meta2)
        self.assertEqual(meta2.forms_star, ["田んぼ"])

        # MainDefinition has non-JMdict structured HTML with no data, Glossary has real JMdict
        note_fallback = {
            "Expression": "田んぼ",
            "MainDefinition": '<div data-sc-content="somethingElse"><span>No POS here</span></div>',
            "Glossary": '<ol><li><i>(1, n, uk, JMdict)</i></li></ol>',
        }
        meta_fb = extract_yomitan_metadata_from_note(note_fallback)
        self.assertIsNotNone(meta_fb)
        self.assertTrue(meta_fb.is_uk)

        # Non-Yomitan note
        note_plain = {"Expression": "猫", "Meaning": "Cat"}
        meta_none = extract_yomitan_metadata_from_note(note_plain)
        self.assertIsNone(meta_none)

    def test_normalize_english_gloss(self) -> None:
        """Test normalization and filtering of English definition glosses."""
        from addon.yomitan_parser import normalize_english_gloss

        self.assertEqual(normalize_english_gloss("to pass away"), "pass away")
        self.assertEqual(normalize_english_gloss("to be angry (at someone)"), "angry")
        self.assertEqual(normalize_english_gloss("a cat"), "cat")
        self.assertEqual(normalize_english_gloss("an apple"), "apple")
        self.assertEqual(normalize_english_gloss("the dog"), "dog")
        self.assertEqual(normalize_english_gloss("[archaic] to fly"), "fly")
        self.assertEqual(normalize_english_gloss("Cupid"), "cupid")
        self.assertEqual(normalize_english_gloss(""), "")
        self.assertEqual(normalize_english_gloss("   "), "")
        self.assertEqual(normalize_english_gloss(None), "")  # type: ignore[arg-type]
        self.assertEqual(normalize_english_gloss("日本語"), "")
        self.assertEqual(normalize_english_gloss("りんご (apple)"), "")
        self.assertEqual(normalize_english_gloss("12345"), "")
        self.assertEqual(normalize_english_gloss("too many words in this definition sentence"), "")

    def test_yomitan_structured_glossary_extraction(self) -> None:
        """Test extraction of English glosses from structured Yomitan ul tags."""
        html_content = (
            '<div class="yomitan-glossary">'
            '<ol><li><i>(v5u, vi)</i>'
            '<ul data-sc-content="glossary">'
            '<li>to pass away</li>'
            '<li>to die</li>'
            '<li><span><ul data-sc-content="examples"><li>nested example</li></ul></span></li>'
            '<li>departure</li>'
            '<li>run (e.g.; fast)</li>'
            '</ul></li></ol></div>'
        )
        meta = parse_yomitan_html(html_content)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.english_glosses, ["pass away", "die", "departure", "run"])

    def test_yomitan_legacy_span_glossary_extraction(self) -> None:
        """Test fallback extraction of English glosses from legacy span tags."""
        html_content = (
            '<div class="yomitan-glossary">'
            '<ol><li><i>(n)</i>'
            '<span>cupid</span>; <span>god of love</span>;'
            '<span data-sc-content="refGlosses">reference item</span>'
            '<span><span>nested</span> span</span>'
            '</li></ol></div>'
        )
        meta = parse_yomitan_html(html_content)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.english_glosses, ["cupid", "god of love", "nested span"])

    def test_normalize_english_gloss_parenthetical_with_japanese(self) -> None:
        """Test stripping parentheticals containing Japanese annotations while preserving Latin glosses."""
        self.assertEqual(normalize_english_gloss("cat (猫)"), "cat")
        self.assertEqual(normalize_english_gloss("dog [犬]"), "dog")
        self.assertEqual(normalize_english_gloss("猫"), "")
        self.assertEqual(normalize_english_gloss("cat (feline)"), "cat")

    def test_yomitan_excluded_containers(self) -> None:
        """Test that excluded containers (examples, antonyms) do not leak into English glosses."""
        html_content = (
            '<div class="yomitan-glossary">'
            '<ol><li><i>(n)</i>'
            '<span>cupid<ul data-sc-content="examples">'
            '<li><span>He shoots arrows like cupid.</span></li>'
            '</ul> arrow</span>; <span>god of love</span>'
            '<div data-sc-content="antonyms">'
            '<span>devil</span>'
            '</div>'
            '</li></ol></div>'
        )
        meta = parse_yomitan_html(html_content)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.english_glosses, ["cupid arrow", "god of love"])

    def test_extract_yomitan_metadata_merges_fields(self) -> None:
        """Test merging metadata across fields when an earlier field only contains glosses."""
        note = {
            "MainDefinition": (
                '<div class="yomitan-glossary">'
                '<ol><li><span>cat</span>; <span>feline</span></li></ol>'
                '</div>'
            ),
            "Glossary": (
                '<div class="yomitan-glossary">'
                '<ol><li><i>(n, vs)</i></li></ol>'
                '</div>'
            ),
        }
        meta = extract_yomitan_metadata_from_note(note)
        self.assertIsNotNone(meta)
        self.assertIn("n", meta.pos_tags)
        self.assertIn("vs", meta.pos_tags)
        self.assertEqual(meta.english_glosses, ["cat", "feline"])

    def test_extract_english_glosses_from_note(self) -> None:
        """Test extracting English glosses from various note structures."""
        from addon.yomitan_parser import extract_english_glosses_from_note

        # 1. From pre-parsed yomitan_meta
        meta = parse_yomitan_html('<ul data-sc-content="glossary"><li>to run</li></ul>')
        glosses = extract_english_glosses_from_note({}, yomitan_meta=meta)
        self.assertEqual(glosses, ["run"])

        # 2. From note with structured MainDefinition
        note_structured = {
            "Expression": "走る",
            "MainDefinition": '<ul data-sc-content="glossary"><li>to sprint</li><li>to dash</li></ul>',
        }
        self.assertEqual(extract_english_glosses_from_note(note_structured), ["sprint", "dash"])

        # 3. From note with plain-text Meaning field
        note_plain = {
            "Expression": "亡くなる",
            "Meaning": "to pass away; to die (euphemistic)",
        }
        self.assertEqual(extract_english_glosses_from_note(note_plain), ["pass away", "die"])

        # 4. From note with English field
        note_english = {
            "Expression": "猫",
            "English": "cat, feline",
        }
        self.assertEqual(extract_english_glosses_from_note(note_english), ["cat", "feline"])

        # 5. Vocabulary meaning preferred over sentence translation
        note_sentence = {
            "Expression": "主治医",
            "Sentence Meaning": "A dog barked loudly",
            "Meaning": "doctor",
        }
        self.assertEqual(extract_english_glosses_from_note(note_sentence), ["doctor"])

        # 6. Invalid / empty note
        self.assertEqual(extract_english_glosses_from_note(None), [])
        self.assertEqual(extract_english_glosses_from_note({}), [])

    def test_empty_and_malformed(self):
        """Test edge cases: None, empty, malformed HTML."""
        self.assertIsNone(parse_yomitan_html(None))
        self.assertIsNone(parse_yomitan_html(""))
        self.assertIsNone(parse_yomitan_html("Just plain text with no markers"))
        self.assertIsNone(extract_yomitan_metadata_from_note(None))
        self.assertIsNone(extract_yomitan_metadata_from_note({}))

    def test_yomitan_structured_glossary_nested_list_preserves_trailing_text(self) -> None:
        """Verify nested list inside glossary item does not prematurely close the parent item."""
        html_input = (
            '<ul data-sc-content="glossary">'
            "<li>prefix <ul><li>nested item</li></ul> suffix</li>"
            "</ul>"
        )
        meta = parse_yomitan_html(html_input)
        self.assertIsNotNone(meta)
        self.assertIn("prefix nested item suffix", meta.english_glosses)

    def test_extract_yomitan_metadata_fallback_scans_yomitan_marker(self) -> None:
        """Verify legacy Yomitan markup in non-definition fields is scanned via 'yomitan' marker."""
        note = {
            "Sentence Meaning": '<div class="yomitan-glossary"><ol><li><i>(n)</i> <span>doctor</span></li></ol></div>',
        }
        meta = extract_yomitan_metadata_from_note(note)
        self.assertIsNotNone(meta)
        self.assertIn("n", meta.pos_tags)
        self.assertIn("doctor", meta.english_glosses)

    def test_yomitan_structured_glossary_excludes_nested_references_and_notes(self) -> None:
        """Verify nested references and notes inside a glossary item are not absorbed as glosses."""
        html_input = (
            '<ul data-sc-content="glossary">'
            "<li>cat"
            '<ul data-sc-content="notes"><li>esp. domestic cat</li></ul>'
            '<ul data-sc-content="references"><li>see also: feline</li></ul>'
            "</li>"
            "</ul>"
        )
        meta = parse_yomitan_html(html_input)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.english_glosses, ["cat"])

    def test_strip_html_excluded_containers_multi_item_nesting(self) -> None:
        """Verify strip_html_excluded_containers removes nested example lists with multiple children."""
        from addon.yomitan_parser import strip_html_excluded_containers

        html_input = (
            "<div>train"
            '<ul data-sc-content="examples">'
            "<li>Example 1: to run</li>"
            "<li>Example 2: to jump</li>"
            "</ul>"
            "<span>locomotive</span>"
            "</div>"
        )
        cleaned = strip_html_excluded_containers(html_input)
        self.assertNotIn("Example 1", cleaned)
        self.assertNotIn("to run", cleaned)
        self.assertNotIn("Example 2", cleaned)
        self.assertNotIn("to jump", cleaned)
        self.assertIn("train", cleaned)
        self.assertIn("locomotive", cleaned)

    def test_strip_html_excluded_containers_void_elements_and_block_boundaries(self) -> None:
        """Verify void elements do not leave exclusion active and block containers insert boundaries."""
        from addon.yomitan_parser import strip_html_excluded_containers

        # Void elements: <img class="example"> should not discard following text
        html_img = '<div><img class="example">cat</div>'
        self.assertIn("cat", strip_html_excluded_containers(html_img))

        # <br> should create a boundary
        html_br = "cat<br>dog"
        self.assertEqual(strip_html_excluded_containers(html_br).strip(), "cat; dog")

        # Block-level excluded container boundary: "cat<ul>...</ul>dog" -> "cat; dog"
        html_block = 'cat<ul data-sc-content="examples"><li>to run</li></ul>dog'
        cleaned = strip_html_excluded_containers(html_block)
        parts = [p.strip() for p in cleaned.split(";") if p.strip()]
        self.assertEqual(parts, ["cat", "dog"])

    def test_plain_text_meaning_preferred_over_marked_up_sentence_meaning(self) -> None:
        """Verify plain-text Meaning takes priority over Yomitan marked-up Sentence Meaning."""
        note = {
            "Expression": "主治医",
            "Sentence Meaning": '<div class="yomitan-glossary"><ol><li><i>(n)</i> <span>dog barked</span></li></ol></div>',
            "Meaning": "doctor",
        }
        self.assertEqual(extract_english_glosses_from_note(note), ["doctor"])

    def test_sense_capping_to_first_two_senses(self) -> None:
        """Verify YomitanHTMLParser only extracts glosses from senses 1 and 2."""
        html_multi_sense = (
            '<div class="yomitan-glossary"><ol>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(1, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>circumstances</li><li>condition</li></ul></span></li>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(2, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>arranging</li></ul></span></li>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(3, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>lending money</li></ul></span></li>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(4, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>in all</li></ul></span></li>'
            '</ol></div>'
        )
        meta = parse_yomitan_html(html_multi_sense)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.english_glosses, ["circumstances", "condition", "arranging"])
        self.assertNotIn("lending money", meta.english_glosses)
        self.assertNotIn("in all", meta.english_glosses)

    def test_secondary_dictionary_reset_ignored_for_glosses(self) -> None:
        """Verify secondary dictionary resetting to sense 1 does not pollute primary glosses."""
        html_two_dicts = (
            '<div class="yomitan-glossary"><ol>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(1, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>apple</li></ul></span></li>'
            '<li data-dictionary="Daijisen"><i>(1, 名)</i> '
            '<span><ul data-sc-content="glossary"><li>ringo fruit tree</li></ul></span></li>'
            '</ol></div>'
        )
        meta = parse_yomitan_html(html_two_dicts)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.english_glosses, ["apple"])
        self.assertNotIn("ringo fruit tree", meta.english_glosses)

    def test_grammar_dictionary_excluded_from_glosses_and_stripper(self) -> None:
        """Verify grammar encyclopedias are completely excluded from definition extraction."""
        from addon.yomitan_parser import strip_html_excluded_containers

        html_grammar = (
            '<div class="yomitan-glossary"><ol>'
            '<li data-dictionary="日本語文法辞典(全集)">'
            '<span>pasokon means personal computer</span></li>'
            '</ol></div>'
        )
        meta = parse_yomitan_html(html_grammar)
        # Even if parsed, english_glosses must be empty because dictionary is marked as grammar
        self.assertTrue(meta is None or not meta.english_glosses)

        cleaned = strip_html_excluded_containers(html_grammar)
        self.assertNotIn("personal computer", cleaned)
        self.assertNotIn("pasokon", cleaned)

        # Basic Japanese Grammar (DOJG) test
        html_dojg = (
            '<div class="yomitan-glossary">'
            '<div data-dictionary="A Dictionary of Basic Japanese Grammar">'
            '<span>example of telephone call</span></div></div>'
        )
        cleaned_dojg = strip_html_excluded_containers(html_dojg)
        self.assertNotIn("telephone call", cleaned_dojg)

    def test_grammar_dict_before_jmdict_does_not_block_primary(self) -> None:
        """Verify excluded grammar dictionary appearing first does not establish primary dict or drop JMdict senses."""
        html_grammar_first = (
            '<div class="yomitan-glossary"><ol>'
            '<li data-dictionary="日本語文法辞典(全集)">'
            '<span>pasokon means personal computer</span></li>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(1, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>primary sense</li></ul></span></li>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(2, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>secondary sense</li></ul></span></li>'
            '</ol></div>'
        )
        meta = parse_yomitan_html(html_grammar_first)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.english_glosses, ["primary sense", "secondary sense"])

    def test_plain_text_quantities_not_truncated_as_sense_3(self) -> None:
        """Verify definitions with quantities like '3 people' or '3 times' are not truncated as sense 3."""
        note_quantity = {"Expression": "単語", "Meaning": "a room for 3 people"}
        self.assertEqual(extract_english_glosses_from_note(note_quantity), ["room for 3 people"])

        note_times = {"Expression": "単語", "Meaning": "done 3 times daily"}
        self.assertEqual(extract_english_glosses_from_note(note_times), ["done 3 times daily"])

    def test_decimal_quantity_not_truncated_as_sense_3(self) -> None:
        """Verify decimal quantities like '3.5 liter' are not truncated as sense 3 or split."""
        note_decimal = {"Expression": "単語", "Meaning": "a 3.5 liter bottle"}
        self.assertEqual(extract_english_glosses_from_note(note_decimal), ["3.5 liter bottle"])

    def test_japanese_only_dict_before_jmdict_preserves_jmdict_senses(self) -> None:
        """Verify Japanese-only dict with no glosses appearing first does not drop JMdict secondary senses."""
        html_ja_first = (
            '<div class="yomitan-glossary"><ol>'
            '<li data-dictionary="KireiCake"><i>(名)</i> '
            '<span>リンゴの木</span></li>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(1, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>apple</li></ul></span></li>'
            '<li data-dictionary="JMdict [2026-08-20]"><i>(2, n, JMdict)</i> '
            '<span><ul data-sc-content="glossary"><li>apple tree</li></ul></span></li>'
            '</ol></div>'
        )
        meta = parse_yomitan_html(html_ja_first)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.english_glosses, ["apple", "apple tree"])

    def test_circled_number_gloss_normalization(self) -> None:
        """Verify circled number markers are stripped before NFKC so search keys remain clean."""
        self.assertEqual(normalize_english_gloss("① pass away"), "pass away")
        self.assertEqual(normalize_english_gloss("① cat"), "cat")
        self.assertEqual(normalize_english_gloss("② dog"), "dog")

    def test_plain_text_sense_3_truncation(self) -> None:
        """Verify plain text definition extraction truncates at sense 3."""
        note_numbered = {"Expression": "単語", "Meaning": "1. apple; 2. fruit; 3. slang"}
        self.assertEqual(extract_english_glosses_from_note(note_numbered), ["apple", "fruit"])

        note_parentheses = {"Expression": "単語", "Meaning": "1) dog; 2) hound; 3) canine"}
        self.assertEqual(extract_english_glosses_from_note(note_parentheses), ["dog", "hound"])

        note_circled = {"Expression": "単語", "Meaning": "① cat ② feline ③ pet"}
        self.assertEqual(extract_english_glosses_from_note(note_circled), ["cat", "feline"])

        note_bracketed = {"Expression": "単語", "Meaning": "1. start; 2. begin; 3〔rare〕 drink"}
        self.assertEqual(extract_english_glosses_from_note(note_bracketed), ["start", "begin"])

        note_paren_jmdict = {
            "Expression": "単語",
            "Meaning": "(1, n, JMdict) apple (2, n, JMdict) fruit (3, n, JMdict) slang",
        }
        self.assertEqual(extract_english_glosses_from_note(note_paren_jmdict), ["apple", "fruit"])

        note_wide_paren = {
            "Expression": "単語",
            "Meaning": "（1, n）dog（2, n）hound（3, n）canine",
        }
        self.assertEqual(extract_english_glosses_from_note(note_wide_paren), ["dog", "hound"])

    def test_normalize_english_gloss_the_rest(self) -> None:
        """Verify normalize_english_gloss preserves 'the rest' while stripping other leading articles."""
        self.assertEqual(normalize_english_gloss("the rest"), "the rest")
        self.assertEqual(normalize_english_gloss("the rest of"), "rest of")
        self.assertEqual(normalize_english_gloss("the dog"), "dog")
        self.assertEqual(normalize_english_gloss("a car"), "car")

    def test_merge_metadata_preserves_target_english_glosses(self) -> None:
        """Verify _merge_metadata preserves primary field glosses and does not overwrite with secondary."""
        from addon.yomitan_parser import YomitanMetadata, _merge_metadata

        target = YomitanMetadata(pos_tags={"n"}, english_glosses=["doctor"])
        source = YomitanMetadata(pos_tags={"n"}, english_glosses=["physician"])
        _merge_metadata(target, source)
        self.assertEqual(target.english_glosses, ["doctor"])

        # If target has no glosses, it accepts source's glosses
        empty_target = YomitanMetadata(pos_tags={"n"})
        _merge_metadata(empty_target, source)
        self.assertEqual(empty_target.english_glosses, ["physician"])

    def test_header_before_structured_entry_not_completed(self) -> None:
        """Verify top-level header span does not lock or complete dictionary before structured entry."""
        html = (
            "<span>JMdict</span>"
            '<div data-dictionary="JMdict">'
            '<ul data-sc-content="glossary">'
            "<li>apple</li>"
            "</ul>"
            "</div>"
        )
        meta = parse_yomitan_html(html)
        self.assertEqual(meta.english_glosses, ["apple"])

    def test_legacy_entry_numbered_example_before_sense_2(self) -> None:
        """Verify numbered sense marker inside excluded example does not complete primary dictionary."""
        html = (
            "<i>(1, n)</i><span>first definition</span>"
            '<div data-sc-content="examples"><i>(1)</i> Example sentence here</div>'
            "<i>(2, n)</i><span>second definition</span>"
        )
        meta = parse_yomitan_html(html)
        self.assertEqual(meta.english_glosses, ["first definition", "second definition"])

    def test_is_grammar_sense_with_pronoun_and_affixes(self) -> None:
        """Verify is_grammar_sense treats pn as content POS, preserving lexical affix combinations."""
        from addon.yomitan_parser import is_grammar_sense

        # Standalone affix is grammar sense
        self.assertTrue(is_grammar_sense({"pref"}))
        self.assertTrue(is_grammar_sense({"suf"}))
        # Affix with noun is content sense (not grammar sense)
        self.assertFalse(is_grammar_sense({"pref", "n"}))
        # Affix with pronoun is content sense (not grammar sense)
        self.assertFalse(is_grammar_sense({"pref", "pn"}))
        self.assertFalse(is_grammar_sense({"suf", "pn"}))
        # Particles and copulas remain grammar senses regardless
        self.assertTrue(is_grammar_sense({"prt", "n"}))
        self.assertTrue(is_grammar_sense({"cop"}))
        self.assertFalse(is_grammar_sense({"n"}))


if __name__ == "__main__":
    unittest.main()
