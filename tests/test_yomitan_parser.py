"""Unit tests for Yomitan HTML metadata extraction."""
import unittest

from yomitan_parser import (
    extract_yomitan_metadata_from_note,
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

    def test_forms_list(self):
        """Test extracting ★ forms from unordered list under (forms...) tag."""
        html_list = (
            '<ol>'
            '<li data-dictionary="JMdict [2025-11-01]"><i>(forms, JMdict [2025-11-01])</i> '
            '<ul><li>手袋（★）</li><li>手ぶくろ</li></ul></li>'
            '</ol>'
        )
        meta = parse_yomitan_html(html_list)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.forms_star, ["手袋"])

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

    def test_empty_and_malformed(self):
        """Test edge cases: None, empty, malformed HTML."""
        self.assertIsNone(parse_yomitan_html(None))
        self.assertIsNone(parse_yomitan_html(""))
        self.assertIsNone(parse_yomitan_html("Just plain text with no markers"))
        self.assertIsNone(extract_yomitan_metadata_from_note(None))
        self.assertIsNone(extract_yomitan_metadata_from_note({}))


if __name__ == "__main__":
    unittest.main()
