"""Hermetic compatibility checks for automatic and configured note metadata."""
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from addon.batch import safe_extract_note_pos_type
from addon.config import DEFAULT_CONFIG, note_field_selection
from addon.fields import (
    FieldConfigurationError,
    FieldSelection,
    resolve_fields,
    select_fields,
    validate_overrides,
)
from addon.synonyms import _is_noun_note, _is_verb_note, lookup_bilingual_glosses
from addon.text_utils import _iter_note_readings, generate_search_candidates
from addon.yomitan_parser import extract_english_glosses_from_note, extract_yomitan_metadata_from_note
from scripts.eval_deck_search import detect_source_field

try:
    from aqt.qt import QApplication
    HAS_QT = True
except ImportError:
    HAS_QT = False

LAPIS_FIELDS = (
    "Expression ExpressionFurigana ExpressionReading ExpressionAudio SelectionText MainDefinition "
    "DefinitionPicture Sentence SentenceFurigana SentenceAudio Picture Glossary Hint IsWordAndSentenceCard "
    "IsClickCard IsSentenceCard IsAudioCard PitchPosition PitchCategories Frequency FreqSort MiscInfo"
).split()
KIKU_FIELDS = LAPIS_FIELDS + ["SentenceTranslation", "RelatedExpression"]
SENREN_FIELDS = (
    "word reading sentence sentenceFurigana sentenceTranslation sentenceCard audioCard notes hint selectionText "
    "definition wordAudio sentenceAudio picture glossary pitchAccents pitchPositions pitchCategories "
    "frequencies freqSort miscInfo dictionaryPreference"
).split()
STRUCTURED = '<ul data-sc-content="glossary"><li>cat</li></ul>'


class ModelNote(dict):
    """Minimal mapping with Anki's note-type identity interface."""

    def __init__(self, model_name: str, fields: Dict[str, str]) -> None:
        """Store fields and a stable model name without requiring a collection."""
        super().__init__(fields)
        self.model_name = model_name

    def note_type(self) -> Dict[str, str]:
        """Return the model name used by configuration resolution."""
        return {"name": self.model_name}


class TestFieldResolution(unittest.TestCase):
    """Check deterministic matching, validation, and per-model isolation."""

    def test_matching_priority_ambiguity_and_deduplication(self) -> None:
        """Prefer exact spellings and never pick an arbitrary casefold collision."""
        self.assertEqual(resolve_fields(["reading", "Reading"], ["Reading"]), ["Reading"])
        self.assertEqual(resolve_fields(["reading", "READING"], ["Reading"]), [])
        self.assertEqual(resolve_fields(["reading"], ["Reading", "reading"]), ["reading"])
        self.assertEqual(resolve_fields(["definition", "Glossary"], ["Definition", "Glossary"]),
                         ["definition", "Glossary"])

    def test_override_priority_missing_and_disabled(self) -> None:
        """Keep explicit order and warn without silently filling absent roles."""
        selected = select_fields(["one", "two", "Glossary"], "Custom", {"Custom": {
            "definition_fields": ["TWO", "missing", "one", "two"], "reading_fields": [],
        }})
        self.assertEqual(selected.definition_fields, ("two", "one"))
        self.assertEqual(selected.reading_fields, ())
        self.assertIsNone(selected.furigana_fields)
        self.assertEqual(len(selected.warnings), 1)
        self.assertIn("missing", selected.warnings[0])
        ambiguous = select_fields(["reading", "READING"], "Custom", {"Custom": {"reading_fields": ["Reading"]}})
        self.assertEqual(ambiguous.reading_fields, ())
        self.assertIn("ambiguous", ambiguous.warnings[0])

    def test_reject_malformed_overrides(self) -> None:
        """Invalid shapes and unknown roles must fail before metadata search."""
        for value in (None, [], {"Custom": []}, {"Custom": {"definitions": []}},
                      {"Custom": {"reading_fields": "Reading"}}, {"Custom": {"reading_fields": [None]}},
                      {"Custom": {"reading_fields": [""]}}, {"": {}}):
            with self.subTest(value=value), self.assertRaisesRegex(FieldConfigurationError, "configuration"):
                validate_overrides(value)

    def test_per_model_isolation(self) -> None:
        """A custom mapping affects only the exact configured note-type name."""
        config = {"note_type_field_overrides": {"Custom": {"reading_fields": ["KanaCustom"]}}}
        fields = {"reading": "ねこ", "KanaCustom": "いぬ"}
        custom = note_field_selection(ModelNote("Custom", fields), config)
        standard = note_field_selection(ModelNote("Senren", fields), config)
        self.assertEqual(_iter_note_readings(fields, custom), [("KanaCustom", "reading", "いぬ")])
        self.assertEqual(_iter_note_readings(fields, standard), [("reading", "reading", "ねこ")])

    def test_benchmark_source_detection(self) -> None:
        """Lowercase word works without relying on field order; existing priority remains."""
        self.assertEqual(detect_source_field({"audioCard": "", "word": "猫"}), "word")
        self.assertEqual(detect_source_field({"word": "猫", "Expression": "犬"}), "Expression")


class TestMetadataCompatibility(unittest.TestCase):
    """Exercise metadata consumers rather than only alias tables."""

    def test_standard_layouts(self) -> None:
        """All three standard layouts expose readings and plain/structured definitions."""
        for layout, reading, glossary in ((LAPIS_FIELDS, "ExpressionReading", "Glossary"),
                                         (KIKU_FIELDS, "ExpressionReading", "Glossary"),
                                         (SENREN_FIELDS, "reading", "glossary")):
            with self.subTest(reading=reading, fields=len(layout)):
                note = dict.fromkeys(layout, "")
                note.update({reading: "ねこ", glossary: "cat"})
                self.assertEqual(_iter_note_readings(note), [(reading, "reading", "ねこ")])
                self.assertEqual(extract_english_glosses_from_note(note), ["cat"])
                note[glossary] = STRUCTURED
                self.assertEqual(extract_yomitan_metadata_from_note(note).english_glosses, ["cat"])

    def test_lowercase_selection_and_definition(self) -> None:
        """Auxiliary selected text and definitions retain vocabulary priority."""
        self.assertEqual(extract_english_glosses_from_note({"selectionText": "cat"}), ["cat"])
        self.assertEqual(extract_english_glosses_from_note({"definition": "dog", "selectionText": "cat"}), ["dog"])
        self.assertEqual(extract_english_glosses_from_note({"sentenceTranslation": "a cat", "sentenceEng": "dog"}), [])
        self.assertEqual(_iter_note_readings({"sentenceFurigana": "猫[ねこ]がいる"}), [])

    def test_definition_override_prevents_fallback_leakage(self) -> None:
        """Unselected metadata cannot return through the any-field structured scan."""
        note = {"Custom": "dog", "Glossary": STRUCTURED, "selectionText": "bird"}
        selection = FieldSelection(definition_fields=("Custom",))
        self.assertEqual(extract_english_glosses_from_note(note, field_selection=selection), ["dog"])
        self.assertIsNone(extract_yomitan_metadata_from_note(note, field_selection=selection))
        disabled = FieldSelection(definition_fields=())
        self.assertEqual(extract_english_glosses_from_note(note, field_selection=disabled), [])
        self.assertIsNone(extract_yomitan_metadata_from_note(note, field_selection=disabled))
        self.assertEqual(extract_yomitan_metadata_from_note({"Unknown": STRUCTURED}).english_glosses, ["cat"])

    def test_custom_structured_metadata_and_pos(self) -> None:
        """Custom fields feed both structured extraction and ranking POS inference."""
        note = {"Custom": '<i>(1, v1, vt, JMdict [2025-11-01])</i>', "Glossary": STRUCTURED}
        selection = FieldSelection(definition_fields=("Custom",))
        self.assertEqual(safe_extract_note_pos_type(note, selection), "verb")
        self.assertIsNone(safe_extract_note_pos_type(note, FieldSelection(definition_fields=())))

    def test_verb_safeguards(self) -> None:
        """Lowercase and renamed definitions cannot bypass verb-versus-noun filtering."""
        for name in ("definition", "Custom"):
            with self.subTest(name=name):
                note = {name: "to train; to drill"}
                selection = None if name == "definition" else FieldSelection(definition_fields=(name,))
                self.assertTrue(_is_verb_note(None, note, selection))
                self.assertFalse(_is_noun_note(None, note, selection))
                with patch("addon.synonyms.lookup_synonyms", return_value=["電車"]):
                    self.assertEqual(lookup_bilingual_glosses(note, field_selection=selection), [])

    def test_reading_and_furigana_candidates(self) -> None:
        """Explicit fields reach the real candidate pipeline with unchanged Tier 2 confidence."""
        note = {"word": "檸檬", "reading": "ねこ", "CustomReading": "れもん", "CustomFurigana": "犬[いぬ]"}
        selection = FieldSelection(reading_fields=("CustomReading",), furigana_fields=("CustomFurigana",))
        with patch("addon.text_utils.lookup_concept_map", return_value=[]), \
             patch("addon.text_utils.lookup_kaishi_concept", return_value=[]), \
             patch("addon.text_utils.lookup_sudachi_synonyms", return_value=[]), \
             patch("addon.text_utils.lookup_llm_concept", return_value=[]), \
             patch("addon.text_utils.lookup_llm_concept_tier2", return_value=[]):
            candidates = generate_search_candidates(note["word"], note=note, field_selection=selection, allow_bilingual=False)
        self.assertIn(("れもん", "note_reading_CustomReading", 2), candidates)
        self.assertIn(("いぬ", "note_furigana_CustomFurigana", 2), candidates)
        self.assertNotIn("ねこ", [candidate[0] for candidate in candidates])


@unittest.skipUnless(HAS_QT, "aqt unavailable in headless CI")
class TestFieldDialog(unittest.TestCase):
    """Exercise real Qt mapping controls with mocked collection and catalog boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize offscreen Qt in the required Anki environment."""
        cls.app = QApplication.instance() or QApplication(["field-tests", "-platform", "offscreen"])

    def test_dialog_defaults_and_saved_choices(self) -> None:
        """Standard layouts and valid saved source/target selections work in both modes."""
        from addon.ui import IrasutoyaDialog
        for layout, source, target in ((LAPIS_FIELDS, "Expression", "DefinitionPicture"),
                                       (KIKU_FIELDS, "Expression", "DefinitionPicture"),
                                       (SENREN_FIELDS, "word", "picture")):
            for ids in ([1], [1, 2]):
                for saved in (False, True):
                    with self.subTest(source=source, ids=ids, saved=saved):
                        note = dict.fromkeys(layout, "")
                        mw = MagicMock()
                        mw.col.get_note.return_value = note
                        config = dict(DEFAULT_CONFIG)
                        if saved:
                            config.update(last_source_field=layout[1], last_target_field=layout[2])
                        with patch("addon.ui.mw", mw), patch("addon.ui.dialog.get_config", return_value=config), \
                             patch("addon.ui.dialog.default_store", return_value=MagicMock()), \
                             patch.object(IrasutoyaDialog, "_init_single_note_picker"), \
                             patch.object(IrasutoyaDialog, "_init_batch_view"):
                            dialog = IrasutoyaDialog(None, None, ids)
                            self.assertEqual(dialog.combo_source.currentText(), layout[1] if saved else source)
                            self.assertEqual(dialog.combo_target.currentText(), layout[2] if saved else target)
                            dialog.close()

    def test_warning_and_invalid_configuration(self) -> None:
        """Missing fields are visible and malformed overrides block dialog searches."""
        from addon.ui import IrasutoyaDialog
        config: Dict[str, Any] = dict(DEFAULT_CONFIG)
        config["note_type_field_overrides"] = {"Custom": {"definition_fields": ["Missing"]}}
        note = ModelNote("Custom", {"word": "猫", "picture": ""})
        mw = MagicMock()
        mw.col.get_note.return_value = note
        with patch("addon.ui.mw", mw), patch("addon.ui.dialog.get_config", return_value=config), \
             patch("addon.ui.dialog.default_store", return_value=MagicMock()), \
             patch.object(IrasutoyaDialog, "_init_batch_view"):
            dialog = IrasutoyaDialog(None, None, [1, 2])
            self.assertEqual(dialog._metadata_selection(note).definition_fields, ())
            self.assertIn("Missing", dialog.field_warning_label.text())
            self.assertFalse(dialog.field_warning_label.isHidden())
            dialog.close()
            config["note_type_field_overrides"] = []
            with self.assertRaises(FieldConfigurationError):
                IrasutoyaDialog(None, None, [1, 2])

    def test_single_and_batch_preview_forward_each_notes_overrides(self) -> None:
        """Dialog initialization and batch preview resolve metadata for each note type."""
        from addon.ui import IrasutoyaDialog
        notes = {
            1: ModelNote("Custom", {"word": "猫", "picture": "", "CustomReading": "ねこ"}),
            2: ModelNote("Senren", {"word": "犬", "picture": "", "reading": "いぬ"}),
        }
        config = dict(DEFAULT_CONFIG)
        config["note_type_field_overrides"] = {"Custom": {"reading_fields": ["CustomReading"]}}
        mw = MagicMock()
        mw.col.get_note.side_effect = notes.__getitem__
        for ids, module in (([1], "single_view"), ([1, 2], "batch_view")):
            with self.subTest(ids=ids), patch("addon.ui.mw", mw), \
                 patch("addon.ui.dialog.get_config", return_value=config), \
                 patch("addon.ui.dialog.default_store", return_value=MagicMock()), \
                 patch("aqt.qt.QTimer.singleShot"), \
                 patch(f"addon.ui.{module}.generate_search_candidates", return_value=[("猫", "primary", 1)]) as generate:
                dialog = IrasutoyaDialog(None, None, ids)
                if len(ids) > 1:
                    dialog._update_batch_preview_table()
                selections = {call.kwargs["note"].model_name: call.kwargs["field_selection"]
                              for call in generate.call_args_list}
                self.assertEqual(selections["Custom"].reading_fields, ("CustomReading",))
                if len(ids) > 1:
                    self.assertIsNone(selections["Senren"].reading_fields)
                dialog.close()

    def test_batch_execution_carries_selection_to_ranking(self) -> None:
        """The collection read snapshots overrides used by both candidates and worker POS."""
        from addon.batch import run_batch_fill
        notes = {
            1: ModelNote("Custom", {"word": "猫", "picture": "", "Custom": "cat"}),
            2: ModelNote("Senren", {"word": "犬", "picture": "", "glossary": "dog"}),
        }
        config = dict(DEFAULT_CONFIG)
        config["note_type_field_overrides"] = {"Custom": {"definition_fields": ["Custom"]}}
        mw = MagicMock()
        mw.col.get_note.side_effect = notes.__getitem__
        mw.progress.want_cancel.return_value = False
        client = MagicMock()
        client.search_with_fallbacks.return_value = ([], None, None)
        with patch("aqt.mw", mw), patch("aqt.operations.QueryOp") as query_op, \
             patch("addon.batch.default_store", return_value=MagicMock()), \
             patch("addon.batch.get_config", return_value=config), \
             patch("addon.batch.IrasutoyaClient", return_value=client), \
             patch("addon.batch.is_llm_rejected", return_value=False), \
             patch("addon.batch.generate_search_candidates", return_value=[("猫", "primary", 1)]) as generate, \
             patch("addon.batch.safe_extract_note_pos_type", return_value=None) as pos:
            run_batch_fill(None, [1, 2], "word", "picture", "skip")
            operation = query_op.call_args.kwargs
            data = operation["op"](mw.col)
            operation["success"](data)
            mw.taskman.with_progress.call_args.kwargs["task"]()
        self.assertEqual(generate.call_args_list[0].kwargs["field_selection"].definition_fields, ("Custom",))
        self.assertIsNone(generate.call_args_list[1].kwargs["field_selection"].definition_fields)
        self.assertEqual(pos.call_args_list[0].kwargs["field_selection"].definition_fields, ("Custom",))
        self.assertIsNone(pos.call_args_list[1].kwargs["field_selection"].definition_fields)
