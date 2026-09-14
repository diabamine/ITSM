"""Anonymisation: ground truth, detection, pseudonyms and the scoring."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from itsm.data.synthetic import generate
from itsm.privacy.anonymiser import Anonymiser
from itsm.privacy.entities import DIRECT_IDENTIFIERS, EntityType, PiiSpan, merge_spans
from itsm.privacy.evaluate import match_spans, residual_pii_rate, score_detection
from itsm.privacy.redact import Policy, Redactor, tag_for
from itsm.privacy.rules import RuleBasedRecognizer

# --------------------------------------------------------------- ground truth


def test_generated_spans_point_at_the_right_characters(tickets: pd.DataFrame) -> None:
    """The reference offsets must slice exactly the planted value.

    Everything downstream is scored against these offsets, so a silent
    off-by-one here would make every anonymisation metric meaningless.
    """
    checked = 0
    for _, row in tickets.iterrows():
        payload = json.loads(row["pii_spans"])
        for field in ("designation", "description"):
            for item in payload[field]:
                assert str(row[field])[item["start"] : item["end"]] == item["text"]
                checked += 1
    assert checked > 100


def test_dataset_contains_every_entity_type(tickets: pd.DataFrame) -> None:
    found = {
        item["entity"]
        for _, row in tickets.iterrows()
        for item in json.loads(row["pii_spans"])["description"]
    }
    assert found == {str(entity) for entity in EntityType}


def test_only_a_share_of_tickets_carry_pii(tickets: pd.DataFrame) -> None:
    """Some tickets must be clean, or precision cannot be measured."""
    with_pii = sum(
        1 for _, row in tickets.iterrows() if json.loads(row["pii_spans"])["description"]
    )
    assert 0.2 < with_pii / len(tickets) < 0.8


# ------------------------------------------------------------------- entities


def test_merge_spans_keeps_the_longest_overlap() -> None:
    short = PiiSpan(start=5, end=10, entity=EntityType.PERSON, text="Dupont", score=0.9)
    long = PiiSpan(start=0, end=10, entity=EntityType.PERSON, text="Jean Dupont", score=0.8)
    other = PiiSpan(start=20, end=30, entity=EntityType.EMAIL, text="a@b.example")

    merged = merge_spans([short, long, other])
    assert [span.text for span in merged] == ["Jean Dupont", "a@b.example"]


def test_invalid_spans_are_rejected() -> None:
    with pytest.raises(ValueError):
        PiiSpan(start=10, end=5, entity=EntityType.PERSON)


# ------------------------------------------------------------- rule detection


@pytest.mark.parametrize(
    ("text", "entity", "value"),
    [
        (
            "Vous pouvez me joindre sur jean.dupont@acme-corp.example.",
            EntityType.EMAIL,
            "jean.dupont@acme-corp.example",
        ),
        (
            "Merci de contacter Sophie Bernard au 06 12 34 56 78.",
            EntityType.PHONE,
            "06 12 34 56 78",
        ),
        ("Please contact Sarah Johnson on +44 7123 456789.", EntityType.PERSON, "Sarah Johnson"),
        (
            "Le RIB est FR76 1234 5678 9012 3456 7890 123.",
            EntityType.IBAN,
            "FR76 1234 5678 9012 3456 7890 123",
        ),
        (
            "Securite sociale : 1 85 04 75 123 456 78.",
            EntityType.NATIONAL_ID,
            "1 85 04 75 123 456 78",
        ),
        (
            "La carte 4539 1488 0343 6467 a ete refusee.",
            EntityType.CREDIT_CARD,
            "4539 1488 0343 6467",
        ),
        ("Poste de Thomas Guerin, ne le 19/03/1996.", EntityType.DATE_OF_BIRTH, "19/03/1996"),
        (
            "Le ticket concerne Karine Chevalier, matricule EMP-45231.",
            EntityType.EMPLOYEE_ID,
            "EMP-45231",
        ),
        (
            "Mon adresse est 12 rue de la Paix, 75002 Paris.",
            EntityType.ADDRESS,
            "12 rue de la Paix, 75002 Paris",
        ),
        ("Machine de Sophie Bernard : adresse 10.20.30.40.", EntityType.IP_ADDRESS, "10.20.30.40"),
    ],
)
def test_rules_detect_each_entity_type(text: str, entity: EntityType, value: str) -> None:
    spans = RuleBasedRecognizer().detect(text)
    assert any(span.entity is entity and span.text == value for span in spans), spans


@pytest.mark.parametrize(
    "text",
    [
        "Le VPN se deconnecte a Toulouse depuis la mise a jour Office.",
        "Outlook bloque sur tentative de connexion depuis ce matin.",
        "MFA enrollment failing on new phone. Please check the licence.",
        "La transaction SAP ST22 se termine en dump.",
    ],
)
def test_rules_do_not_flag_products_or_sites(text: str) -> None:
    """Over-redaction destroys the vocabulary the classifier lives on."""
    assert RuleBasedRecognizer().detect(text) == []


def test_trigger_needs_a_word_boundary() -> None:
    """'telephone' ends in the trigger 'phone'; it must not start a name."""
    spans = RuleBasedRecognizer().detect(
        "Connexion impossible depuis le nouveau telephone Merci de votre aide."
    )
    assert not [span for span in spans if span.entity is EntityType.PERSON]


def test_a_bare_date_is_not_a_date_of_birth() -> None:
    spans = RuleBasedRecognizer().detect("La mise a jour du 19/03/1996 a echoue.")
    assert not [span for span in spans if span.entity is EntityType.DATE_OF_BIRTH]


# ------------------------------------------------------------------ redaction


def test_every_entity_becomes_its_tag() -> None:
    redactor = Redactor()
    for entity in EntityType:
        assert redactor.replacement_for(entity, "whatever") == f"<{entity}>"
        assert tag_for(entity) == f"<{entity}>"


def test_a_tag_never_resembles_the_value_it_replaced() -> None:
    """The whole point of tags over realistic surrogates.

    A fake name is indistinguishable from a real one downstream, so a leak and
    a correct redaction look identical. A tag cannot be mistaken for data.
    """
    tag = Redactor().replacement_for(EntityType.PERSON, "Jean Dupont")
    assert tag.startswith("<") and tag.endswith(">")
    assert "Jean" not in tag and "Dupont" not in tag


def test_keep_policy_leaves_the_value_untouched() -> None:
    redactor = Redactor(policy={EntityType.IP_ADDRESS: Policy.KEEP})
    assert redactor.replacement_for(EntityType.IP_ADDRESS, "10.1.2.3") == "10.1.2.3"
    assert redactor.replacement_for(EntityType.PERSON, "Jean Dupont") == "<PERSON>"


def test_redaction_is_deterministic() -> None:
    text = "Contacter Sophie Bernard."
    spans = RuleBasedRecognizer().detect(text)
    assert Redactor().apply(text, spans).text == "Contacter <PERSON>."
    assert Redactor().apply(text, spans).text == "Contacter <PERSON>."


def test_anonymise_text_removes_the_values() -> None:
    text = "Merci de contacter Sophie Bernard au 06 12 34 56 78 ou sophie.bernard@acme.example."
    anonymised, spans, mapping = Anonymiser().anonymise_text(text)

    assert "Sophie Bernard" not in anonymised
    assert "06 12 34 56 78" not in anonymised
    assert "sophie.bernard@acme.example" not in anonymised
    assert anonymised == "Merci de contacter <PERSON> au <PHONE> ou <EMAIL>."
    assert len(spans) == 3
    assert len(mapping) == 3


# --------------------------------------------------------------- the scoring


def test_match_spans_counts_the_three_levels() -> None:
    reference = [PiiSpan(start=0, end=11, entity=EntityType.PERSON, text="Jean Dupont")]
    exact = [PiiSpan(start=0, end=11, entity=EntityType.PERSON, text="Jean Dupont")]
    loose = [PiiSpan(start=5, end=11, entity=EntityType.PERSON, text="Dupont")]
    mistyped = [PiiSpan(start=0, end=11, entity=EntityType.EMAIL, text="Jean Dupont")]

    assert match_spans(reference, exact)["strict"] == 1
    # A six-character hit on an eleven-character name: overlapping, so covered,
    # but IoU 0.55 and the boundaries differ, hence no strict match.
    partial = match_spans(reference, loose)
    assert (partial["strict"], partial["covered"]) == (0, 1)
    # Wrong type still removes the data: covered, but not a typed match.
    typed = match_spans(reference, mistyped)
    assert (typed["partial"], typed["covered"]) == (0, 1)


def test_score_detection_reports_leaks_and_false_positives() -> None:
    references = [[PiiSpan(start=0, end=11, entity=EntityType.PERSON, text="Jean Dupont")], []]
    predictions = [[], [PiiSpan(start=0, end=5, entity=EntityType.PERSON, text="Hello")]]

    scores = score_detection(references, predictions, texts=["Jean Dupont", "Hello"])
    assert scores["recall_coverage"] == 0.0
    assert scores["missed_direct_identifiers"] == 1
    assert scores["leak_rate_direct_identifiers"] == 1.0
    assert scores["false_positive_spans"] == 1


def test_residual_rate_catches_a_partial_redaction() -> None:
    """Redacting the surname only is a leak, however good the span looked."""
    references = [[PiiSpan(start=0, end=11, entity=EntityType.PERSON, text="Jean Dupont")]]

    # Whole value gone, first name left behind: not a full leak, but the
    # reader still knows who it is.
    partial = residual_pii_rate(["Jean <PERSON>"], references)
    assert partial["residual_values"] == 0
    assert partial["residual_partial_values"] == 1
    assert partial["residual_partial_rate_direct_identifiers"] == 1.0

    # Nothing left at all.
    clean = residual_pii_rate(["<PERSON>"], references)
    assert clean["residual_values"] == 0
    assert clean["residual_partial_values"] == 0

    # Untouched.
    leaked = residual_pii_rate(["Jean Dupont"], references)
    assert leaked["residual_values"] == 1
    assert leaked["residual_rate"] == 1.0


def test_direct_identifiers_are_the_dangerous_ones() -> None:
    assert EntityType.PERSON in DIRECT_IDENTIFIERS
    assert EntityType.IP_ADDRESS not in DIRECT_IDENTIFIERS


# ------------------------------------------------------------ the whole thing


def test_rules_backend_leaves_no_direct_identifier_on_the_corpus(
    tickets: pd.DataFrame,
) -> None:
    """The end-to-end contract: no name, mailbox or phone survives.

    Scored on the output text rather than on spans, because that is what
    actually ships.
    """
    sample = tickets.head(400)
    texts = sample["description"].astype(str).tolist()
    references = [
        [PiiSpan.from_dict(item) for item in json.loads(payload)["description"]]
        for payload in sample["pii_spans"]
    ]

    outcome = Anonymiser().anonymise_column(texts)
    detection = score_detection(references, outcome.spans, texts=texts)
    residual = residual_pii_rate(outcome.anonymised, references)

    assert detection["recall_coverage"] > 0.98
    assert detection["precision_partial"] > 0.95
    assert residual["residual_rate_direct_identifiers"] < 0.01
    # Nothing of any value survives, whole or in part.
    assert residual["residual_values"] == 0
    assert residual["residual_partial_values"] == 0


def test_generator_can_produce_a_corpus_without_pii() -> None:
    """Turning the injection off must leave the text clean."""
    dataset = generate(n_tickets=120, days=30, seed=5, pii_rate=0.0, pii_designation_rate=0.0)
    spans = [
        item
        for payload in dataset.tickets["pii_spans"]
        for field in ("designation", "description")
        for item in json.loads(payload)[field]
    ]
    assert spans == []


# ------------------------------------------------------- the transformer path


@pytest.mark.slow
def test_transformer_catches_names_the_gazetteer_never_saw() -> None:
    """The reason the second backend exists.

    The rule engine recognises a person from a trigger word or a known first
    name. A NER model reads the sentence, so it finds names no list contains -
    which is the realistic case on a real corpus. Skipped unless the ``hf``
    extra is installed.
    """
    pytest.importorskip("transformers")
    from itsm.privacy.transformer import TransformerRecognizer

    # No trigger word and no known first name: the two signals the rule engine
    # depends on are both absent, which is the everyday case on real tickets.
    text = "Krzysztof Nowak cannot log in to SAP since the last update."
    spans = TransformerRecognizer().detect(text)
    persons = {span.text for span in spans if span.entity is EntityType.PERSON}

    assert any("Nowak" in person for person in persons)
    assert not [
        span for span in RuleBasedRecognizer().detect(text) if span.entity is EntityType.PERSON
    ]


@pytest.mark.slow
def test_rules_still_win_when_a_trigger_is_present() -> None:
    """The rule engine is not a strawman: with a trigger it is exact."""
    text = "Le dossier a ete ouvert par Krzysztof Nowak pour le compte de Mei Ling Chen."
    persons = {
        span.text for span in RuleBasedRecognizer().detect(text) if span.entity is EntityType.PERSON
    }
    assert persons == {"Krzysztof Nowak", "Mei Ling Chen"}


@pytest.mark.slow
def test_transformer_still_relies_on_the_rules_for_formats() -> None:
    """A NER model knows nothing about IBANs; the hybrid is the point."""
    pytest.importorskip("transformers")
    from itsm.privacy.transformer import TransformerRecognizer

    spans = TransformerRecognizer().detect(
        "Vous pouvez me joindre sur julien.moreau@acme-corp.example, matricule EMP-45231."
    )
    entities = {span.entity for span in spans}
    assert EntityType.EMAIL in entities
    assert EntityType.EMPLOYEE_ID in entities


# ---------------------------------------------------------- the presidio path
#
# The mapping policy is tested without the extra installed: it is pure logic
# over analyser results, and it is where the decisions live (which entities are
# dropped, how offsets are tightened). Only the tests that genuinely need the
# engine are gated on the import.


class _Result:
    """Stand-in for a presidio ``RecognizerResult``."""

    def __init__(self, entity_type: str, start: int, end: int, score: float = 0.85) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


def test_presidio_maps_only_the_entities_we_treat_as_personal() -> None:
    """ORG, NRP, URL and a bare date are deliberately not personal data here."""
    from itsm.privacy.presidio import PresidioRecognizer

    text = "Camille Dubois at SAP opened it on 14/07/1988, see https://intranet.example."
    results = [
        _Result("PERSON", 0, 14),
        _Result("ORGANIZATION", 18, 21),
        _Result("DATE_TIME", 35, 45),
        _Result("URL", 51, 74),
    ]
    spans = PresidioRecognizer()._to_spans(text, results)

    assert [span.entity for span in spans] == [EntityType.PERSON]
    assert spans[0].text == "Camille Dubois"
    assert spans[0].source == "presidio:PERSON"


def test_presidio_tightens_spans_onto_their_content() -> None:
    """Offsets must stay comparable with the reference spans, which are tight."""
    from itsm.privacy.presidio import PresidioRecognizer

    text = "Contact Camille Dubois, please."
    # A padded span, as analysers routinely return for phone numbers and names.
    spans = PresidioRecognizer()._to_spans(text, [_Result("PERSON", 7, 22)])

    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Camille Dubois"


def test_presidio_drops_a_span_that_is_only_punctuation() -> None:
    """A span that tightens to nothing must not become a zero-length PiiSpan."""
    from itsm.privacy.presidio import PresidioRecognizer

    assert PresidioRecognizer()._to_spans("a  .  b", [_Result("PERSON", 1, 6)]) == []


def test_presidio_asks_for_locations_only_when_told_to() -> None:
    """Office sites are not personal data, so LOCATION is off by default."""
    from itsm.privacy.presidio import PresidioRecognizer

    assert "LOCATION" not in PresidioRecognizer().entities
    assert "LOCATION" in PresidioRecognizer(map_locations=True).entities
    assert "PERSON" in PresidioRecognizer().entities


def test_presidio_backend_is_built_without_loading_anything() -> None:
    """The factory must stay cheap: the engine loads on first detection."""
    from itsm.privacy.anonymiser import build_recognizer

    recognizer = build_recognizer("presidio", model_name="fr_core_news_lg", language="fr")

    assert recognizer.name == "presidio"
    assert recognizer.language == "fr"
    assert recognizer.spacy_model == "fr_core_news_lg"
    assert recognizer._engine is None


def test_presidio_runs_alone_so_that_it_can_be_benchmarked() -> None:
    """A composed detector cannot be measured: its gaps are covered."""
    from itsm.privacy.anonymiser import build_recognizer

    assert build_recognizer("presidio").with_structural is False
    assert build_recognizer("presidio", with_structural=True).with_structural is True


def test_isolating_a_backend_strips_the_rule_layer_from_the_transformer() -> None:
    """``--isolate`` has to hold every backend to the same terms."""
    pytest.importorskip("transformers")
    from itsm.privacy.anonymiser import build_recognizer

    assert build_recognizer("transformer").with_structural is True
    assert build_recognizer("transformer", with_structural=False).with_structural is False


def test_each_backend_gets_its_own_kind_of_model() -> None:
    """A transformer checkpoint handed to spaCy is a silent, confusing failure."""
    from itsm.config import Settings

    settings = Settings(presidio_spacy_model="fr_core_news_lg")

    assert settings.pii_model("transformer") == settings.ner_model
    assert settings.pii_model("presidio") == "fr_core_news_lg"
    assert settings.pii_model("rules") is None


@pytest.mark.slow
def test_presidio_catches_names_the_gazetteer_never_saw() -> None:
    """Same reason as the transformer backend, different engine.

    Skipped unless the ``presidio`` extra and the spaCy pipeline are installed.
    """
    pytest.importorskip("presidio_analyzer")
    from itsm.privacy.presidio import PresidioRecognizer

    text = "Krzysztof Nowak cannot log in to SAP since the last update."
    spans = PresidioRecognizer().detect(text)
    persons = {span.text for span in spans if span.entity is EntityType.PERSON}

    assert any("Nowak" in person for person in persons)
    assert not [
        span for span in RuleBasedRecognizer().detect(text) if span.entity is EntityType.PERSON
    ]


@pytest.mark.slow
def test_presidio_alone_misses_the_local_identifiers() -> None:
    """The gap the benchmark is supposed to show, asserted rather than assumed.

    Standalone, Presidio finds the mailbox - it has an ``EMAIL_ADDRESS``
    recogniser - and misses the matricule, which no predefined recogniser
    covers. Composing the rule layer underneath would hide exactly that, which
    is why this backend does not compose by default.
    """
    pytest.importorskip("presidio_analyzer")
    from itsm.privacy.presidio import PresidioRecognizer

    text = "Reach me on julien.moreau@acme-corp.example, matricule EMP-45231."

    alone = {span.entity for span in PresidioRecognizer().detect(text)}
    assert EntityType.EMAIL in alone
    assert EntityType.EMPLOYEE_ID not in alone

    composed = {span.entity for span in PresidioRecognizer(with_structural=True).detect(text)}
    assert EntityType.EMPLOYEE_ID in composed
