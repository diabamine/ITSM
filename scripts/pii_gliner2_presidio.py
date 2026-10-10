"""Minimal PII extraction with GLiNER2 and Presidio, side by side.

    pip install "gliner2[local]" presidio-analyzer presidio-anonymizer \
        protobuf sentencepiece
    python -m spacy download en_core_web_lg
    python scripts/pii_gliner2_presidio.py

Both return character offsets, which is what makes their output comparable -
and directly convertible to :class:`itsm.privacy.entities.PiiSpan`.

Two things that cost time when running this for the first time:

* ``protobuf`` and ``sentencepiece`` are **not** pulled in by ``gliner2[local]``
  but its DeBERTa-v2 tokenizer needs them; without them the load fails with an
  ``ImportError`` several frames deep.
* Presidio's default configuration asks for ``en_core_web_lg``, not ``sm``. It
  tries to download it on first use, but the running interpreter cannot see a
  package installed after it started - so install it beforehand.
"""

from __future__ import annotations

TEXT = (
    "Please contact Camille Dubois on +33 6 12 34 56 78 or "
    "camille.dubois@groupe-exemple.example, employee number EMP-45231. "
    "Her card 4012 8888 8888 1881 was declined at 12 rue de la Paix, Paris."
)


def with_gliner2(text: str) -> list[tuple[str, str, int, int, float]]:
    """Zero-shot extraction: the labels are just strings you choose."""
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")
    labels = ["person", "email", "phone number", "employee number", "credit card", "address"]
    result = model.extract_entities(text, labels, include_spans=True, include_confidence=True)

    return [
        (label, item["text"], item["start"], item["end"], round(item["confidence"], 3))
        for label, items in result["entities"].items()
        for item in items
    ]


def with_presidio(text: str) -> list[tuple[str, str, int, int, float]]:
    """Rule + NER pipeline: fixed entity catalogue, checksum-validated formats."""
    from presidio_analyzer import AnalyzerEngine

    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language="en")

    return [
        (r.entity_type, text[r.start : r.end], r.start, r.end, round(r.score, 3))
        for r in sorted(results, key=lambda r: r.start)
    ]


def redact(text: str) -> str:
    """Replace every detected span with its entity tag."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    results = AnalyzerEngine().analyze(text=text, language="en")
    return AnonymizerEngine().anonymize(text=text, analyzer_results=results).text


if __name__ == "__main__":
    print(f"TEXT: {TEXT}\n")
    for name, detect in (("GLiNER2", with_gliner2), ("PRESIDIO", with_presidio)):
        print(f"--- {name} ---")
        for entity, value, start, end, score in detect(TEXT):
            print(f"  {entity:<16} {value!r:<45} [{start}:{end}] {score}")
        print()
    print(f"REDACTED (presidio): {redact(TEXT)}")
