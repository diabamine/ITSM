"""Anonymisation endpoint.

Exposed so that a caller can scrub text *before* sending it anywhere else -
including before the other endpoints of this very API. A service that only
anonymises its own storage, while accepting raw personal data over the wire and
writing it to a request log, has moved the problem rather than solved it.

The response deliberately returns the entity **types and offsets** but never
the matched values: an API response is logged, cached and screenshotted, and
echoing the personal data back would recreate the exposure the call exists to
remove.
"""

from __future__ import annotations

from fastapi import APIRouter

from itsm.api.deps import AnonymiserDep, SettingsDep
from itsm.api.schemas import AnonymisationRequest, AnonymisationResponse, AnonymisedItem, SpanOut

router = APIRouter(prefix="/v1", tags=["privacy"])


@router.post("/anonymise", response_model=AnonymisationResponse, summary="Remove personal data")
def anonymise(
    request: AnonymisationRequest, anonymiser: AnonymiserDep, settings: SettingsDep
) -> AnonymisationResponse:
    """Detect and replace personal data in each text.

    Every identifier is replaced by its entity tag - ``<PERSON>``, ``<EMAIL>``,
    ``<IBAN>`` - so the output states plainly that it was processed and what
    was removed. Nothing of the value survives.
    """
    items: list[AnonymisedItem] = []
    for text in request.texts:
        anonymised, spans, replacements = anonymiser.anonymise_text(text)
        items.append(
            AnonymisedItem(
                text=anonymised,
                n_replaced=len(replacements),
                spans=[
                    SpanOut(
                        start=span.start,
                        end=span.end,
                        entity=str(span.entity),
                        score=span.score,
                    )
                    for span in spans
                ],
            )
        )
    return AnonymisationResponse(backend=settings.anonymisation_backend, results=items)
