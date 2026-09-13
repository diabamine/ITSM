"""Frequency helpers shared by the time-based aggregations.

pandas 2.2 renamed the *resample* offset aliases (``M`` -> ``ME``, ``Q`` ->
``QE``, ``Y``/``A`` -> ``YE``) but :class:`pandas.Period` still only accepts the
old spellings. Any code that takes a user-supplied frequency and calls
``to_period`` therefore has to translate, or it raises
``ValueError: Invalid frequency: ME`` the first time somebody picks a monthly
granularity.

One helper, used by every aggregation that groups into periods, so the UI, the
CLI and the API can all speak the resample dialect.
"""

from __future__ import annotations

#: Resample alias -> Period alias.
_PERIOD_ALIASES = {"ME": "M", "MS": "M", "QE": "Q", "QS": "Q", "YE": "Y", "YS": "Y", "A": "Y"}


def period_freq(freq: str) -> str:
    """Return the ``Period`` alias matching a resample offset alias.

    Args:
        freq: Offset alias as accepted by ``resample`` (``D``, ``W``, ``ME``...).

    Returns:
        The alias ``to_period`` accepts; unknown values are returned unchanged
        so pandas raises its own, more informative error.

    Examples:
        >>> period_freq("ME"), period_freq("W"), period_freq("D")
        ('M', 'W', 'D')
    """
    return _PERIOD_ALIASES.get(freq.upper(), freq)
