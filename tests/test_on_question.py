"""Players: the cards show the halftime change, and the counts chart is tied back to the rate and the swing."""

from __future__ import annotations

import pandas as pd

from courtside.viz import story as S


def _calls(plays, h1, h2):
    ph = pd.DataFrame([{"player_name": "A Guard", "split": s, "plays": plays, "tov": plays * p, "tov_pct": p,
                        "tov_share": .3, "play_share": .3} for s, p in (("H1", h1), ("H2", h2))])
    return S.change_calls(ph).loc["A Guard"]


def test_counts_sentence_follows_the_call():
    within = S.counts_vs_rate("A Guard", _calls(190, .17, .22), 33, 42)
    assert within.startswith("These are counts: 33 turnovers before halftime, 42 after.")
    assert "inside Guard's normal swing" in within
    beyond = S.counts_vs_rate("A Guard", _calls(3000, .12, .20), 360, 600)
    assert "Guard is sloppier after halftime, beyond his normal swing" in beyond
    few = S.counts_vs_rate("A Guard", _calls(20, .1, .3), 2, 6)
    assert "too few plays to tell" in few and "1 turnovers" not in few
