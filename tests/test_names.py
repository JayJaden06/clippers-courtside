"""Short names never mix two players up."""

from __future__ import annotations

from courtside import names as N


def test_shared_last_names_get_an_initial():
    s = N.short_names(["Baba Miller", "Jordan Miller", "Darius Garland", "Derrick Jones Jr.",
                       "Yanic Konan Niederhäuser"])
    assert s == {"Baba Miller": "B. Miller", "Jordan Miller": "J. Miller", "Darius Garland": "Garland",
                 "Derrick Jones Jr.": "Jones Jr.", "Yanic Konan Niederhäuser": "Konan Niederhäuser"}


def test_same_initial_too_falls_back_to_the_full_name():
    s = N.short_names(["John Smith", "Jake Smith", "Ben Smith"])
    assert s == {"John Smith": "John Smith", "Jake Smith": "Jake Smith", "Ben Smith": "B. Smith"}


def test_short_uses_the_team_and_falls_back_to_the_last_name():
    N.set_team(["Baba Miller", "Jordan Miller", "Kris Dunn"])
    try:
        assert N.short("Baba Miller") == "B. Miller" and N.short("Kris Dunn") == "Dunn"
        assert N.short("Ryan Dunn") == "Dunn"  # not on the team: plain last name
        # in a sentence that also names Phoenix's Ryan Dunn, ours gets his full name
        assert N.distinct("Kris Dunn", "Ryan Dunn") == "Kris Dunn"
        assert N.distinct("Kris Dunn", "Zion Williamson") == "Dunn"
    finally:
        N.set_team([])
