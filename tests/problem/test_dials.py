from quantic.problem.dials import Dials


def test_every_dial_defaults_off():
    d = Dials()
    assert not d.concave_impact
    assert not d.discrete_participation
    assert not d.cvar_risk
    assert not d.block_trades


def test_m2a_enumerates_eight_combinations():
    """D3 is M2b; M2a's gate is 8 combinations, not 16."""
    combos = Dials.combinations()
    assert len(combos) == 8
    assert all(not d.cvar_risk for d in combos)
    assert len(set(combos)) == 8


def test_including_cvar_enumerates_all_sixteen():
    combos = Dials.combinations(include_cvar=True)
    assert len(combos) == 16
    assert len(set(combos)) == 16
    assert sum(d.cvar_risk for d in combos) == 8


def test_combinations_are_deterministically_ordered():
    assert Dials.combinations() == Dials.combinations()


def test_the_all_off_combination_is_present_and_first():
    assert Dials.combinations()[0] == Dials()


def test_label_names_the_active_dials():
    assert Dials().label == "none"
    assert Dials(concave_impact=True).label == "D1"
    assert Dials(concave_impact=True, block_trades=True).label == "D1+D4"
    assert Dials(discrete_participation=True).label == "D2"
    assert Dials(cvar_risk=True).label == "D3"


def test_labels_are_unique_across_all_combinations():
    """Every results row is keyed by this, so a collision would merge two cells."""
    combos = Dials.combinations(include_cvar=True)
    assert len({d.label for d in combos}) == len(combos)


def test_dials_are_hashable():
    assert len({Dials(), Dials()}) == 1
