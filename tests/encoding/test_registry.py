import pytest

from quantic.encoding.registry import RegistryBuilder, VarRegistry


def test_keys_index_in_insertion_order():
    b = RegistryBuilder()
    assert b.add(("x", 0, 0, 0)) == 0
    assert b.add(("x", 0, 0, 1)) == 1
    assert b.add(("y", 0, 0)) == 2
    reg = b.freeze()
    assert len(reg) == 3
    assert reg.index(("y", 0, 0)) == 2
    assert reg.keys[0] == ("x", 0, 0, 0)


def test_a_duplicate_key_is_refused_at_add():
    b = RegistryBuilder()
    b.add(("zeta", 0))
    with pytest.raises(ValueError, match="already registered"):
        b.add(("zeta", 0))


def test_an_unknown_key_names_itself_in_the_error():
    reg = RegistryBuilder().freeze()
    with pytest.raises(KeyError, match=r"\('u', 3, 1\)"):
        reg.index(("u", 3, 1))


def test_membership_does_not_raise():
    b = RegistryBuilder()
    b.add(("g", 0, 0))
    reg = b.freeze()
    assert ("g", 0, 0) in reg
    assert ("g", 0, 1) not in reg


def test_a_registry_built_with_duplicates_directly_is_refused():
    """freeze() is not the only door; the type itself must hold the invariant."""
    with pytest.raises(ValueError, match="duplicate"):
        VarRegistry(keys=(("x", 0), ("x", 0)))
