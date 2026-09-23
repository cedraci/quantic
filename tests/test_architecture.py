import subprocess


def test_package_imports_and_exposes_version():
    import quantic

    assert isinstance(quantic.__version__, str)
    assert quantic.__version__


def test_layer_contract_holds():
    """The layered architecture from spec section 12 is enforced mechanically."""
    result = subprocess.run(
        ["lint-imports"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
