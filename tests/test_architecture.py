import subprocess
import sys
from pathlib import Path


def test_package_imports_and_exposes_version():
    import quantic

    assert isinstance(quantic.__version__, str)
    assert quantic.__version__


def test_layer_contract_holds():
    """The layered architecture from spec section 12 is enforced mechanically."""
    # Resolve the console script next to the running interpreter rather than via
    # PATH: pytest is routinely invoked as `.venv/Scripts/python -m pytest` with
    # the venv unactivated, and PATH lookup then finds nothing.
    scripts = Path(sys.executable).parent
    exe = scripts / "lint-imports.exe"
    if not exe.exists():
        exe = scripts / "lint-imports"
    assert exe.exists(), f"lint-imports not installed next to {sys.executable}"

    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
