import subprocess
import sys

import pytest


@pytest.mark.parametrize("missing_modules", [("networkx",), ("smt",), ("networkx", "smt")])
def test_arch_sbo_import_without_optional_dependencies(missing_modules):
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import sys

missing_modules = sys.argv[1:]

class MissingDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in missing_modules:
            raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)

sys.meta_path.insert(0, MissingDependency())

from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.algo.arch_sbo import HAS_SMT, get_arch_sbo_gp
from sb_arch_opt.algo.arch_sbo.models import ModelFactory
from sb_arch_opt.algo.arch_sbo.graph import HAS_NETWORKX, GraphRepresentation, LdWloa, NodeLabeling

for module in missing_modules:
    assert module not in sys.modules
if 'smt' in missing_modules:
    assert not HAS_SMT
if 'networkx' in missing_modules:
    assert not HAS_NETWORKX
""", *missing_modules],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
