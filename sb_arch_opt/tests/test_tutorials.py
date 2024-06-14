import os
import pytest
import testbook
from testbook.client import TestbookNotebookClient

_docs_path = f'{os.path.dirname(__file__)}/../../docs'
_t = 60*20


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@testbook.testbook(f'{_docs_path}/tutorial.ipynb', execute=False, timeout=_t)
def test_tutorial(tb: TestbookNotebookClient):
    code_cells = []
    for cell in tb.cells:
        if cell.cell_type == 'code':
            code_cells.append(cell)

    # Set less infills to reduce testing time
    sbo_example_cell = code_cells[1]
    code = sbo_example_cell.source.split('\n')
    for i, line in enumerate(code):
        if line.startswith('n_infill'):
            code[i] = 'n_infill = 2'
            break
    sbo_example_cell.source = '\n'.join(code)

    tb.execute()


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@testbook.testbook(f'{_docs_path}/tutorial_tunable_meta_problem.ipynb', execute=True, timeout=_t)
def test_tunable_hierarchical_meta_problem_tutorial(tb):
    pass
