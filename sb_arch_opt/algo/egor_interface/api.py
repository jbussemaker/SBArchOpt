"""
MIT License

Copyright: (c) 2023, Onera
Contact: remi.lafage@onera.fr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.algo.egor_interface.algo import *

__all__ = ["HAS_EGOBOX", "get_egor_optimizer"]


def get_egor_optimizer(
    problem: ArchOptProblemBase,
    n_init: int,
    results_folder: "None|str" = None,
    **kwargs
):
    """
    Gets the main interface to Egor.

    Use the `minimize` method to run the DOE and infill loops.
    `kwargs` arguments are directly pass to the native Egor object.
    See help(egobox.Egor) for more information
    """
    check_dependencies()
    return EgorArchOptInterface(problem, n_init, results_folder, **kwargs)
