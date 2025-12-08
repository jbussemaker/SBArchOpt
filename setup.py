"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

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
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sb_arch_opt import __version__
from setuptools import setup, find_packages


def _get_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as fp:
        return fp.read()


if __name__ == '__main__':
    setup(
        name='sb-arch-opt',
        version=__version__,
        description='SBArchOpt: Surrogate-Based Architecture Optimization',
        long_description=_get_readme(),
        long_description_content_type='text/markdown',
        author='Jasper Bussemaker',
        author_email='jasper.bussemaker@dlr.de',
        classifiers=[
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
        ],
        license='MIT',
        install_requires=[
            'numpy<2.0',
            'pymoo~=0.6.1',
            'scipy',
            'deprecated',
            'pandas',
            'cached-property~=1.5',
            'ConfigSpace~=1.2.1',
            'more-itertools~=9.1',
            'appdirs',
        ],
        extras_require={
            'arch_sbo': [
                'smt~=2.2,!=2.4,!=2.10.0',
                'jenn~=1.0',  # Until bug has been fixed: https://github.com/SMTorg/smt/issues/769
                'numba',
                'scikit-learn',
            ],
            # 'ota': [  # pip install -r requirements-ota.txt
            #     'open_turb_arch @ git+https://github.com/jbussemaker/OpenTurbofanArchitecting@pymoo_optional#egg=open_turb_arch',
            # ],
            # 'assignment': [  # pip install -r requirements-assignment.txt
            #     'assign_enc @ git+https://github.com/jbussemaker/AssignmentEncoding#egg=assign_enc',
            # ],
            'botorch': [
                'ax-platform~=0.3.0',
                'botorch~=0.8.2',
            ],
            'trieste': [
                'trieste~=4.5',
            ],
            # 'tpe': [  # Not compatible with newer ConfigSpace
            #     'tpe==0.0.8',
            # ],
            'hebo': [
                'HEBO',
            ],
            'rocket': [
                'ambiance',
            ],
            'egor': [
                'egobox~=0.14.0',
            ],
        },
        python_requires='>=3.7',
        packages=find_packages(include='sb_arch_opt*'),
    )
