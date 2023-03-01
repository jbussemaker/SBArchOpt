"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
from sb_arch_opt import __version__
from setuptools import setup

if __name__ == '__main__':
    setup(
        name='sb-arch-opt',
        version=__version__,
        description='SBArchOpt',
        long_description='Surrogate-Based Architecture Optimization',
        author='Jasper Bussemaker',
        author_email='jasper.bussemaker@dlr.de',
        classifiers=[
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        ],
        license='GPLv3',
        install_requires=[
            'pytest',
            'numpy',
            'pymoo==0.6.0',
        ],
        extras_require={
            'simple_sbo': [
                'smt==1.3.0',
            ],
        },
        python_requires='>=3.7',
        packages=['sb_arch_opt'],
    )
