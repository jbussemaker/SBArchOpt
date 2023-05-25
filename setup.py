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
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sb_arch_opt import __version__
from setuptools import setup


def _get_readme():
    with open(os.path.join(os.path.dirname(__file__), 'readme.md'), 'r') as fp:
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
            'numpy',
            'pymoo~=0.6.0.1',
            'scipy',
            'deprecated',
            'pandas',
            'cached-property~=1.5',
            'ConfigSpace~=0.6.1',
            'more-itertools~=9.1',
        ],
        extras_require={
            'arch_sbo': [
                'smt~=2.0b3',
            ],
            'ota': [
                'open_turb_arch @ git+https://github.com/jbussemaker/OpenTurbofanArchitecting@pymoo_optional#egg=open_turb_arch',
            ],
            'assignment': [
                'assign_enc @ git+https://github.com/jbussemaker/AssignmentEncoding#egg=assign_enc',
            ],
            'botorch': [
                'ax-platform~=0.3.0',
                'botorch~=0.8.2',
            ],
            'trieste': [
                'trieste~=1.0.0',
                # Until https://github.com/GPflow/GPflow/pull/2050 is merged and GPflow has been updated
                'gpflow==2.7.0', 'keras==2.10.0', 'tensorflow-probability==0.18.0',
            ],
            'tpe': [
                'tpe==0.0.8',
            ],
            'hebo': [
                'HEBO @ git+https://github.com/huawei-noah/HEBO@f050865fd2f554b5ca94642667257b365c753f29#subdirectory=HEBO',
            ],
        },
        python_requires='>=3.7',
        packages=['sb_arch_opt'],
        package_data={
            'sb_arch_opt.problems': ['turbofan_data/*'],
        },
    )
