# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


with open('REQUIREMENTS') as f:
    requirements = f.read()

with open('VERSION') as f:
    version = f.read()

classifiers=[  # Optional
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Education',
        'Topic :: Utilities',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ]

setup(
    name='bayhess',
    version=version,
    description='Bayesian Hessian Approximation for Stochastic Optimization',
    long_description_content_type="text/x-rst",
    long_description=readme,
    author='Andre Gustavo Carlon',
    author_email='agcarlon@gmail.com',
    keywords='stochastic optimization, Bayesian inference',
    classifiers=classifiers,
    license='GPLv3',
    install_requires=requirements,
    setup_requires=['numpydoc',
                    'sphinx>=1.3.1',
                    'sphinx_rtd_theme>=0.1.7'],
    packages=find_packages(exclude=('tests', 'docs')),
    url='https://github.com/agcarlon/bayhess',
    project_urls={  # Optional
        'Documentation': 'https://bayhess.readthedocs.io/',
        'Source': 'https://github.com/agcarlon/bayhess',
        'Manuscript': 'https://arxiv.org/abs/2208.00441'
    },
)
