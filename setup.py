"""Author: Brandon Trabucco, Copyright 2019"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tf-nightly',
    'requests',
    'nltk',
    'numpy',
    'matplotlib',
    'git+https://github.com/Maluuba/nlg-eval.git@master']


setup(
    name='best_first',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('best_first')],
    description='Best First Decoding For Image Captioning')
