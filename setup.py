import os

from distutils.core             import setup
from distutils.command.build_py import build_py
from pathlib                    import Path
from subprocess                 import run

setup(name='automaxout',
      description='Library for automatic training of Maxout Networks.',
      author='Jordan Gumm',
      author_email='jordan@variantanalytics.com',
      packages=['automaxout', 'automaxout.models']
)
