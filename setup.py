import pathlib
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

README_FILE = "README.md"
VERSION_FILE = "tensorguide/_version.py"
VERSION_REGEXP = r"^__version__ = \'(\d+\.\d+\.\d+)\'"

r = re.search(VERSION_REGEXP, open(VERSION_FILE).read(), re.M)
if r is None:
    raise RuntimeError(f"Unable to find version string in {VERSION_FILE}.")

version = r.group(1)
long_description = open(README_FILE, encoding="utf-8").read()

# NOTE: there is no requirements file because this project only uses modules from the Python Standard Library!
# REQUIREMENTS_FILE = 'requirements.txt'
# install_requires = [str(r) for r in parse_requirements(open(REQUIREMENTS_FILE, 'rt'))]

setup(
    name="tensorguide",
    version=version,
    description="TensorGuide is an autodifferentiatin framework written in Python without external modules!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ryan Saxe",
    author_email="ryancsaxe@gmail.com",
    url="https://github.com/RyanSaxe/tensorguide",
    packages=find_packages(),
    # install_requires=install_requires,
)
