#!/usr/bin/env python

from distutils.core import setup
import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='synchromesh',
    version='0.1',
    author='Gabriel Poesia, Kanishk Gandhi, Noah Goodman',
    author_email='kanishk.gandhi@stanford.edu',
    description='Reliable code generation from language models',
    long_description = "long description",
    long_description_content_type = "text/markdown",
    url='https://www.github.com/kanishkg/synchromesh/',
    project_urls = {
        "Bug Tracker": "https://github.com/kanishkg/synchromesh/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = ['synchromesh'],
    python_requires = ">=3.6",
    install_requires=['lark', 'openai', 'regex']
)