#!/usr/bin/env python
""" Package setup/installation and metadata for lambdata
"""

import setuptools

REQUIRED = [
    "numpy",
    "pandas",
    "scikit-learn",
]

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="lambdata-chadeowen",
    version="0.0.4",
    author="chadeowen",
    description="A collection of Data Science helper functions",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/chadeowen/lambdata",
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=REQUIRED,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
