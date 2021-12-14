from setuptools import setup

with open("../VERSION") as f:
    version = f.read().strip()

with open("../README.md", "r") as f:
    long_description = f.read()

setup(
    name="adbdgl_adapter",
    author="ArangoDB",
    author_email="hackers@arangodb.com",
    version=version,
    description="Convert ArangoDB graphs to DGL & vice-versa.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arangoml/dgl-adapter",
    packages=["adbdgl_adapter"],
    include_package_data=True,
    python_requires=">=3.6",
    license="Apache Software License",
    install_requires=[
        "python-arango==7.2.0",
        "torch==1.10.0",
        "dgl==0.6.1",
    ],
    tests_require=["pytest", "pytest-cov"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
)
