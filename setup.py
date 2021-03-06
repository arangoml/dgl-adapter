from setuptools import setup

with open("./README.md") as fp:
    long_description = fp.read()

setup(
    name="adbdgl_adapter",
    author="Anthony Mahanna",
    author_email="anthony.mahanna@arangodb.com",
    description="Convert ArangoDB graphs to DGL & vice-versa.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arangoml/dgl-adapter",
    keywords=["arangodb", "dgl", "adapter"],
    packages=["adbdgl_adapter"],
    include_package_data=True,
    python_requires=">=3.7",
    license="Apache Software License",
    install_requires=[
        "requests>=2.27.1",
        "dgl>=0.6.1",
        "torch>=1.10.2",
        "python-arango>=7.4.1",
        "setuptools>=45",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8>=3.8.0",
            "isort>=5.0.0",
            "mypy>=0.790",
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "coveralls>=3.3.1",
            "types-setuptools",
            "types-requests",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
)
