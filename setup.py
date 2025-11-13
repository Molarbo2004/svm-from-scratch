from setuptools import setup, find_packages

setup(
    name="svm-from-scratch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib"
    ],
    python_requires=">=3.8",
)