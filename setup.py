from setuptools import setup

setup(
    name="ifBO",
    version="0.2.1",
    description="In-context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Herilalaina Rakotoarison, Steven Adriaensen, Neeratyoy Mallik, Samir Garibov, Edward Bergman",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.2,<2",
        "scipy>=1.13.1",
        "requests>=2.23.0",
    ],
)