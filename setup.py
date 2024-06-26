from setuptools import setup, find_packages

# Read requirements.txt and use its contents as the install_requires list
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ifBo',
    version='0.0.1',
    author='ML lab - University of Freiburg',
    description='In-context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)