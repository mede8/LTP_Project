from setuptools import setup, find_packages

setup(
    name='LTP_Project',
    version='0.1',
    packages=find_packages(where='Modules'),
    install_requires=[
        'transformers',
        'datasets',
        'pandas',
    ],
)
