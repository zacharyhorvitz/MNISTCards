'''Setup for torch4u'''
from setuptools import setup, find_packages

setup(
    name='MNISTCards',
    version='0.0.1',
    url='',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'numpy',
        'matplotlib'
    ],
    include_package_data=True
)



