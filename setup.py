from setuptools import find_packages,setup
from typing import List


hypen = '-e .'
def get_requirements(file_path):
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('/n','')for req in requirements]

        if hypen in requirements:
            requirements.remove(hypen)

setup(
    name='End-end-ML-project',
    version='0.0.1',
    author='Thirumurugan240',
    author_email='thirumuruganchandru01@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')

)