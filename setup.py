from setuptools import find_packages, setup

setup(
    name='optimization',
    version='1.1.3',
    description='Inventory optimization model.',
    author='Erik Ingwersen',
    author_email='erik.ingwersen@br.ey.com',
    license='EY & Quest Diagnostics',
    packages=find_packages(),
    package_data={'': ['*.ini']},
    include_package_data=True,
    )