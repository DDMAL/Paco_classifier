from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Paco_classifier',
    version='1.0.0',
    description='Paco classifier',
    url='https://github.com/DDMAL/Paco_classifier',
    author='Khoi Nguyen, Wanyi Lin, Paco Castellanos',
    license='MIT License',
    packages=['Paco_classifier'],
    install_requires=requirements,
    python_requires=">=2.7.0"
)
