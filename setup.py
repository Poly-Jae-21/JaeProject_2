from setuptools import setup, find_packages

setup(
    name='meta_city_environment',
    version='1.0.0',
    description='Generalizable urban planning for EVFCS placement',
    author='Jae Heo',
    author_email='heo27@purdue.edu',
    url='https://github.com/Poly-Jae-21/JaeProject_2',
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.12"
    ],
    install_requires=['numpy', 'gym>0.26.1', 'pyglet']
)