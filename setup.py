from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("abismal/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/rs-station/abismal/issues",
    "Source Code": "https://github.com/rs-station/abismal",
}


LONG_DESCRIPTION = """
Stochastic merging for diffraction data.
"""

setup(
    name='abismal',
    version='0.0.1',
    author='Kevin M. Dalton',
    author_email='kmdalton@slac.stanford.edu',
    license="MIT",
    include_package_data=True,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    description='Stochastic merging for diffraction data.',
    project_urls=PROJECT_URLS,
    python_requires=">=3.7,<3.12",
    url="https://github.com/rs-station/abismal",
    install_requires=[
        "reciprocalspaceship>=0.9.16",
        "tqdm",
        "torch",
        "rs-distributions",
        "rs-booster",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
    #entry_points={
    #    "console_scripts": [
    #        "abismal=abismal.command_line.abismal:main",
    #    ]
    #},
)
