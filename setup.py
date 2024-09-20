from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("abismal/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version

__version__ = getVersionNumber()

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/rs-station/abismal/issues",
    "Source Code": "https://github.com/rs-station/abismal",
}


LONG_DESCRIPTION = """
Stochastic merging for diffraction data.
"""

setup(
    name='abismal',
    version=__version__,
    author='Kevin M. Dalton',
    author_email='kevinmdalton@gmail.com',
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
        "tensorflow",
        "tf_keras",
        "tensorflow-probability",
        "rs-booster",
        "ray",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
    entry_points={
        "console_scripts": [
            "abismal=abismal.command_line.abismal:main",
        ]
    },
)
