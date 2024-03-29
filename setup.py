from setuptools import setup, find_packages


PROJECT_URLS = {
    "Bug Tracker": "https://github.com/rs-station/abismal/issues",
    "Source Code": "https://github.com/rs-station/abismal",
}


LONG_DESCRIPTION = """
Stochastic merging for diffraction data.
"""

setup(
    name='abismal',
    version='0.0.0',
    author='Kevin M. Dalton',
    author_email='kevinmdalton@gmail.com',
    license="All Rights Reserved",
    include_package_data=True,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    description='Stochastic merging for diffraction data.',
    project_urls=PROJECT_URLS,
    python_requires=">=3.7,<3.11",
    url="https://github.com/rs-station/abismal",
    install_requires=[
        "reciprocalspaceship>=0.9.16",
        "tqdm",
        "tensorflow",
        "tensorflow-probability",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)
