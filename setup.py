import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ciecam02",
    version="1.0.5",
    author="Dannyvi",
    author_email="dannyvis@icloud.com",
    description="An easy fast transformer between rgb and ciecam02 color space.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dannyvi/ciecam02",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy >= 1.7.1",
    ],
)

