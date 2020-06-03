import setuptools
 
with open("README.md", "r") as fh:
    long_description = fh.read()
 
setuptools.setup(
    name="deepynets",
 
    version="0.0.1",
 
    author="Akarsh Saxena",
 
    author_email="akarsh.saxena.as@gmail.com",
 
    description="A Deep Learning Framework",
 
    long_description=long_description,
 
    long_description_content_type="text/markdown",
 
    url="https://github.com/akarsh-saxena",
    packages=setuptools.find_packages(exclude=['ProjectFiles', '.git', '.idea', '.gitattributes', '.gitignore']),
 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)