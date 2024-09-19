from setuptools import setup, find_packages

setup(
    name="quasiq",
    version="0.1.0",
    packages=find_packages(),
    author="joey00072",
    author_email="00shxf@gmail.com",
    description="A collection of implementations of LLM, papers, and other models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/joey00072/quasiq",
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.6",
)