from setuptools import setup, find_packages

setup(
    name="finance-assist",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'langgraph',
        'langchain-core',
        'pydantic'
    ]
)