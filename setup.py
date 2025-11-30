import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "mcp_rag_agent",
    version = "0.1.0",
    author = "Luis Rodrigues, PhD",
    author_email = "luisrodriguesphd@gmail.com",
    description = ("""A project for implementing a conversational agent powered by RAG to answer client-specific questions by performing semantic search on a vector database using a MCP server."""),
    license = "MIT",
    url = "",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=read('README.md'),
)
