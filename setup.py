from setuptools import setup, find_packages

setup(
    name="rag-pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "clickhouse-driver",
        "pytest",
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "structlog",
        "pydantic",
        "aiohttp"
    ],
)
