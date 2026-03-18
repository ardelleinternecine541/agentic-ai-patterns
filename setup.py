from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentic-patterns",
    version="0.1.0",
    author="Camilo Girardelli",
    author_email="contact@girardelli.tech",
    description="Design patterns for building autonomous AI agent systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/girardelli/agentic-ai-patterns",
    project_urls={
        "Bug Tracker": "https://github.com/girardelli/agentic-ai-patterns/issues",
        "Documentation": "https://github.com/girardelli/agentic-ai-patterns",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.3.0",
        "anthropic>=0.7.0",
        "pydantic>=2.0.0",
        "tenacity>=8.2.0",
        "redis>=5.0.0",
        "structlog>=23.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords="AI agents patterns orchestration reasoning",
    long_description_content_type="text/markdown",
)
