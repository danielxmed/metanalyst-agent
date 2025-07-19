"""Setup script for Metanalyst-Agent"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="metanalyst-agent",
    version="0.1.0",
    author="Nobrega Medtech",
    author_email="contact@nobregamedtech.com",
    description="AI-first multi-agent system for automated medical meta-analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nobrega-medtech/metanalyst-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "production": [
            "psycopg2-binary>=2.9.9",
            "redis>=5.0.8",
            "pymongo>=4.8.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "notebook>=7.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "metanalyst-agent=metanalyst_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "metanalyst_agent": [
            "config/*.yaml",
            "templates/*.html",
            "static/*",
        ],
    },
    keywords=[
        "meta-analysis",
        "systematic-review",
        "medical-research",
        "ai-agents",
        "langgraph",
        "evidence-based-medicine",
        "biostatistics",
        "literature-review"
    ],
    project_urls={
        "Bug Reports": "https://github.com/nobrega-medtech/metanalyst-agent/issues",
        "Source": "https://github.com/nobrega-medtech/metanalyst-agent",
        "Documentation": "https://metanalyst-agent.readthedocs.io/",
    },
)