"""
Setup script para Metanalyst-Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="metanalyst-agent",
    version="0.1.0",
    author="Nobrega Medtech",
    author_email="contact@nobregamedtech.com",
    description="Sistema Multi-Agente Autônomo para Meta-análises",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nobregamedtech/metanalyst-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.24.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "metanalyst-agent=metanalyst_agent.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)