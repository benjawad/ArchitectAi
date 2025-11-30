"""
Setup configuration for ArchitectAI MCP
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="architectai-mcp",
    version="1.0.0",
    description="Autonomous Cloud Refactoring - Design Pattern Analysis & AI-Powered Recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ArchitectAI Team",
    author_email="team@architectai.dev",
    url="https://github.com/yourusername/architectai-mcp",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "mcp>=0.1.0",
        "anthropic>=0.7.0",
        "plantuml>=0.3.0",
        "pydantic>=2.0.0",
        "langchain-core>=0.1.0",
        "gradio>=4.0.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "modal": [
            "modal>=0.55.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "architectai-mcp=mcp_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Quality Assurance",
    ],
    keywords=[
        "design-patterns",
        "code-analysis",
        "architecture",
        "refactoring",
        "ai",
        "uml",
        "python",
        "mcp",
        "anthropic",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/architectai-mcp/issues",
        "Documentation": "https://github.com/yourusername/architectai-mcp/wiki",
        "Source Code": "https://github.com/yourusername/architectai-mcp",
    },
)
