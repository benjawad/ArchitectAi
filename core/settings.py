"""
Application settings and configuration.
"""
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, Field
from typing import List
import os
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ========== FastAPI Settings ==========
    APP_NAME: str = "Memory Chatbot API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "A chatbot with long-term memory capabilities"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    
    # ========== LLM Settings (from your config.py) ==========
    # GROQ_API_KEY: str | None = Field(None, env="GROQ_API_KEY")
    # OPENAI_API_KEY: str | None = Field(None, env="OPENAI_API_KEY")
    # OPENAI_ENDPOINT: AnyHttpUrl | None = Field(None, env="OPENAI_ENDPOINT")
    # LANGSMITH_API_KEY: str | None = Field(None, env="LANGSMITH_API_KEY")
    sambanova_api_key: str | None = Field(None, env="SAMBANOVA_API_KEY")
    nebius_api_key: str | None = Field(None, env="NEBIUS_API_KEY")
    
    # LLM Configuration
    MODEL_NAME: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.0
    ENV: str = Field("development", env="ENV")
    
    # Nebius LLM Configuration
    NEBIUS_API_KEY: str | None = Field(None, env="NEBIUS_API_KEY")
    NEBIUS_ENDPOINT: str | None = Field(None, env="NEBIUS_ENDPOINT")
    
    # SambaNova LLM Configuration
    SAMBANOVA_API_KEY: str | None = Field(None, env="SAMBANOVA_API_KEY")
    SAMBANOVA_ENDPOINT: str | None = Field(None, env="SAMBANOVA_ENDPOINT")
    
    # OpenAI LLM Configuration
    OPENAI_API_KEY: str | None = Field(None, env="OPENAI_API_KEY")
    
    # ========== MCP Configuration ==========
    # GitHub MCP
    GITHUB_TOKEN: str | None = Field(None, env="GITHUB_TOKEN")
    
    # PostgreSQL MCP
    POSTGRES_CONNECTION_STRING: str | None = Field(None, env="POSTGRES_CONNECTION_STRING")
    
    # Filesystem MCP
    MCP_ALLOWED_DIRECTORIES: str = Field(".", env="MCP_ALLOWED_DIRECTORIES")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # Allow both GROQ_API_KEY and groq_api_key

# Create settings instance
settings = Settings()