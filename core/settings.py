"""
Application settings and configuration.
"""
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, Field, ConfigDict
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
    sambanova_api_key: str | None = Field(None, validation_alias="SAMBANOVA_API_KEY")
    nebius_api_key: str | None = Field(None, validation_alias="NEBIUS_API_KEY")
    
    # LLM Configuration
    MODEL_NAME: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.0
    ENV: str = Field("development", validation_alias="ENV")
    
    # Nebius LLM Configuration
    NEBIUS_API_KEY: str | None = Field(None, validation_alias="NEBIUS_API_KEY")
    NEBIUS_ENDPOINT: str | None = Field(None, validation_alias="NEBIUS_ENDPOINT")
    
    # SambaNova LLM Configuration
    SAMBANOVA_API_KEY: str | None = Field(None, validation_alias="SAMBANOVA_API_KEY")
    SAMBANOVA_ENDPOINT: str | None = Field(None, validation_alias="SAMBANOVA_ENDPOINT")
    
    # OpenAI LLM Configuration
    OPENAI_API_KEY: str | None = Field(None, validation_alias="OPENAI_API_KEY")
    # Gemini LLM Configuration
    GEMINI_API_KEY: str | None = Field(None, validation_alias="GEMINI_API_KEY")

    # ========== MCP Configuration ==========
    # GitHub MCP
    GITHUB_TOKEN: str | None = Field(None, validation_alias="GITHUB_TOKEN")
    
    # PostgreSQL MCP
    POSTGRES_CONNECTION_STRING: str | None = Field(None, validation_alias="POSTGRES_CONNECTION_STRING")
    
    # Filesystem MCP
    MCP_ALLOWED_DIRECTORIES: str = Field(".", validation_alias="MCP_ALLOWED_DIRECTORIES")
    
    model_config = ConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False
        )
# Create settings instance
settings = Settings()