"""
LLM Providers - Strategy Pattern Implementation
Each provider is a separate class following the Strategy Pattern.
"""
from abc import ABC, abstractmethod
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from core.settings import settings

# Try importing Gemini support
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMProvider(ABC):
    """
    Abstract Base Class for LLM Providers (Strategy Pattern).
    All providers must implement this interface.
    """
    
    @abstractmethod
    def create_llm(self, model: Optional[str] = None, temperature: float = 0.0) -> BaseChatModel:
        """
        Create and return an LLM instance.
        
        Args:
            model: Model identifier (uses default if None)
            temperature: Temperature setting
            
        Returns:
            Configured LLM instance
        """
        pass
    
    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate that provider configuration is complete."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass


class NebiusProvider(LLMProvider):
    """Nebius LLM Provider Implementation."""
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize Nebius provider.
        
        Args:
            api_key: API key (defaults to settings.NEBIUS_API_KEY)
            endpoint: API endpoint (defaults to settings.NEBIUS_ENDPOINT)
        """
        self.api_key = api_key or settings.NEBIUS_API_KEY
        self.endpoint = endpoint or settings.NEBIUS_ENDPOINT
        self.validate_configuration()
    
    @property
    def default_model(self) -> str:
        return "moonshotai/Kimi-K2-Instruct"
    
    def validate_configuration(self) -> None:
        """Validate Nebius configuration."""
        if not self.api_key or not self.endpoint:
            raise RuntimeError(
                "Nebius configuration incomplete. "
                "Set NEBIUS_API_KEY and NEBIUS_ENDPOINT in your .env file."
            )
    
    def create_llm(self, model: Optional[str] = None, temperature: float = 0.0) -> ChatOpenAI:
        """
        Create Nebius LLM instance with timeout protection.
        
        Includes:
        - Timeout protection (60 seconds)
        - Automatic retry on transient errors (max 3 attempts)
        """
        return ChatOpenAI(
            base_url=str(self.endpoint),
            api_key=self.api_key,
            model=model or self.default_model,
            temperature=temperature,
            request_timeout=60.0,  # 60 second timeout
            max_retries=3,  # Retry up to 3 times on transient errors
        )


class SambanovaProvider(LLMProvider):
    """SambaNova LLM Provider Implementation."""
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize SambaNova provider.
        
        Args:
            api_key: API key (defaults to settings.SAMBANOVA_API_KEY)
            endpoint: API endpoint (defaults to settings.SAMBANOVA_ENDPOINT)
        """
        self.api_key = api_key or settings.SAMBANOVA_API_KEY
        self.endpoint = endpoint or settings.SAMBANOVA_ENDPOINT
        self.validate_configuration()
    
    @property
    def default_model(self) -> str:
        return "Llama-4-Maverick-17B-128E-Instruct"
    
    def validate_configuration(self) -> None:
        """Validate SambaNova configuration."""
        if not self.api_key or not self.endpoint:
            raise RuntimeError(
                "SambaNova configuration incomplete. "
                "Set SAMBANOVA_API_KEY and SAMBANOVA_ENDPOINT in your .env file."
            )
    
    def create_llm(self, model: Optional[str] = None, temperature: float = 0.0) -> ChatOpenAI:
        """
        Create SambaNova LLM instance with rate limit handling.
        
        Includes:
        - Timeout protection (60 seconds)
        - Automatic retry on rate limits (max 3 attempts)
        - Exponential backoff between retries
        """
        return ChatOpenAI(
            base_url=str(self.endpoint),
            api_key=self.api_key,
            model=model or self.default_model,
            temperature=temperature,
            request_timeout=60.0,  # 60 second timeout
            max_retries=3,  # Retry up to 3 times on rate limits
        )


class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider Implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: API key (defaults to settings.OPENAI_API_KEY)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.validate_configuration()
    
    @property
    def default_model(self) -> str:
        return "gpt-5.1"
    
    def validate_configuration(self) -> None:
        """Validate OpenAI configuration."""
        if not self.api_key:
            raise RuntimeError(
                "OpenAI configuration incomplete. "
                "Set OPENAI_API_KEY in your .env file."
            )
    
    def create_llm(self, model: Optional[str] = None, temperature: float = 0.0) -> ChatOpenAI:
        """
        Create OpenAI LLM instance with timeout and retry protection.

        Includes:
        - Timeout protection (60 seconds)
        - Automatic retry on rate limits (max 3 attempts)
        """
        return ChatOpenAI(
            api_key=self.api_key,
            model=model or self.default_model,
            temperature=temperature,
            request_timeout=60.0,  # 60 second timeout
            max_retries=3,  # Retry up to 3 times on rate limits
        )


class GeminiProvider(LLMProvider):
    """Google Gemini LLM Provider Implementation."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini provider.

        Args:
            api_key: API key (defaults to settings.GEMINI_API_KEY)
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError(
                "Gemini support not available. "
                "Install with: pip install langchain-google-genai"
            )
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.validate_configuration()

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    def validate_configuration(self) -> None:
        """Validate Gemini configuration."""
        if not self.api_key:
            raise RuntimeError(
                "Gemini configuration incomplete. "
                "Set GEMINI_API_KEY in your .env file."
            )

    def create_llm(self, model: Optional[str] = None, temperature: float = 0.0) -> BaseChatModel:
        """
        Create Gemini LLM instance with timeout and retry protection.

        Includes:
        - Timeout protection (60 seconds)
        - Automatic retry on rate limits (max 3 attempts)
        """
        return ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=model or self.default_model,
            temperature=temperature,
            request_timeout=60.0,
            max_retries=3,
        )
