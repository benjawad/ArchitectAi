"""
LLM Factory - Factory Pattern Implementation
Centralized factory for creating LLM instances with different providers.
"""
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

from core.llm_providers import (
    LLMProvider,
    NebiusProvider,
    SambanovaProvider,
    OpenAIProvider
)


class LLMFactory:
    """
    Factory class for creating LLM instances.
    Implements Factory Pattern for clean, extensible object creation.
    """
    
    # Registry of available providers
    _providers: dict[str, type[LLMProvider]] = {
        "nebius": NebiusProvider,
        "sambanova": SambanovaProvider,
        "openai": OpenAIProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[LLMProvider]) -> None:
        """
        Register a new LLM provider (Open/Closed Principle).
        
        Args:
            name: Provider identifier
            provider_class: Provider class implementing LLMProvider
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(
        cls,
        provider_name: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **provider_kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance using the specified provider.
        
        Args:
            provider_name: Name of the provider ("nebius", "sambanova")
            model: Optional model name (uses provider default if None)
            temperature: Temperature setting (0.0 - 1.0)
            **provider_kwargs: Additional provider-specific arguments
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If provider is not registered
            RuntimeError: If provider configuration is invalid
            
        Examples:
            >>> llm = LLMFactory.create("nebius")
            >>> llm = LLMFactory.create("sambanova", model="custom-model", temperature=0.7)
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        
        # Instantiate provider and create LLM
        provider_class = cls._providers[provider_name]
        provider = provider_class(**provider_kwargs)
        
        return provider.create_llm(model=model, temperature=temperature)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """Get list of registered provider names."""
        return list(cls._providers.keys())


# Convenience functions for backward compatibility
def create_nebius_llm(
    model: Optional[str] = None,
    temperature: float = 0.0
) -> BaseChatModel:
    """
    Create a Nebius LLM instance.
    
    Args:
        model: Model name (uses default if None)
        temperature: Temperature setting
        
    Returns:
        Configured Nebius LLM instance
    """
    return LLMFactory.create("nebius", model=model, temperature=temperature)


def create_sambanova_llm(
    model: Optional[str] = None,
    temperature: float = 0.0
) -> BaseChatModel:
    """
    Create a SambaNova LLM instance.
    
    Args:
        model: Model name (uses default if None)
        temperature: Temperature setting
        
    Returns:
        Configured SambaNova LLM instance
    """
    return LLMFactory.create("sambanova", model=model, temperature=temperature)


def create_openai_llm(
    model: Optional[str] = None,
    temperature: float = 0.0
) -> BaseChatModel:
    """
    Create an OpenAI LLM instance.
    
    Args:
        model: Model name (uses default if None)
        temperature: Temperature setting
        
    Returns:
        Configured OpenAI LLM instance
    """
    return LLMFactory.create("openai", model=model, temperature=temperature)
