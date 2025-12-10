"""
Unit tests for LLM Factory

Tests for factory pattern implementation, provider registration, 
and LLM instance creation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel

from core.llm_factory import (
    LLMFactory,
    create_nebius_llm,
    create_sambanova_llm,
    create_openai_llm,
    create_gemini_llm,
)


class TestLLMFactoryCreate:
    """Test LLMFactory.create() method"""
    
    def test_create_sambanova_provider(self):
        """Should create SambaNova LLM instance"""
        result = LLMFactory.create("sambanova")
        
        assert result is not None
        assert hasattr(result, 'invoke')  # Check it's a BaseChatModel
    
    def test_create_nebius_provider(self):
        """Should create Nebius LLM instance"""
        result = LLMFactory.create("nebius")
        
        assert result is not None
        assert hasattr(result, 'invoke')
    
    def test_create_openai_provider(self):
        """Should create OpenAI LLM instance"""
        result = LLMFactory.create("openai")
        
        assert result is not None
        assert hasattr(result, 'invoke')
    
    def test_create_gemini_provider(self):
        """Should create Gemini LLM instance"""
        result = LLMFactory.create("gemini")
        
        assert result is not None
        assert hasattr(result, 'invoke')
    
    def test_create_unknown_provider(self):
        """Should raise ValueError for unknown provider"""
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create("unknown_provider")
        
        assert "Unknown provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)
    
    def test_create_with_custom_model(self):
        """Should accept custom model parameter"""
        result = LLMFactory.create("sambanova", model="custom-model-123")
        
        assert result is not None
        assert hasattr(result, 'model_name')
    
    def test_create_with_custom_temperature(self):
        """Should accept custom temperature parameter"""
        result = LLMFactory.create("sambanova", temperature=0.7)
        
        assert result is not None
        assert result.temperature == 0.7
    
    def test_create_case_insensitive(self):
        """Should handle provider names case-insensitively"""
        # Test uppercase
        result1 = LLMFactory.create("SAMBANOVA")
        assert result1 is not None
        
        # Test mixed case
        result2 = LLMFactory.create("SambaNova")
        assert result2 is not None
    
    def test_create_with_provider_kwargs(self):
        """Should pass additional kwargs to provider"""
        # This will use custom API key and endpoint
        result = LLMFactory.create(
            "sambanova",
            api_key="custom-key",
            endpoint="https://custom.endpoint.com"
        )
        
        assert result is not None


class TestLLMFactoryRegistry:
    """Test provider registry functionality"""
    
    def test_list_providers(self):
        """Should list all registered providers"""
        providers = LLMFactory.list_providers()
        
        assert isinstance(providers, list)
        assert "sambanova" in providers
        assert "nebius" in providers
        assert "openai" in providers
        assert "gemini" in providers
    
    def test_register_custom_provider(self):
        """Should allow registering custom providers"""
        from core.llm_providers import LLMProvider
        
        class CustomProvider(LLMProvider):
            def create_llm(self, model=None, temperature=0.0):
                return MagicMock(spec=BaseChatModel)
            
            def validate_configuration(self):
                pass
            
            @property
            def default_model(self):
                return "custom-model"
        
        LLMFactory.register_provider("custom", CustomProvider)
        
        assert "custom" in LLMFactory.list_providers()
    
    def test_register_provider_overwrites_existing(self):
        """Should allow overwriting existing providers"""
        from core.llm_providers import LLMProvider, SambanovaProvider
        
        original_sambanova = LLMFactory._providers["sambanova"]
        
        class NewSambanovaProvider(LLMProvider):
            def create_llm(self, model=None, temperature=0.0):
                return MagicMock(spec=BaseChatModel)
            
            def validate_configuration(self):
                pass
            
            @property
            def default_model(self):
                return "new-model"
        
        LLMFactory.register_provider("sambanova", NewSambanovaProvider)
        
        assert LLMFactory._providers["sambanova"] == NewSambanovaProvider
        
        # Restore original
        LLMFactory._providers["sambanova"] = original_sambanova


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility"""
    
    def test_create_sambanova_llm(self):
        """Should create SambaNova LLM"""
        result = create_sambanova_llm()
        
        assert result is not None
        assert hasattr(result, 'invoke')
    
    def test_create_sambanova_llm_with_params(self):
        """Should pass parameters through to factory"""
        result = create_sambanova_llm(model="custom-model", temperature=0.5)
        
        assert result is not None
        assert result.temperature == 0.5
    
    def test_create_nebius_llm(self):
        """Should create Nebius LLM"""
        result = create_nebius_llm()
        
        assert result is not None
        assert hasattr(result, 'invoke')
    
    def test_create_openai_llm(self):
        """Should create OpenAI LLM"""
        result = create_openai_llm()
        
        assert result is not None
        assert hasattr(result, 'invoke')
    
    def test_create_gemini_llm(self):
        """Should create Gemini LLM"""
        result = create_gemini_llm()
        
        assert result is not None
        assert hasattr(result, 'invoke')


class TestFactoryErrorHandling:
    """Test error handling in factory"""
    
    def test_unknown_provider_error_message(self):
        """Should include available providers in error message"""
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create("nonexistent")
        
        error_msg = str(exc_info.value)
        assert "Available providers" in error_msg
        assert "sambanova" in error_msg.lower()
    
    def test_invalid_provider_name_format(self):
        """Should handle invalid provider names"""
        with pytest.raises(ValueError):
            LLMFactory.create("")
        
        with pytest.raises(ValueError):
            LLMFactory.create("   ")


class TestFactoryIntegration:
    """Integration tests for factory"""
    
    def test_factory_is_singleton_like(self):
        """Factory methods should be static/class methods"""
        # Verify we can call without instantiation
        assert callable(LLMFactory.create)
        assert callable(LLMFactory.list_providers)
        assert callable(LLMFactory.register_provider)
    
    def test_create_multiple_instances_independently(self):
        """Should be able to create multiple instances"""
        llm1 = LLMFactory.create("sambanova", temperature=0.0)
        llm2 = LLMFactory.create("sambanova", temperature=0.9)
        
        assert llm1 is not None
        assert llm2 is not None
        assert llm1.temperature != llm2.temperature


# Fixtures for pytest
@pytest.fixture
def reset_factory():
    """Reset factory to known state"""
    original_providers = LLMFactory._providers.copy()
    yield
    LLMFactory._providers = original_providers


@pytest.fixture
def mock_llm():
    """Create mock LLM instance"""
    return MagicMock(spec=BaseChatModel)
