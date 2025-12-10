"""
Unit tests for LLM Providers

Tests for provider implementations, configuration validation,
and LLM instance creation with proper error handling.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel

from core.llm_providers import (
    LLMProvider,
    NebiusProvider,
    SambanovaProvider,
    OpenAIProvider,
    GeminiProvider,
)


class TestLLMProviderInterface:
    """Test abstract LLMProvider interface"""
    
    def test_llm_provider_is_abstract(self):
        """Should not allow instantiating abstract LLMProvider"""
        with pytest.raises(TypeError):
            LLMProvider()
    
    def test_llm_provider_requires_create_llm(self):
        """Should require create_llm implementation"""
        class IncompleteProvider(LLMProvider):
            def validate_configuration(self):
                pass
            
            @property
            def default_model(self):
                return "model"
        
        with pytest.raises(TypeError):
            IncompleteProvider()
    
    def test_llm_provider_requires_validate_configuration(self):
        """Should require validate_configuration implementation"""
        class IncompleteProvider(LLMProvider):
            def create_llm(self, model=None, temperature=0.0):
                return MagicMock(spec=BaseChatModel)
            
            @property
            def default_model(self):
                return "model"
        
        with pytest.raises(TypeError):
            IncompleteProvider()
    
    def test_llm_provider_requires_default_model_property(self):
        """Should require default_model property implementation"""
        class IncompleteProvider(LLMProvider):
            def create_llm(self, model=None, temperature=0.0):
                return MagicMock(spec=BaseChatModel)
            
            def validate_configuration(self):
                pass
        
        with pytest.raises(TypeError):
            IncompleteProvider()


class TestNebiusProvider:
    """Test Nebius LLM Provider"""
    
    @patch('core.llm_providers.settings')
    def test_initialization_with_settings(self, mock_settings):
        """Should initialize with settings values"""
        mock_settings.NEBIUS_API_KEY = "test-key"
        mock_settings.NEBIUS_ENDPOINT = "https://nebius.com"
        
        provider = NebiusProvider()
        
        assert provider.api_key == "test-key"
        assert provider.endpoint == "https://nebius.com"
    
    @patch('core.llm_providers.settings')
    def test_initialization_with_custom_values(self, mock_settings):
        """Should accept custom API key and endpoint"""
        mock_settings.NEBIUS_API_KEY = "default-key"
        mock_settings.NEBIUS_ENDPOINT = "https://default.com"
        
        provider = NebiusProvider(
            api_key="custom-key",
            endpoint="https://custom.com"
        )
        
        assert provider.api_key == "custom-key"
        assert provider.endpoint == "https://custom.com"
    
    @patch('core.llm_providers.settings')
    def test_missing_api_key_raises_error(self, mock_settings):
        """Should raise RuntimeError if API key is missing"""
        mock_settings.NEBIUS_API_KEY = None
        mock_settings.NEBIUS_ENDPOINT = "https://nebius.com"
        
        with pytest.raises(RuntimeError) as exc_info:
            NebiusProvider()
        
        assert "NEBIUS_API_KEY" in str(exc_info.value)
    
    @patch('core.llm_providers.settings')
    def test_missing_endpoint_raises_error(self, mock_settings):
        """Should raise RuntimeError if endpoint is missing"""
        mock_settings.NEBIUS_API_KEY = "test-key"
        mock_settings.NEBIUS_ENDPOINT = None
        
        with pytest.raises(RuntimeError) as exc_info:
            NebiusProvider()
        
        assert "NEBIUS_ENDPOINT" in str(exc_info.value)
    
    @patch('core.llm_providers.settings')
    def test_default_model(self, mock_settings):
        """Should return correct default model"""
        mock_settings.NEBIUS_API_KEY = "test-key"
        mock_settings.NEBIUS_ENDPOINT = "https://nebius.com"
        
        provider = NebiusProvider()
        
        assert provider.default_model == "moonshotai/Kimi-K2-Instruct"
    
    @patch('core.llm_providers.ChatOpenAI')
    @patch('core.llm_providers.settings')
    def test_create_llm(self, mock_settings, mock_chat_openai):
        """Should create ChatOpenAI instance with correct parameters"""
        mock_settings.NEBIUS_API_KEY = "test-key"
        mock_settings.NEBIUS_ENDPOINT = "https://nebius.com"
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_chat_openai.return_value = mock_llm
        
        provider = NebiusProvider()
        result = provider.create_llm()
        
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://nebius.com"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["request_timeout"] == 60.0
        assert call_kwargs["max_retries"] == 3
        assert result is mock_llm
    
    @patch('core.llm_providers.ChatOpenAI')
    @patch('core.llm_providers.settings')
    def test_create_llm_with_custom_model(self, mock_settings, mock_chat_openai):
        """Should use custom model if provided"""
        mock_settings.NEBIUS_API_KEY = "test-key"
        mock_settings.NEBIUS_ENDPOINT = "https://nebius.com"
        mock_chat_openai.return_value = MagicMock(spec=BaseChatModel)
        
        provider = NebiusProvider()
        provider.create_llm(model="custom-model")
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == "custom-model"
    
    @patch('core.llm_providers.ChatOpenAI')
    @patch('core.llm_providers.settings')
    def test_create_llm_with_custom_temperature(self, mock_settings, mock_chat_openai):
        """Should use custom temperature if provided"""
        mock_settings.NEBIUS_API_KEY = "test-key"
        mock_settings.NEBIUS_ENDPOINT = "https://nebius.com"
        mock_chat_openai.return_value = MagicMock(spec=BaseChatModel)
        
        provider = NebiusProvider()
        provider.create_llm(temperature=0.7)
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["temperature"] == 0.7


class TestSambanovaProvider:
    """Test SambaNova LLM Provider"""
    
    @patch('core.llm_providers.settings')
    def test_initialization_with_settings(self, mock_settings):
        """Should initialize with settings values"""
        mock_settings.SAMBANOVA_API_KEY = "test-key"
        mock_settings.SAMBANOVA_ENDPOINT = "https://sambanova.com"
        
        provider = SambanovaProvider()
        
        assert provider.api_key == "test-key"
        assert provider.endpoint == "https://sambanova.com"
    
    @patch('core.llm_providers.settings')
    def test_missing_api_key_raises_error(self, mock_settings):
        """Should raise RuntimeError if API key is missing"""
        mock_settings.SAMBANOVA_API_KEY = None
        mock_settings.SAMBANOVA_ENDPOINT = "https://sambanova.com"
        
        with pytest.raises(RuntimeError) as exc_info:
            SambanovaProvider()
        
        assert "SAMBANOVA_API_KEY" in str(exc_info.value)
    
    @patch('core.llm_providers.settings')
    def test_default_model(self, mock_settings):
        """Should return correct default model"""
        mock_settings.SAMBANOVA_API_KEY = "test-key"
        mock_settings.SAMBANOVA_ENDPOINT = "https://sambanova.com"
        
        provider = SambanovaProvider()
        
        assert provider.default_model == "Llama-4-Maverick-17B-128E-Instruct"
    
    @patch('core.llm_providers.ChatOpenAI')
    @patch('core.llm_providers.settings')
    def test_create_llm(self, mock_settings, mock_chat_openai):
        """Should create ChatOpenAI instance with rate limit handling"""
        mock_settings.SAMBANOVA_API_KEY = "test-key"
        mock_settings.SAMBANOVA_ENDPOINT = "https://sambanova.com"
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_chat_openai.return_value = mock_llm
        
        provider = SambanovaProvider()
        result = provider.create_llm()
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["max_retries"] == 3
        assert call_kwargs["request_timeout"] == 60.0
        assert result is mock_llm


class TestOpenAIProvider:
    """Test OpenAI LLM Provider"""
    
    @patch('core.llm_providers.settings')
    def test_initialization_with_settings(self, mock_settings):
        """Should initialize with settings values"""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        provider = OpenAIProvider()
        
        assert provider.api_key == "test-key"
    
    @patch('core.llm_providers.settings')
    def test_initialization_with_custom_key(self, mock_settings):
        """Should accept custom API key"""
        mock_settings.OPENAI_API_KEY = "default-key"
        
        provider = OpenAIProvider(api_key="custom-key")
        
        assert provider.api_key == "custom-key"
    
    @patch('core.llm_providers.settings')
    def test_missing_api_key_raises_error(self, mock_settings):
        """Should raise RuntimeError if API key is missing"""
        mock_settings.OPENAI_API_KEY = None
        
        with pytest.raises(RuntimeError) as exc_info:
            OpenAIProvider()
        
        assert "OPENAI_API_KEY" in str(exc_info.value)
    
    @patch('core.llm_providers.settings')
    def test_default_model(self, mock_settings):
        """Should return correct default model"""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        provider = OpenAIProvider()
        
        assert provider.default_model == "gpt-5.1"
    
    @patch('core.llm_providers.ChatOpenAI')
    @patch('core.llm_providers.settings')
    def test_create_llm(self, mock_settings, mock_chat_openai):
        """Should create ChatOpenAI instance"""
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_chat_openai.return_value = mock_llm
        
        provider = OpenAIProvider()
        result = provider.create_llm()
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["request_timeout"] == 10.0
        assert call_kwargs["max_retries"] == 3
        assert result is mock_llm


class TestGeminiProvider:
    """Test Google Gemini LLM Provider"""
    
    @patch('core.llm_providers.GEMINI_AVAILABLE', True)
    @patch('core.llm_providers.settings')
    def test_initialization_with_settings(self, mock_settings):
        """Should initialize with settings values"""
        mock_settings.GEMINI_API_KEY = "test-key"
        
        provider = GeminiProvider()
        
        assert provider.api_key == "test-key"
    
    @patch('core.llm_providers.GEMINI_AVAILABLE', False)
    def test_initialization_without_gemini_library(self):
        """Should raise error if Gemini library not installed"""
        with pytest.raises(RuntimeError) as exc_info:
            GeminiProvider()
        
        assert "Gemini support not available" in str(exc_info.value)
        assert "langchain-google-genai" in str(exc_info.value)
    
    @patch('core.llm_providers.GEMINI_AVAILABLE', True)
    @patch('core.llm_providers.settings')
    def test_missing_api_key_raises_error(self, mock_settings):
        """Should raise RuntimeError if API key is missing"""
        mock_settings.GEMINI_API_KEY = None
        
        with pytest.raises(RuntimeError) as exc_info:
            GeminiProvider()
        
        assert "GEMINI_API_KEY" in str(exc_info.value)
    
    @patch('core.llm_providers.GEMINI_AVAILABLE', True)
    @patch('core.llm_providers.settings')
    def test_default_model(self, mock_settings):
        """Should return correct default model"""
        mock_settings.GEMINI_API_KEY = "test-key"
        
        provider = GeminiProvider()
        
        assert provider.default_model == "gemini-2.5-flash"
    
    @patch('core.llm_providers.ChatGoogleGenerativeAI')
    @patch('core.llm_providers.GEMINI_AVAILABLE', True)
    @patch('core.llm_providers.settings')
    def test_create_llm(self, mock_settings, mock_chat_gemini):
        """Should create ChatGoogleGenerativeAI instance"""
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_chat_gemini.return_value = mock_llm
        
        provider = GeminiProvider()
        result = provider.create_llm()
        
        call_kwargs = mock_chat_gemini.call_args[1]
        assert call_kwargs["google_api_key"] == "test-key"
        assert call_kwargs["request_timeout"] == 10.0
        assert call_kwargs["max_retries"] == 3
        assert result is mock_llm
    
    @patch('core.llm_providers.ChatGoogleGenerativeAI')
    @patch('core.llm_providers.GEMINI_AVAILABLE', True)
    @patch('core.llm_providers.settings')
    def test_create_llm_with_custom_model(self, mock_settings, mock_chat_gemini):
        """Should use custom model if provided"""
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_chat_gemini.return_value = MagicMock(spec=BaseChatModel)
        
        provider = GeminiProvider()
        provider.create_llm(model="gemini-pro")
        
        call_kwargs = mock_chat_gemini.call_args[1]
        assert call_kwargs["model"] == "gemini-pro"


class TestProviderConfiguration:
    """Test configuration validation across providers"""
    
    @patch('core.llm_providers.settings')
    def test_validate_configuration_called_on_init(self, mock_settings):
        """Should call validate_configuration during initialization"""
        mock_settings.SAMBANOVA_API_KEY = None
        mock_settings.SAMBANOVA_ENDPOINT = "https://sambanova.com"
        
        with pytest.raises(RuntimeError):
            SambanovaProvider()
    
    @patch('core.llm_providers.settings')
    def test_multiple_validation_errors_message(self, mock_settings):
        """Should include all missing config in error message"""
        mock_settings.NEBIUS_API_KEY = None
        mock_settings.NEBIUS_ENDPOINT = None
        
        with pytest.raises(RuntimeError) as exc_info:
            NebiusProvider()
        
        error_msg = str(exc_info.value)
        assert "NEBIUS_API_KEY" in error_msg
        assert "NEBIUS_ENDPOINT" in error_msg


class TestProviderIntegration:
    """Integration tests for providers"""
    
    @patch('core.llm_providers.ChatOpenAI')
    @patch('core.llm_providers.settings')
    def test_all_providers_create_llm(self, mock_settings, mock_chat_openai):
        """All providers should be able to create LLM instances"""
        mock_settings.NEBIUS_API_KEY = "nebius-key"
        mock_settings.NEBIUS_ENDPOINT = "https://nebius.com"
        mock_settings.SAMBANOVA_API_KEY = "sambanova-key"
        mock_settings.SAMBANOVA_ENDPOINT = "https://sambanova.com"
        mock_settings.OPENAI_API_KEY = "openai-key"
        mock_settings.GEMINI_API_KEY = "gemini-key"
        mock_chat_openai.return_value = MagicMock(spec=BaseChatModel)
        
        nebius = NebiusProvider()
        sambanova = SambanovaProvider()
        openai = OpenAIProvider()
        
        assert nebius.create_llm() is not None
        assert sambanova.create_llm() is not None
        assert openai.create_llm() is not None


# Fixtures for pytest
@pytest.fixture
def mock_settings():
    """Create mock settings"""
    return Mock()


@pytest.fixture
def valid_nebius_settings():
    """Create valid Nebius settings"""
    with patch('core.llm_providers.settings') as mock:
        mock.NEBIUS_API_KEY = "test-key"
        mock.NEBIUS_ENDPOINT = "https://nebius.com"
        yield mock


@pytest.fixture
def valid_sambanova_settings():
    """Create valid SambaNova settings"""
    with patch('core.llm_providers.settings') as mock:
        mock.SAMBANOVA_API_KEY = "test-key"
        mock.SAMBANOVA_ENDPOINT = "https://sambanova.com"
        yield mock
