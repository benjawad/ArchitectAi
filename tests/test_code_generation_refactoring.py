"""
Unit tests for Code Generation and Refactoring Services

Tests for code generation, refactoring proposals, and code transformations.
"""

import sys
from pathlib import Path
import tempfile

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel

from services.code_generation_service import CodeGenerator
from services.refactoring_service import RefactoringAdvisor


SAMPLE_CODE = '''
def process_payment(method, amount):
    if method == "credit":
        print(f"Credit: {amount}")
    elif method == "paypal":
        print(f"PayPal: {amount}")
    else:
        print("Unknown method")
'''

SAMPLE_STRUCTURE = [
    {
        "name": "PaymentProcessor",
        "type": "class",
        "bases": [],
        "methods": [
            {"name": "process", "args": ["amount"]},
            {"name": "validate", "args": ["amount"]}
        ],
        "attributes": [
            {"name": "gateway", "type": "PaymentGateway"}
        ]
    },
    {
        "name": "OrderService",
        "type": "class",
        "bases": [],
        "methods": [
            {"name": "create_order", "args": ["items"]},
            {"name": "checkout", "args": ["amount"]}
        ],
        "attributes": []
    }
]


class TestCodeGenerator:
    """Test code generation functionality"""
    
    def test_code_generator_initialization(self):
        """Should initialize with LLM"""
        mock_llm = Mock(spec=BaseChatModel)
        generator = CodeGenerator(mock_llm)
        
        assert generator.llm is mock_llm
    
    @patch('services.code_generation_service.create_openai_llm')
    def test_code_generator_default_llm(self, mock_create):
        """Should use default OpenAI LLM if none provided"""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_create.return_value = mock_llm
        
        generator = CodeGenerator()
        
        assert generator.llm is not None
    
    def test_generate_refactored_code(self):
        """Should generate refactored code"""
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = "def new_function():\n    pass"
        mock_llm.invoke.return_value = mock_response
        
        generator = CodeGenerator(mock_llm)
        result = generator.generate_refactored_code(
            SAMPLE_CODE,
            "Apply Factory Pattern",
            "payment.py"
        )
        
        assert result is not None
        assert "new_function" in result
    
    def test_clean_output_python_markdown(self):
        """Should clean Python markdown formatting"""
        mock_llm = Mock(spec=BaseChatModel)
        generator = CodeGenerator(mock_llm)
        
        dirty = "```python\ndef hello():\n    pass\n```"
        result = generator._clean_output(dirty)
        
        assert "```" not in result
        assert "def hello" in result
    
    def test_clean_output_generic_markdown(self):
        """Should clean generic markdown formatting"""
        mock_llm = Mock(spec=BaseChatModel)
        generator = CodeGenerator(mock_llm)
        
        dirty = "```\ndef test():\n    pass\n```"
        result = generator._clean_output(dirty)
        
        assert "```" not in result
    
    def test_clean_output_no_markdown(self):
        """Should handle code without markdown"""
        mock_llm = Mock(spec=BaseChatModel)
        generator = CodeGenerator(mock_llm)
        
        clean = "def test():\n    pass"
        result = generator._clean_output(clean)
        
        assert result == clean
    
    def test_save_code_creates_directories(self):
        """Should create parent directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock(spec=BaseChatModel)
            generator = CodeGenerator(mock_llm)
            
            code = "print('hello')"
            result = generator.save_code(
                "src/module/test.py",
                code,
                Path(tmpdir)
            )
            
            assert Path(result).exists()
            assert Path(result).read_text() == code
    
    def test_save_code_overwrites_existing(self):
        """Should overwrite existing files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock(spec=BaseChatModel)
            generator = CodeGenerator(mock_llm)
            
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("old code")
            
            new_code = "new code"
            generator.save_code("test.py", new_code, Path(tmpdir))
            
            assert file_path.read_text() == new_code
    
    def test_save_code_returns_path(self):
        """Should return saved file path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock(spec=BaseChatModel)
            generator = CodeGenerator(mock_llm)
            
            result = generator.save_code(
                "test.py",
                "code",
                Path(tmpdir)
            )
            
            assert isinstance(result, str)
            assert "test.py" in result


class TestRefactoringAdvisor:
    """Test refactoring proposal generation"""
    
    def test_refactoring_advisor_initialization(self):
        """Should initialize with LLM"""
        mock_llm = Mock(spec=BaseChatModel)
        advisor = RefactoringAdvisor(mock_llm)
        
        assert advisor.llm is mock_llm
    
    @patch('services.refactoring_service.create_openai_llm')
    def test_refactoring_advisor_default_llm(self, mock_create):
        """Should use default OpenAI LLM if none provided"""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_create.return_value = mock_llm
        
        advisor = RefactoringAdvisor()
        
        assert advisor.llm is not None
    
    def test_propose_improvement(self):
        """Should propose architectural improvements"""
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = '''{
            "title": "Factory Pattern",
            "description": "Refactor to use Factory",
            "affected_classes": ["PaymentProcessor"],
            "proposed_uml": "@startuml\\nclass Factory\\n@enduml"
        }'''
        mock_llm.invoke.return_value = mock_response
        
        advisor = RefactoringAdvisor(mock_llm)
        result = advisor.propose_improvement(SAMPLE_STRUCTURE)
        
        assert "title" in result
        assert "description" in result
        assert result["title"] == "Factory Pattern"
    
    def test_propose_improvement_handles_error(self):
        """Should handle LLM errors gracefully"""
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        advisor = RefactoringAdvisor(mock_llm)
        result = advisor.propose_improvement(SAMPLE_STRUCTURE)
        
        assert "error" in result
    
    def test_summarize_for_llm(self):
        """Should create summary for LLM"""
        mock_llm = Mock(spec=BaseChatModel)
        advisor = RefactoringAdvisor(mock_llm)
        
        summary = advisor._summarize_for_llm(SAMPLE_STRUCTURE)
        
        assert isinstance(summary, list)
        assert len(summary) > 0
        assert "class" in summary[0]
        assert "methods" in summary[0]
    
    def test_summarize_skips_module_functions(self):
        """Should skip Module_Functions in summary"""
        mock_llm = Mock(spec=BaseChatModel)
        advisor = RefactoringAdvisor(mock_llm)
        
        structure = SAMPLE_STRUCTURE + [
            {
                "name": "Module_Functions",
                "type": "module",
                "bases": [],
                "methods": [],
                "attributes": []
            }
        ]
        
        summary = advisor._summarize_for_llm(structure)
        
        # Module_Functions should be skipped
        names = [item["class"] for item in summary]
        assert "Module_Functions" not in names
    
    def test_clean_json_output_with_backticks(self):
        """Should clean JSON with markdown backticks"""
        mock_llm = Mock(spec=BaseChatModel)
        advisor = RefactoringAdvisor(mock_llm)
        
        dirty = '```json\n{"key": "value"}\n```'
        result = advisor._clean_json_output(dirty)
        
        assert "```" not in result
        assert '{"key": "value"}' in result
    
    def test_clean_json_output_generic_backticks(self):
        """Should clean JSON with generic backticks"""
        mock_llm = Mock(spec=BaseChatModel)
        advisor = RefactoringAdvisor(mock_llm)
        
        dirty = '```\n{"key": "value"}\n```'
        result = advisor._clean_json_output(dirty)
        
        assert "```" not in result


class TestCodeGenerationIntegration:
    """Integration tests for code generation"""
    
    def test_generate_and_save_workflow(self):
        """Should generate code and save to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock(spec=BaseChatModel)
            mock_response = MagicMock()
            mock_response.content = "def new_function():\n    pass"
            mock_llm.invoke.return_value = mock_response
            
            generator = CodeGenerator(mock_llm)
            
            code = generator.generate_refactored_code(
                SAMPLE_CODE,
                "Refactor",
                "test.py"
            )
            
            path = generator.save_code("test.py", code, Path(tmpdir))
            
            assert Path(path).exists()
            assert Path(path).read_text() == code


class TestRefactoringIntegration:
    """Integration tests for refactoring"""
    
    def test_analyze_and_propose_workflow(self):
        """Should analyze structure and propose improvements"""
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = '''{
            "title": "Strategy Pattern",
            "description": "Use Strategy for payment processing",
            "affected_classes": ["PaymentProcessor"],
            "proposed_uml": "@startuml\\n@enduml"
        }'''
        mock_llm.invoke.return_value = mock_response
        
        advisor = RefactoringAdvisor(mock_llm)
        result = advisor.propose_improvement(SAMPLE_STRUCTURE)
        
        assert result is not None
        assert "title" in result


# Fixtures
@pytest.fixture
def mock_llm():
    """Create mock LLM"""
    return Mock(spec=BaseChatModel)


@pytest.fixture
def code_generator(mock_llm):
    """Create CodeGenerator instance"""
    return CodeGenerator(mock_llm)


@pytest.fixture
def refactoring_advisor(mock_llm):
    """Create RefactoringAdvisor instance"""
    return RefactoringAdvisor(mock_llm)
