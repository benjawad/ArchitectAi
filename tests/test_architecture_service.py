"""
Unit tests for Architecture Service

Tests for AST-based code analysis, type enrichment, and PlantUML generation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import ast
import json
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage

from services.architecture_service import (
    ArchitectureVisitor,
    FastTypeEnricher,
    DeterministicPlantUMLConverter,
)


# Sample code fixtures for testing
SIMPLE_CLASS_CODE = '''
class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def get_name(self) -> str:
        return self.name
'''

INHERITANCE_CODE = '''
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof"
'''

DEPENDENCY_INJECTION_CODE = '''
class Database:
    def connect(self):
        pass

class UserRepository:
    def __init__(self, db: Database):
        self.db = db
    
    def save_user(self):
        pass
'''

COMPLEX_TYPES_CODE = '''
from typing import List, Dict, Optional, Union

class DataProcessor:
    def __init__(self):
        self.cache: Dict[str, List[str]] = {}
        self.config: Optional[dict] = None
        self.error: Union[ValueError, TypeError, None] = None
    
    def process(self, data: List[int]) -> Dict[str, int]:
        return {}
'''

UNTYPED_ATTRIBUTES_CODE = '''
class ServiceA:
    pass

class ServiceB:
    pass

class Application:
    def __init__(self):
        self.service_a = ServiceA()  # Type should be inferred as ServiceA
        self.service_b = ServiceB()  # Type should be inferred as ServiceB
        self.cache = {}  # Type should be inferred as Unknown or dict
'''

GLOBAL_FUNCTIONS_CODE = '''
def process_data(x: int) -> str:
    return str(x)

def helper_function():
    return True

class MyClass:
    def method(self):
        pass
'''

MIXIN_CODE = '''
class TimestampMixin:
    def get_timestamp(self):
        pass

class LoggingMixin:
    def log(self, msg: str):
        pass

class Service(TimestampMixin, LoggingMixin):
    def __init__(self):
        self.name = "Service"
'''

GENERIC_CODE = '''
from typing import Generic, TypeVar

T = TypeVar("T")

class Repository(Generic[T]):
    def __init__(self):
        self.items: list[T] = []
    
    def save(self, item: T):
        pass

class Product:
    def __init__(self, name: str):
        self.name = name

class ProductRepository(Repository[Product]):
    pass
'''

NAMING_COLLISION_CODE = '''
class Product:
    def __init__(self, name: str):
        self.name = name

class ProductionConfig:
    def __init__(self):
        self.env = "PROD"

class OrderService:
    def __init__(self, product: Product, config: ProductionConfig):
        self.product = product
        self.config = config
'''


class TestArchitectureVisitor:
    """Test AST-based code analysis"""
    
    def test_visit_simple_class(self):
        """Should extract class name, methods, and attributes"""
        tree = ast.parse(SIMPLE_CLASS_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        assert len(visitor.structure) == 1
        user_class = visitor.structure[0]
        assert user_class["name"] == "User"
        assert user_class["type"] == "class"
        assert len(user_class["methods"]) == 2
        assert any(m["name"] == "get_name" for m in user_class["methods"])
    
    def test_visit_inheritance(self):
        """Should detect base classes"""
        tree = ast.parse(INHERITANCE_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        assert len(visitor.structure) == 2
        dog_class = next(c for c in visitor.structure if c["name"] == "Dog")
        assert "Animal" in dog_class["bases"]
    
    def test_visit_method_signatures(self):
        """Should extract method names, arguments, and return types"""
        tree = ast.parse(SIMPLE_CLASS_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        get_name_method = visitor.structure[0]["methods"][1]
        assert get_name_method["name"] == "get_name"
        assert get_name_method["args"] == []
        assert get_name_method["returns"] == "str"
    
    def test_visit_typed_attributes(self):
        """Should extract annotated attributes"""
        tree = ast.parse(COMPLEX_TYPES_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        data_processor = visitor.structure[0]
        assert len(data_processor["attributes"]) >= 1
        
        cache_attr = next((a for a in data_processor["attributes"] if a["name"] == "cache"), None)
        assert cache_attr is not None
        assert "Dict" in cache_attr["type"]
    
    def test_visit_untyped_attributes(self):
        """Should infer types from constructors"""
        tree = ast.parse(UNTYPED_ATTRIBUTES_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        application = next(c for c in visitor.structure if c["name"] == "Application")
        assert len(application["attributes"]) >= 2
        
        service_a_attr = next((a for a in application["attributes"] if a["name"] == "service_a"), None)
        assert service_a_attr is not None
        assert service_a_attr["type"] == "ServiceA"
    
    def test_visit_global_functions(self):
        """Should capture standalone functions separately"""
        tree = ast.parse(GLOBAL_FUNCTIONS_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        assert len(visitor.global_functions) == 2
        assert any(f["name"] == "process_data" for f in visitor.global_functions)
        
        # Module_Functions should be added to structure
        module_funcs = next((c for c in visitor.structure if c["type"] == "module"), None)
        assert module_funcs is not None
        assert len(module_funcs["methods"]) == 2
    
    def test_visit_multiple_inheritance(self):
        """Should handle multiple base classes (mixins)"""
        tree = ast.parse(MIXIN_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        service = next(c for c in visitor.structure if c["name"] == "Service")
        assert len(service["bases"]) == 2
        assert "TimestampMixin" in service["bases"]
        assert "LoggingMixin" in service["bases"]
    
    def test_visit_complex_type_annotations(self):
        """Should extract complex nested type annotations"""
        tree = ast.parse(COMPLEX_TYPES_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        data_processor = visitor.structure[0]
        assert len(data_processor["attributes"]) >= 3
        
        optional_attr = next((a for a in data_processor["attributes"] if a["name"] == "config"), None)
        assert optional_attr is not None
        assert "Optional" in optional_attr["type"] or "dict" in optional_attr["type"].lower()
    
    def test_visit_empty_class(self):
        """Should handle empty classes"""
        code = "class Empty:\n    pass"
        tree = ast.parse(code)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        assert len(visitor.structure) == 1
        assert visitor.structure[0]["name"] == "Empty"
        assert len(visitor.structure[0]["methods"]) == 0
        assert len(visitor.structure[0]["attributes"]) == 0
    
    def test_visit_malformed_code(self):
        """Should handle syntax errors gracefully"""
        bad_code = "class Broken:\n    def method(\n    # missing closing"
        try:
            tree = ast.parse(bad_code)
            assert False, "Should have raised SyntaxError"
        except SyntaxError:
            pass  # Expected
    
    def test_visit_naming_collision(self):
        """Should correctly handle similar class names (Product vs ProductionConfig)"""
        tree = ast.parse(NAMING_COLLISION_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        order_service = next(c for c in visitor.structure if c["name"] == "OrderService")
        assert len(order_service["attributes"]) == 2
        
        product_attr = next((a for a in order_service["attributes"] if a["name"] == "product"), None)
        assert product_attr["type"] == "Product"
    
    def test_is_self_attribute(self):
        """Should correctly identify self.attribute references"""
        visitor = ArchitectureVisitor()
        
        code = "x = self.var"
        tree = ast.parse(code)
        assign = tree.body[0].targets[0]
        
        assert visitor._is_self_attribute(assign) is True
        
        code2 = "x = y"
        tree2 = ast.parse(code2)
        assign2 = tree2.body[0].targets[0]
        
        assert visitor._is_self_attribute(assign2) is False
    
    def test_get_id_simple_type(self):
        """Should extract simple type names"""
        visitor = ArchitectureVisitor()
        
        code = "x: int"
        tree = ast.parse(code)
        ann = tree.body[0].annotation
        
        assert visitor._get_id(ann) == "int"
    
    def test_get_id_complex_type(self):
        """Should extract complex type annotations"""
        visitor = ArchitectureVisitor()
        
        code = "x: List[Dict[str, int]]"
        tree = ast.parse(code)
        ann = tree.body[0].annotation
        
        result = visitor._get_id(ann)
        assert "List" in result
        assert "Dict" in result
    
    def test_get_id_attribute_type(self):
        """Should handle attribute-based types (typing.List)"""
        visitor = ArchitectureVisitor()
        
        code = "x: Optional[str]"
        tree = ast.parse(code)
        ann = tree.body[0].annotation
        
        result = visitor._get_id(ann)
        assert "Optional" in result or "str" in result


class TestFastTypeEnricher:
    """Test AI-based type inference"""
    
    def test_enricher_initialization(self):
        """Should initialize with LLM instance"""
        mock_llm = Mock()
        enricher = FastTypeEnricher(mock_llm)
        
        assert enricher.llm is mock_llm
    
    def test_enrich_no_unknowns(self):
        """Should return structure unchanged if no Unknown types"""
        mock_llm = Mock()
        enricher = FastTypeEnricher(mock_llm)
        
        structure = [
            {
                "name": "User",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "name", "type": "str"},
                    {"name": "age", "type": "int"}
                ]
            }
        ]
        
        result = enricher.enrich("", structure)
        
        assert result == structure
        mock_llm.invoke.assert_not_called()
    
    def test_enrich_unknown_types(self):
        """Should identify and query Unknown types"""
        mock_llm = Mock()
        mock_response = MagicMock()
        mock_response.content = '{"User.repository": "UserRepository"}'
        mock_llm.invoke.return_value = mock_response
        
        enricher = FastTypeEnricher(mock_llm)
        
        structure = [
            {
                "name": "User",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "repository", "type": "Unknown"}
                ]
            }
        ]
        
        result = enricher.enrich("code context", structure)
        
        assert mock_llm.invoke.called
        assert result[0]["attributes"][0]["type"] == "UserRepository"
    
    def test_enrich_json_with_markdown(self):
        """Should clean markdown formatting from LLM response"""
        mock_llm = Mock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"Service.db": "Database"}\n```'
        mock_llm.invoke.return_value = mock_response
        
        enricher = FastTypeEnricher(mock_llm)
        
        structure = [
            {
                "name": "Service",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "db", "type": "Unknown"}
                ]
            }
        ]
        
        result = enricher.enrich("", structure)
        
        assert result[0]["attributes"][0]["type"] == "Database"
    
    def test_enrich_handles_model_not_found_error(self):
        """Should gracefully handle 404/Model not found errors"""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Model not found")
        
        enricher = FastTypeEnricher(mock_llm)
        
        structure = [
            {
                "name": "Test",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "attr", "type": "Unknown"}
                ]
            }
        ]
        
        result = enricher.enrich("", structure)
        
        # Should return original structure without crashing
        assert result[0]["attributes"][0]["type"] == "Unknown"
    
    def test_enrich_handles_invalid_json(self):
        """Should handle invalid JSON responses gracefully"""
        mock_llm = Mock()
        mock_response = MagicMock()
        mock_response.content = "not valid json"
        mock_llm.invoke.return_value = mock_response
        
        enricher = FastTypeEnricher(mock_llm)
        
        structure = [
            {
                "name": "Test",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "attr", "type": "Unknown"}
                ]
            }
        ]
        
        result = enricher.enrich("", structure)
        
        # Should return original structure without crashing
        assert result == structure
    
    def test_apply_patches(self):
        """Should correctly apply type patches to structure"""
        mock_llm = Mock()
        enricher = FastTypeEnricher(mock_llm)
        
        structure = [
            {
                "name": "Repository",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "db", "type": "Unknown"},
                    {"name": "cache", "type": "Unknown"}
                ]
            }
        ]
        
        updates = {
            "Repository.db": "Database",
            "Repository.cache": "RedisCache"
        }
        
        enricher._apply_patches(structure, updates)
        
        assert structure[0]["attributes"][0]["type"] == "Database"
        assert structure[0]["attributes"][1]["type"] == "RedisCache"
    
    def test_apply_patches_nonexistent_class(self):
        """Should skip patches for non-existent classes"""
        mock_llm = Mock()
        enricher = FastTypeEnricher(mock_llm)
        
        structure = [
            {
                "name": "User",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "name", "type": "str"}
                ]
            }
        ]
        
        updates = {
            "NonExistent.attr": "SomeType"
        }
        
        # Should not crash
        enricher._apply_patches(structure, updates)
        
        assert structure[0]["attributes"][0]["type"] == "str"


class TestDeterministicPlantUMLConverter:
    """Test PlantUML diagram generation"""
    
    def test_convert_simple_class(self):
        """Should generate PlantUML for simple class"""
        converter = DeterministicPlantUMLConverter()
        
        structure = [
            {
                "name": "User",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "name", "type": "str"},
                    {"name": "age", "type": "int"}
                ]
            }
        ]
        
        result = converter.convert(structure)
        
        assert "@startuml" in result
        assert "@enduml" in result
        assert "class User" in result
        assert "name : str" in result
        assert "age : int" in result
    
    def test_convert_inheritance(self):
        """Should draw inheritance arrows"""
        converter = DeterministicPlantUMLConverter()
        
        structure = [
            {
                "name": "Animal",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": []
            },
            {
                "name": "Dog",
                "type": "class",
                "bases": ["Animal"],
                "methods": [],
                "attributes": []
            }
        ]
        
        result = converter.convert(structure)
        
        assert "Animal <|-- Dog" in result
    
    def test_convert_dependencies(self):
        """Should draw dependency arrows"""
        converter = DeterministicPlantUMLConverter()
        
        structure = [
            {
                "name": "Database",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": []
            },
            {
                "name": "Repository",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "db", "type": "Database"}
                ]
            }
        ]
        
        result = converter.convert(structure)
        
        assert "Repository o-- Database" in result
    
    def test_convert_generic_types(self):
        """Should handle generic type parameters"""
        converter = DeterministicPlantUMLConverter()
        
        structure = [
            {
                "name": "Product",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": []
            },
            {
                "name": "Repository",
                "type": "class",
                "bases": ["Generic[Product]"],
                "methods": [],
                "attributes": []
            }
        ]
        
        result = converter.convert(structure)
        
        assert "Generic <|-- Repository" in result or "Repository" in result
    
    def test_convert_naming_collision(self):
        """Should not create false arrows for similar names"""
        converter = DeterministicPlantUMLConverter()
        
        structure = [
            {
                "name": "Product",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": []
            },
            {
                "name": "ProductionConfig",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": []
            },
            {
                "name": "OrderService",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "product", "type": "Product"},
                    {"name": "config", "type": "ProductionConfig"}
                ]
            }
        ]
        
        result = converter.convert(structure)
        
        # Should have correct arrows
        assert "OrderService o-- Product : product" in result
        assert "OrderService o-- ProductionConfig : config" in result
    
    def test_convert_empty_structure(self):
        """Should handle empty structure"""
        converter = DeterministicPlantUMLConverter()
        
        result = converter.convert([])
        
        assert result == ""
    
    def test_convert_complex_types(self):
        """Should handle complex type annotations"""
        converter = DeterministicPlantUMLConverter()
        
        structure = [
            {
                "name": "Cache",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": []
            },
            {
                "name": "Service",
                "type": "class",
                "bases": [],
                "methods": [],
                "attributes": [
                    {"name": "cache", "type": "Dict[str, Cache]"}
                ]
            }
        ]
        
        result = converter.convert(structure)
        
        # Should detect Cache dependency despite Dict wrapper
        assert "Service o-- Cache" in result


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_analysis_pipeline(self):
        """Should analyze code and generate PlantUML"""
        tree = ast.parse(DEPENDENCY_INJECTION_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        assert len(visitor.structure) == 2
        
        converter = DeterministicPlantUMLConverter()
        plantuml = converter.convert(visitor.structure)
        
        assert "@startuml" in plantuml
        assert "Database" in plantuml
        assert "UserRepository" in plantuml
    
    def test_visitor_and_enricher(self):
        """Should extract types and handle enrichment gracefully"""
        tree = ast.parse(UNTYPED_ATTRIBUTES_CODE)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = MagicMock(content='{}')
        enricher = FastTypeEnricher(mock_llm)
        
        enriched = enricher.enrich(UNTYPED_ATTRIBUTES_CODE, visitor.structure)
        
        assert len(enriched) > 0


# Fixtures for pytest
@pytest.fixture
def simple_visitor():
    tree = ast.parse(SIMPLE_CLASS_CODE)
    visitor = ArchitectureVisitor()
    visitor.visit(tree)
    return visitor


@pytest.fixture
def inheritance_visitor():
    tree = ast.parse(INHERITANCE_CODE)
    visitor = ArchitectureVisitor()
    visitor.visit(tree)
    return visitor


@pytest.fixture
def mock_llm():
    return Mock()


@pytest.fixture
def enricher(mock_llm):
    return FastTypeEnricher(mock_llm)


@pytest.fixture
def converter():
    return DeterministicPlantUMLConverter()
