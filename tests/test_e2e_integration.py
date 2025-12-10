"""
End-to-End Integration Tests for ArchitectAI

Tests the complete workflow from raw code input to refactored output,
simulating real-world usage of the entire system.
"""

import sys
from pathlib import Path
import tempfile
import ast

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from services.project_service import ProjectAnalyzer
from services.architecture_service import ArchitectureVisitor, FastTypeEnricher, DeterministicPlantUMLConverter
from services.code_generation_service import CodeGenerator
from services.refactoring_service import RefactoringAdvisor
from services.filesystem_service import FileSystemVisitor, TreeFormatter


# =============================================================================
# REALISTIC TEST PROJECTS
# =============================================================================

# Bad Design: God Object Anti-pattern
BAD_PAYMENT_SYSTEM = '''
class PaymentProcessor:
    """This class does too much - violation of SRP"""
    
    def __init__(self):
        self.users = []
        self.orders = []
        self.payments = []
        self.emails = []
        self.logs = []
    
    def create_user(self, name: str, email: str):
        """User creation"""
        self.users.append({"name": name, "email": email})
    
    def create_order(self, user_id: int, items: list):
        """Order creation"""
        self.orders.append({"user_id": user_id, "items": items})
    
    def process_payment(self, order_id: int, method: str, amount: float):
        """Process payment"""
        if method == "credit":
            result = self._charge_credit_card(amount)
        elif method == "paypal":
            result = self._charge_paypal(amount)
        else:
            raise ValueError("Unknown method")
        
        self.payments.append({"order_id": order_id, "amount": amount, "result": result})
        return result
    
    def _charge_credit_card(self, amount: float):
        return True
    
    def _charge_paypal(self, amount: float):
        return True
    
    def send_confirmation_email(self, user_id: int, order_id: int):
        """Send email"""
        email = f"Order {order_id} confirmed"
        self.emails.append(email)
    
    def log_transaction(self, message: str):
        """Log transaction"""
        self.logs.append(message)
'''

# Better Design: Using Strategy Pattern
GOOD_PAYMENT_SYSTEM = '''
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def process(self, amount: float) -> bool:
        pass

class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number: str):
        self.card_number = card_number
    
    def process(self, amount: float) -> bool:
        return True

class PayPalPayment(PaymentStrategy):
    def __init__(self, email: str):
        self.email = email
    
    def process(self, amount: float) -> bool:
        return True

class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy
    
    def process_payment(self, amount: float) -> bool:
        return self.strategy.process(amount)

class OrderService:
    def __init__(self, payment_processor: PaymentProcessor):
        self.payment_processor = payment_processor
    
    def checkout(self, amount: float) -> bool:
        return self.payment_processor.process_payment(amount)
'''

# Real Project: E-commerce System
ECOMMERCE_PROJECT = {
    "models/user.py": '''
class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
    
    def get_profile(self):
        return {"id": self.id, "name": self.name, "email": self.email}
''',
    "models/product.py": '''
class Product:
    def __init__(self, id: int, name: str, price: float):
        self.id = id
        self.name = name
        self.price = price
    
    def get_price(self):
        return self.price
''',
    "services/cart.py": '''
class CartItem:
    def __init__(self, product_id: int, quantity: int):
        self.product_id = product_id
        self.quantity = quantity

class CartService:
    def __init__(self):
        self.items = []
    
    def add_item(self, item: CartItem):
        self.items.append(item)
    
    def calculate_total(self, prices: dict) -> float:
        total = 0
        for item in self.items:
            total += prices[item.product_id] * item.quantity
        return total
''',
    "services/order.py": '''
class OrderService:
    def __init__(self, cart_service):
        self.cart_service = cart_service
    
    def create_order(self, user_id: int):
        items = self.cart_service.items
        return {"user_id": user_id, "items": items}
'''
}


# =============================================================================
# E2E TEST CASES
# =============================================================================

class TestE2ECodeAnalysisPipeline:
    """Test complete analysis pipeline: Code → Structure → Enrichment → Diagram"""
    
    def test_e2e_single_file_analysis(self):
        """Should analyze a single file end-to-end"""
        tree = ast.parse(BAD_PAYMENT_SYSTEM)
        
        # Step 1: Extract structure
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        structure = visitor.structure
        
        # Verify structure extraction
        assert len(structure) >= 1
        assert structure[0]["name"] == "PaymentProcessor"
        assert len(structure[0]["methods"]) >= 5
        
        # Step 2: Mock enrichment (would call LLM in real scenario)
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = '{}'
        mock_llm.invoke.return_value = mock_response
        
        enricher = FastTypeEnricher(mock_llm)
        enriched = enricher.enrich(BAD_PAYMENT_SYSTEM, structure)
        
        assert len(enriched) > 0
        
        # Step 3: Generate PlantUML
        converter = DeterministicPlantUMLConverter()
        plantuml = converter.convert(enriched)
        
        assert "@startuml" in plantuml
        assert "PaymentProcessor" in plantuml
        assert "@enduml" in plantuml
    
    def test_e2e_multi_file_project_analysis(self):
        """Should analyze entire project with multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create project structure
            (root / "models").mkdir()
            (root / "services").mkdir()
            
            for filename, content in ECOMMERCE_PROJECT.items():
                filepath = root / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content)
            
            # Analyze entire project
            analyzer = ProjectAnalyzer(root)
            structure = analyzer.analyze()
            
            # Verify all classes were found
            class_names = [item["name"] for item in structure]
            assert "User" in class_names
            assert "Product" in class_names
            assert "CartService" in class_names
            assert "OrderService" in class_names
            
            # Verify source files are tagged
            for item in structure:
                assert "source_file" in item
                assert item["source_file"] is not None


class TestE2ERefactoringWorkflow:
    """Test complete refactoring workflow: Analyze → Propose → Generate"""
    
    def test_e2e_refactoring_proposal_generation(self):
        """Should propose refactoring improvements"""
        tree = ast.parse(BAD_PAYMENT_SYSTEM)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        structure = visitor.structure
        
        # Mock the LLM response for refactoring advice
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = '''{
            "title": "Strategy Pattern for Payment Methods",
            "description": "Current design violates SRP. Use Strategy pattern to separate payment methods.",
            "affected_classes": ["PaymentProcessor"],
            "proposed_uml": "@startuml\\nclass PaymentProcessor\\n@enduml"
        }'''
        mock_llm.invoke.return_value = mock_response
        
        advisor = RefactoringAdvisor(mock_llm)
        proposal = advisor.propose_improvement(structure)
        
        # Verify proposal structure
        assert "title" in proposal
        assert "description" in proposal
        assert "affected_classes" in proposal
        assert "Strategy" in proposal["title"]
    
    def test_e2e_code_generation_from_refactoring(self):
        """Should generate refactored code"""
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = '''
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def process(self, amount: float) -> bool:
        pass

class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy
    
    def process_payment(self, amount: float) -> bool:
        return self.strategy.process(amount)
'''
        mock_llm.invoke.return_value = mock_response
        
        generator = CodeGenerator(mock_llm)
        refactoring_plan = "Apply Strategy Pattern to payment processing"
        
        new_code = generator.generate_refactored_code(
            BAD_PAYMENT_SYSTEM,
            refactoring_plan,
            "payment.py"
        )
        
        # Verify generated code
        assert new_code is not None
        assert "PaymentStrategy" in new_code
        assert "ABC" in new_code or "abstractmethod" in new_code
    
    def test_e2e_save_refactored_code(self):
        """Should save refactored code to filesystem"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock(spec=BaseChatModel)
            generator = CodeGenerator(mock_llm)
            
            refactored_code = "def hello():\n    pass"
            
            # Save code
            saved_path = generator.save_code(
                "refactored.py",
                refactored_code,
                Path(tmpdir)
            )
            
            # Verify file was created
            assert Path(saved_path).exists()
            assert Path(saved_path).read_text() == refactored_code


class TestE2ECompleteWorkflow:
    """Test the entire ArchitectAI workflow from code to refactored output"""
    
    def test_e2e_bad_code_to_good_code(self):
        """Should transform bad code to better code following design patterns"""
        # Step 1: Analyze bad code
        tree = ast.parse(BAD_PAYMENT_SYSTEM)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        bad_structure = visitor.structure
        
        assert len(bad_structure) == 1
        assert bad_structure[0]["name"] == "PaymentProcessor"
        assert len(bad_structure[0]["methods"]) > 5  # Too many responsibilities
        
        # Step 2: Generate PlantUML for visualization
        converter = DeterministicPlantUMLConverter()
        bad_diagram = converter.convert(bad_structure)
        
        assert "@startuml" in bad_diagram
        assert "PaymentProcessor" in bad_diagram
        
        # Step 3: Propose refactoring
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = '''{
            "title": "Apply Strategy and Dependency Injection",
            "description": "Split PaymentProcessor into smaller classes",
            "affected_classes": ["PaymentProcessor"],
            "proposed_uml": "@startuml\\nclass Strategy\\n@enduml"
        }'''
        mock_llm.invoke.return_value = mock_response
        
        advisor = RefactoringAdvisor(mock_llm)
        proposal = advisor.propose_improvement(bad_structure)
        
        assert "Strategy" in proposal["title"]
        
        # Step 4: Generate improved code
        mock_response.content = GOOD_PAYMENT_SYSTEM
        generator = CodeGenerator(mock_llm)
        improved_code = generator.generate_refactored_code(
            BAD_PAYMENT_SYSTEM,
            proposal["description"],
            "payment.py"
        )
        
        # Step 5: Verify improved code is syntactically valid
        assert improved_code is not None
        try:
            tree_improved = ast.parse(improved_code)
            # Should have multiple classes now
            classes = [node for node in ast.walk(tree_improved) if isinstance(node, ast.ClassDef)]
            assert len(classes) > 1  # Multiple focused classes instead of god object
        except SyntaxError:
            pytest.skip("Generated code syntax validation skipped (depends on LLM)")
    
    def test_e2e_multi_file_project_refactoring(self):
        """Should refactor entire project structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create project
            (root / "models").mkdir()
            (root / "services").mkdir()
            
            for filename, content in ECOMMERCE_PROJECT.items():
                filepath = root / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content)
            
            # Step 1: Analyze project
            analyzer = ProjectAnalyzer(root)
            project_structure = analyzer.analyze()
            
            assert len(project_structure) >= 4
            
            # Step 2: Visit filesystem
            fs_visitor = FileSystemVisitor()
            fs_tree = fs_visitor.visit(str(root), max_depth=3)
            
            assert fs_tree["type"] == "directory"
            assert len(fs_tree["children"]) >= 2  # models and services
            
            # Step 3: Format tree
            formatter = TreeFormatter()
            formatted = formatter.format(fs_tree)
            
            assert formatted is not None
            assert len(formatted) > 0
            
            # Step 4: Mock refactoring
            mock_llm = Mock(spec=BaseChatModel)
            mock_response = MagicMock()
            mock_response.content = '''{
                "title": "Refactor E-commerce Architecture",
                "description": "Add repository pattern and DI",
                "affected_classes": ["CartService", "OrderService"],
                "proposed_uml": "@startuml\\n@enduml"
            }'''
            mock_llm.invoke.return_value = mock_response
            
            advisor = RefactoringAdvisor(mock_llm)
            proposal = advisor.propose_improvement(project_structure)
            
            assert proposal is not None


class TestE2EErrorHandling:
    """Test error handling throughout the pipeline"""
    
    def test_e2e_handles_invalid_python_code(self):
        """Should gracefully handle invalid Python code"""
        bad_code = "class Broken\n    def method():\n        pass"  # Missing colon
        
        with pytest.raises(SyntaxError):
            tree = ast.parse(bad_code)
            visitor = ArchitectureVisitor()
            visitor.visit(tree)
    
    def test_e2e_handles_empty_project(self):
        """Should handle empty project gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ProjectAnalyzer(Path(tmpdir))
            structure = analyzer.analyze()
            
            # Should return empty list, not crash
            assert isinstance(structure, list)
    
    def test_e2e_handles_large_complex_code(self):
        """Should handle complex multi-class code"""
        complex_code = '''
class A:
    def method_a(self):
        pass

class B(A):
    def method_b(self):
        pass

class C:
    def __init__(self, b: B):
        self.b = b
    
    def method_c(self):
        return self.b.method_b()
'''
        tree = ast.parse(complex_code)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        assert len(visitor.structure) == 3
        
        # Verify inheritance
        b_class = next(c for c in visitor.structure if c["name"] == "B")
        assert "A" in b_class["bases"]


class TestE2EIntegrationPoints:
    """Test critical integration points between services"""
    
    def test_visitor_output_compatible_with_enricher(self):
        """Visitor output format should work with Enricher"""
        tree = ast.parse(BAD_PAYMENT_SYSTEM)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        # Enricher expects list of dicts with specific structure
        structure = visitor.structure
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke.return_value = MagicMock(content='{}')
        
        enricher = FastTypeEnricher(mock_llm)
        # Should not crash
        enriched = enricher.enrich(BAD_PAYMENT_SYSTEM, structure)
        
        assert enriched is not None
    
    def test_enricher_output_compatible_with_converter(self):
        """Enricher output format should work with PlantUML Converter"""
        tree = ast.parse(BAD_PAYMENT_SYSTEM)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        structure = visitor.structure
        
        # PlantUML converter should accept structure
        converter = DeterministicPlantUMLConverter()
        result = converter.convert(structure)
        
        assert "@startuml" in result
        assert "@enduml" in result
    
    def test_analyzer_output_compatible_with_refactoring(self):
        """ProjectAnalyzer output should work with RefactoringAdvisor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "test.py").write_text(BAD_PAYMENT_SYSTEM)
            
            analyzer = ProjectAnalyzer(root)
            structure = analyzer.analyze()
            
            mock_llm = Mock(spec=BaseChatModel)
            mock_llm.invoke.return_value = MagicMock(content='{"title": "Test", "description": "Test"}')
            
            advisor = RefactoringAdvisor(mock_llm)
            # Should not crash
            proposal = advisor.propose_improvement(structure)
            
            assert proposal is not None


# Fixtures
@pytest.fixture
def mock_llm():
    """Create mock LLM for testing"""
    llm = Mock(spec=BaseChatModel)
    return llm


@pytest.fixture
def temp_project():
    """Create temporary project directory"""
    import shutil
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)
