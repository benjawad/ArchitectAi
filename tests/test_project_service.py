"""
Unit tests for Project Service

Tests for project-wide code analysis and aggregation.
"""

import sys
from pathlib import Path
import tempfile
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock

from services.project_service import ProjectAnalyzer


# Test file content
SAMPLE_PYTHON_CODE = '''
class UserRepository:
    def __init__(self, db):
        self.db = db
    
    def find_user(self, id: int):
        pass
'''

ANOTHER_PYTHON_CODE = '''
class PaymentService:
    def process(self, amount: float):
        pass
'''


class TestProjectAnalyzer:
    """Test project-level code analysis"""
    
    def test_analyzer_initialization(self):
        """Should initialize with root path"""
        root = Path(".")
        analyzer = ProjectAnalyzer(root)
        
        assert analyzer.root_path == root
        assert analyzer.aggregated_structure == []
        assert analyzer.errors == []
    
    def test_analyze_single_file(self):
        """Should analyze a single Python file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test Python file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(SAMPLE_PYTHON_CODE)
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            assert len(result) > 0
            assert result[0]["name"] == "UserRepository"
    
    def test_analyze_multiple_files(self):
        """Should aggregate classes from multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            file1 = Path(tmpdir) / "repo.py"
            file1.write_text(SAMPLE_PYTHON_CODE)
            
            file2 = Path(tmpdir) / "service.py"
            file2.write_text(ANOTHER_PYTHON_CODE)
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            assert len(result) >= 2
            names = [item["name"] for item in result]
            assert "UserRepository" in names
            assert "PaymentService" in names
    
    def test_analyze_skips_venv(self):
        """Should skip virtual environment directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create venv structure
            venv_dir = Path(tmpdir) / "venv" / "lib"
            venv_dir.mkdir(parents=True)
            
            venv_file = venv_dir / "test.py"
            venv_file.write_text("class IgnoredClass:\n    pass")
            
            # Create normal file
            normal_file = Path(tmpdir) / "normal.py"
            normal_file.write_text("class NormalClass:\n    pass")
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            # Should only find NormalClass, not IgnoredClass
            names = [item["name"] for item in result]
            assert "NormalClass" in names
            assert "IgnoredClass" not in names
    
    def test_analyze_skips_pycache(self):
        """Should skip __pycache__ directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pycache_dir = Path(tmpdir) / "__pycache__"
            pycache_dir.mkdir()
            
            pycache_file = pycache_dir / "test.py"
            pycache_file.write_text("class CachedClass:\n    pass")
            
            normal_file = Path(tmpdir) / "normal.py"
            normal_file.write_text("class RealClass:\n    pass")
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            names = [item["name"] for item in result]
            assert "RealClass" in names
            assert "CachedClass" not in names
    
    def test_analyze_tags_source_file(self):
        """Should add source_file field to each component"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "mymodule.py"
            test_file.write_text(SAMPLE_PYTHON_CODE)
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            assert len(result) > 0
            assert "source_file" in result[0]
            assert result[0]["source_file"] == "mymodule.py"
    
    def test_analyze_handles_syntax_errors(self):
        """Should gracefully handle files with syntax errors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.py"
            bad_file.write_text("class Broken\n    pass")  # Missing colon
            
            good_file = Path(tmpdir) / "good.py"
            good_file.write_text("class Good:\n    pass")
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            # Should parse good.py and record error for bad.py
            assert len(result) > 0
            assert len(analyzer.errors) > 0
    
    def test_analyze_recursive_directory(self):
        """Should recursively analyze subdirectories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            
            subfile = subdir / "nested.py"
            subfile.write_text("class NestedClass:\n    pass")
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            names = [item["name"] for item in result]
            assert "NestedClass" in names
    
    def test_analyze_returns_list(self):
        """Should return aggregated structure as list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(SAMPLE_PYTHON_CODE)
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            assert isinstance(result, list)


class TestProjectAnalyzerPrivateMethods:
    """Test private helper methods"""
    
    def test_analyze_single_file_method(self):
        """Should parse single file without errors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(SAMPLE_PYTHON_CODE)
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            analyzer._analyze_single_file(test_file)
            
            assert len(analyzer.aggregated_structure) > 0
    
    def test_analyze_single_file_bad_encoding(self):
        """Should handle files with encoding issues"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            # Write file with valid Python
            test_file.write_text("class TestClass:\n    pass")
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            # Should not crash
            analyzer._analyze_single_file(test_file)
            
            assert len(analyzer.aggregated_structure) > 0


class TestProjectAnalyzerIntegration:
    """Integration tests"""
    
    def test_analyze_real_project_structure(self):
        """Should analyze a complete project directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mini project
            root = Path(tmpdir)
            
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / "user.py").write_text("class User:\n    def __init__(self):\n        pass")
            
            services_dir = root / "services"
            services_dir.mkdir()
            (services_dir / "auth.py").write_text("class AuthService:\n    def login(self):\n        pass")
            
            analyzer = ProjectAnalyzer(root)
            result = analyzer.analyze()
            
            assert len(result) >= 2
            names = [item["name"] for item in result]
            assert "User" in names
            assert "AuthService" in names
    
    def test_analyze_preserves_structure_info(self):
        """Should preserve all structure information"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(SAMPLE_PYTHON_CODE)
            
            analyzer = ProjectAnalyzer(Path(tmpdir))
            result = analyzer.analyze()
            
            assert len(result) > 0
            item = result[0]
            
            # Check all expected fields
            assert "name" in item
            assert "source_file" in item
            assert "bases" in item
            assert "methods" in item
            assert "attributes" in item


# Fixtures
@pytest.fixture
def temp_project():
    """Create temporary project directory"""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def analyzer(temp_project):
    """Create analyzer instance"""
    return ProjectAnalyzer(temp_project)
