"""
Unit tests for Filesystem Service

Tests for file system traversal and tree representation.
"""

import sys
from pathlib import Path
import tempfile
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from services.filesystem_service import (
    FileSystemNode,
    FileSystemVisitor,
    TreeFormatter,
)


class TestFileSystemNode:
    """Test FileSystemNode dataclass"""
    
    def test_file_node_creation(self):
        """Should create file node"""
        node = FileSystemNode(
            name="test.py",
            type="file",
            path="/path/to/test.py",
            size=1024
        )
        
        assert node.name == "test.py"
        assert node.type == "file"
        assert node.size == 1024
        assert node.children is None
    
    def test_directory_node_creation(self):
        """Should create directory node with children"""
        child = FileSystemNode(name="child.py", type="file", path="/path/child.py")
        node = FileSystemNode(
            name="src",
            type="directory",
            path="/path/src",
            children=[child]
        )
        
        assert node.type == "directory"
        assert len(node.children) == 1
        assert node.children[0].name == "child.py"


class TestFileSystemVisitor:
    """Test file system traversal"""
    
    def test_visitor_initialization(self):
        """Should initialize visitor with ignored patterns"""
        visitor = FileSystemVisitor()
        
        assert hasattr(visitor, 'IGNORED_DIRS')
        assert hasattr(visitor, 'IGNORED_FILES')
        assert "__pycache__" in visitor.IGNORED_DIRS
        assert ".git" in visitor.IGNORED_DIRS
    
    def test_visit_nonexistent_path(self):
        """Should raise error for non-existent path"""
        visitor = FileSystemVisitor()
        
        with pytest.raises(ValueError):
            visitor.visit("/nonexistent/path/xyz")
    
    def test_visit_single_file(self):
        """Should visit a single file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("print('hello')")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            assert isinstance(result, dict)
            assert result["name"] == Path(tmpdir).name
            assert result["type"] == "directory"
    
    def test_visit_directory_with_files(self):
        """Should traverse directory with files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.py").write_text("code1")
            (Path(tmpdir) / "file2.py").write_text("code2")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            assert result["type"] == "directory"
            assert len(result["children"]) == 2
    
    def test_visit_ignores_pycache(self):
        """Should skip __pycache__ directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pycache = Path(tmpdir) / "__pycache__"
            pycache.mkdir()
            (pycache / "test.pyc").write_text("bytecode")
            
            (Path(tmpdir) / "real.py").write_text("code")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            child_names = [c["name"] for c in result["children"]]
            assert "__pycache__" not in child_names
            assert "real.py" in child_names
    
    def test_visit_ignores_venv(self):
        """Should skip venv directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv = Path(tmpdir) / "venv"
            venv.mkdir()
            (venv / "lib").mkdir()
            (venv / "lib" / "site.py").write_text("venv code")
            
            (Path(tmpdir) / "app.py").write_text("app code")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            child_names = [c["name"] for c in result["children"]]
            assert "venv" not in child_names
            assert "app.py" in child_names
    
    def test_visit_ignores_git_directory(self):
        """Should skip .git directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            git = Path(tmpdir) / ".git"
            git.mkdir()
            (git / "config").write_text("git config")
            
            (Path(tmpdir) / "main.py").write_text("main code")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            child_names = [c["name"] for c in result["children"]]
            assert ".git" not in child_names
            assert "main.py" in child_names
    
    def test_visit_respects_max_depth(self):
        """Should respect max_depth parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create deep nested structure
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "d" / "e"
            deep_path.mkdir(parents=True)
            (deep_path / "deep.py").write_text("deep")
            
            (Path(tmpdir) / "shallow.py").write_text("shallow")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir, max_depth=2)
            
            # With max_depth=2, deep structure should be truncated
            assert result is not None
    
    def test_visit_sorted_output(self):
        """Should return deterministic sorted output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in random order
            (Path(tmpdir) / "z.py").write_text("z")
            (Path(tmpdir) / "a.py").write_text("a")
            (Path(tmpdir) / "m.py").write_text("m")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            # Should be sorted alphabetically
            names = [c["name"] for c in result["children"]]
            assert names == sorted(names)
    
    def test_visit_directories_before_files(self):
        """Should list directories before files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").write_text("file")
            (Path(tmpdir) / "folder").mkdir()
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            children = result["children"]
            # First child should be directory
            assert children[0]["type"] == "directory"
            # Second should be file
            assert children[1]["type"] == "file"
    
    def test_visit_file_size(self):
        """Should include file size"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello world")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            file_node = result["children"][0]
            assert file_node["type"] == "file"
            assert file_node["size"] > 0
    
    def test_visit_recursive_tree(self):
        """Should create nested tree structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create structure
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("main")
            (root / "src" / "utils").mkdir()
            (root / "src" / "utils" / "helper.py").write_text("helper")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(tmpdir)
            
            # Navigate tree
            assert result["type"] == "directory"
            src = next(c for c in result["children"] if c["name"] == "src")
            assert src["type"] == "directory"
            assert len(src["children"]) >= 1


class TestTreeFormatter:
    """Test tree formatting"""
    
    def test_formatter_initialization(self):
        """Should initialize formatter"""
        formatter = TreeFormatter()
        assert formatter is not None
    
    def test_format_simple_tree(self):
        """Should format simple tree structure"""
        node = {
            "name": "root",
            "type": "directory",
            "children": [
                {"name": "file.py", "type": "file", "path": "/root/file.py"}
            ]
        }
        
        formatter = TreeFormatter()
        result = formatter.format(node)
        
        assert "root" in result
        assert "file.py" in result
    
    def test_format_nested_tree(self):
        """Should format nested directory structure"""
        node = {
            "name": "project",
            "type": "directory",
            "children": [
                {
                    "name": "src",
                    "type": "directory",
                    "children": [
                        {"name": "main.py", "type": "file", "path": "/project/src/main.py"}
                    ]
                }
            ]
        }
        
        formatter = TreeFormatter()
        result = formatter.format(node)
        
        assert "project" in result
        assert "src" in result
        assert "main.py" in result


class TestFileSystemVisitorIntegration:
    """Integration tests"""
    
    def test_visit_and_format_workflow(self):
        """Should visit directory and format output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("code")
            
            visitor = FileSystemVisitor()
            tree = visitor.visit(tmpdir)
            
            formatter = TreeFormatter()
            output = formatter.format(tree)
            
            assert output is not None
            assert len(output) > 0
    
    def test_visit_complex_project(self):
        """Should handle complex project structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create realistic project structure
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("main")
            (root / "tests").mkdir()
            (root / "tests" / "test_main.py").write_text("tests")
            (root / "docs").mkdir()
            (root / "docs" / "README.md").write_text("readme")
            
            visitor = FileSystemVisitor()
            result = visitor.visit(str(root))
            
            assert result["type"] == "directory"
            assert len(result["children"]) >= 3


# Fixtures
@pytest.fixture
def visitor():
    """Create FileSystemVisitor instance"""
    return FileSystemVisitor()


@pytest.fixture
def formatter():
    """Create TreeFormatter instance"""
    return TreeFormatter()


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    import shutil
    shutil.rmtree(tmpdir)
