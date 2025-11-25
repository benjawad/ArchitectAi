import os
import ast
import logging
from pathlib import Path
from typing import List, Dict, Any

from services.architecture_service import ArchitectureVisitor

class ProjectAnalyzer:
    """
    Orchestrates the analysis of an entire directory.
    Aggregates results from multiple files into one structure.
    """
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.aggregated_structure: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def analyze(self) -> List[Dict[str, Any]]:
        """
        Walks through the directory, parses every .py file, and merges findings.
        """
        logging.info(f"📂 Starting project analysis at: {self.root_path}")
        
        # 1. Walk through all files recursively
        for file_path in self.root_path.rglob("*.py"):
            
            # Skip virtual environments, git, and caches
            # This is CRITICAL to avoid freezing on big projects
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules"] for p in parts):
                continue
                
            self._analyze_single_file(file_path)

        logging.info(f"✅ Project analysis complete. Found {len(self.aggregated_structure)} components.")
        return self.aggregated_structure

    def _analyze_single_file(self, file_path: Path):
        """
        Parses a single file using ArchitectureVisitor and appends classes to the main list.
        """
        try:
            code = file_path.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(code)
            
            # Re-use our existing visitor!
            visitor = ArchitectureVisitor()
            visitor.visit(tree)
            
            # Tag each component with its source file (Visual Context)
            # This is useful so you know WHERE a class is defined
            for item in visitor.structure:
                item['source_file'] = file_path.name
                # Optional: Prefix module name to avoid name collisions? 
                # For now, let's keep it simple.
                
            # Merge into main list
            self.aggregated_structure.extend(visitor.structure)
            
        except Exception as e:
            error_msg = f"Failed to parse {file_path.name}: {e}"
            logging.warning(error_msg)
            self.errors.append(error_msg)