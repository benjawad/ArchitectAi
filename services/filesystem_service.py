import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Literal

@dataclass
class FileSystemNode:
    name: str
    type: Literal["file", "directory"]
    path: str
    size: Optional[int] = None
    children: Optional[List['FileSystemNode']] = None

class FileSystemVisitor:
    """
    A deterministic visitor for the file system.
    Acts exactly like ast.NodeVisitor but for folders.
    """
    
    # Hardcoded junk filter (The "Context Window Saver")
    IGNORED_DIRS = {
        "__pycache__", "node_modules", ".git", ".vscode", ".idea", 
        "dist", "build", "coverage", ".venv", "venv", "env"
    }
    
    IGNORED_FILES = {
        ".DS_Store", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"
    }

    def visit(self, root_path: str, max_depth: int = 4) -> dict:
        """
        Public entry point. Returns a Dictionary representing the tree.
        """
        path = Path(root_path).resolve()
        if not path.exists():
            raise ValueError(f"Path not found: {root_path}")
            
        # Start the recursion
        node = self._visit_node(path, current_depth=0, max_depth=max_depth)
        return asdict(node) if node else {}

    def _visit_node(self, path: Path, current_depth: int, max_depth: int) -> Optional[FileSystemNode]:
        """
        Recursive logic. 
        If directory: returns Node with children.
        If file: returns Node (leaf).
        """
        # 1. Check Filters (The "Guard Clauses")
        if path.name in self.IGNORED_DIRS or path.name in self.IGNORED_FILES:
            return None
        
        # 2. Handle File (Leaf Node)
        if path.is_file():
            return FileSystemNode(
                name=path.name,
                type="file",
                path=str(path),
                size=path.stat().st_size
            )

        # 3. Handle Directory (Container Node)
        if path.is_dir():
            # Stop recursion if too deep
            if current_depth >= max_depth:
                return FileSystemNode(
                    name=path.name,
                    type="directory",
                    path=str(path),
                    children=[] # Truncated
                )

            children_nodes = []
            try:
                # Deterministic Sort: Directories first, then files (A-Z)
                # This ensures the LLM always sees the same structure.
                entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
                
                for entry in entries:
                    child = self._visit_node(entry, current_depth + 1, max_depth)
                    if child:
                        children_nodes.append(child)
                        
            except PermissionError:
                pass # Skip locked folders

            return FileSystemNode(
                name=path.name,
                type="directory",
                path=str(path),
                children=children_nodes
            )
            
        return None

# ---------------------------------------------------------
# The Formatter (Equivalent to your PlantUML Converter)
# ---------------------------------------------------------
class TreeFormatter:
    def format(self, node_dict: dict) -> str:
        """Converts the JSON tree into a visual string."""
        lines = []
        self._render(node_dict, lines, "", is_last=True)
        return "\n".join(lines)

    def _render(self, node: dict, lines: list, prefix: str, is_last: bool):
        # Visual logic (└── vs ├──)
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node['name']}")
        
        prefix += "    " if is_last else "│   "
        
        children = node.get("children", []) or []
        count = len(children)
        
        for i, child in enumerate(children):
            self._render(child, lines, prefix, i == count - 1)

