from typing import Protocol
from langchain_core.language_models.chat_models import BaseChatModel
import ast
import json

class CodeConverter(Protocol):
    def convert(self, code: str) -> str:
        pass

class ArchitectureVisitor(ast.NodeVisitor):
    def __init__(self):
        self.structure = []
        self.current_class = None

    def visit_ClassDef(self, node):
        self.current_class = {
            "name": node.name,
            "bases": [self._get_id(b) for b in node.bases],
            "methods": [],
            "attributes": []
        }
        self.generic_visit(node)
        self.structure.append(self.current_class)
        self.current_class = None

    def visit_FunctionDef(self, node):
        if self.current_class:
            args = [arg.arg for arg in node.args.args]
            self.current_class["methods"].append({
                "name": node.name,
                "args": args
            })
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if self.current_class and self._is_self_attribute(node.target):
            attr_name = node.target.attr
            attr_type = self._get_id(node.annotation)
            self.current_class["attributes"].append({
                "name": attr_name,
                "type": attr_type
            })

    def visit_Assign(self, node):
        if not self.current_class:
            return
        if isinstance(node.value, ast.Call):
            class_name = self._get_id(node.value.func)
            if class_name in ["str", "int", "list", "dict", "len", "super"]:
                return
            for target in node.targets:
                if self._is_self_attribute(target):
                    self.current_class["attributes"].append({
                        "name": target.attr,
                        "type": class_name
                    })
        self.generic_visit(node)

    def _is_self_attribute(self, node) -> bool:
        return (isinstance(node, ast.Attribute) and 
                isinstance(node.value, ast.Name) and 
                node.value.id == 'self')

    def _get_id(self, node) -> str:
        if node is None:
            return "None"
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._get_id(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        elif isinstance(node, ast.Subscript):
            container_name = self._get_id(node.value)
            inner_name = self._get_id(node.slice)
            return f"{container_name}[{inner_name}]"
        elif isinstance(node, ast.Tuple):
            elements = [self._get_id(e) for e in node.elts]
            return ", ".join(elements)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return "Unknown"

class PythonToPlantUMLConverter:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _create_prompt(self, structure_data: str) -> str:
        return f"""
        Act as a Senior Software Architect.
        I have analyzed a Python file and extracted the class structure into JSON.
        
        Task: Convert this JSON metadata into a PlantUML Class Diagram.
        
        Rules:
        1. Use the class names and method names provided in the JSON.
        2. Draw inheritance arrows (`<|--`) based on the 'bases' field.
        3. Do not invent methods that are not in the JSON.
        4. Output ONLY raw PlantUML syntax (@startuml ... @enduml).

        Structure Data (JSON):
        ```json
        {structure_data}
        ```
        """

    def convert(self, structure_json) -> str:
        # Handle both list and string inputs
        if isinstance(structure_json, list):
            structure_json = json.dumps(structure_json, indent=2)
        
        if not structure_json or not str(structure_json).strip():
            raise ValueError("Code cannot be empty")
        prompt = self._create_prompt(structure_json)
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            raise RuntimeError(f"LLM Conversion failed: {e}")
    
    def _clean_output(self, text: str) -> str:
        """Clean markdown formatting from output"""
        text = text.strip()
        if text.startswith("```python"):
            text = text.replace("```python", "", 1)
        if text.startswith("```"):
            text = text.replace("```", "", 1)
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

class DesignPatternProvider:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    def _create_analysis_prompt(self, uml_diagram: str) -> str:
        return f"""Analyze this PlantUML class diagram and identify design patterns used.

For each pattern found, provide:
1. Pattern name (e.g., Observer, Strategy, Factory)
2. Classes/interfaces involved
3. Brief explanation of how the pattern is implemented
4. Benefits in this context

PlantUML diagram:
```plantuml
{uml_diagram}
```

Output format:
---
PATTERN NAME: [name]
COMPONENTS: [class names]
EXPLANATION: [how it's implemented in a few sentences(very short explanation)]
---"""
    
    def _create_recommendations_prompt(self, uml_diagram: str) -> str:
        return f"""Analyze this PlantUML class diagram and provide recommendations for improving its design patterns.

Focus on:
1. SOLID principles violations
2. Missing design patterns that could help
3. Over-engineering (unnecessary complexity)
4. Refactoring suggestions
5. Best practices not followed

PlantUML diagram:
```plantuml
{uml_diagram}
```

Provide actionable recommendations with clear explanations of why each change would improve the design."""
    
    def _create_improved_uml_prompt(self, uml_diagram: str) -> str:
        return f"""Based on best practices and design patterns, improve this PlantUML class diagram.

Original diagram:
```plantuml
{uml_diagram}
```

Create an improved version that:
1. Applies recommended design patterns (Factory, Strategy, Observer, etc.)
2. Follows SOLID principles
3. Has better separation of concerns
4. Includes proper abstractions and interfaces
5. Is more maintainable and scalable

Output ONLY the improved PlantUML code with no explanations or markdown formatting."""
    
    def _create_code_from_uml_prompt(self, uml_diagram: str) -> str:
        return f"""Convert this improved PlantUML class diagram to Python code that follows best practices.

PlantUML diagram:
```plantuml
{uml_diagram}
```

Requirements:
1. Generate complete, runnable Python code
2. Include proper type hints
3. Add docstrings for all classes and methods
4. Implement all relationships shown in the diagram
5. Use appropriate design patterns
6. Follow PEP 8 standards
7. Include example usage

Output ONLY the complete Python code (with explanatory comments.) with no explanations or markdown formatting."""
    
    def convert(self, uml_diagram: str) -> str:
        if not uml_diagram or not uml_diagram.strip():
            raise ValueError("PlantUML diagram cannot be empty")
        try:
            prompt = self._create_analysis_prompt(uml_diagram)
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to analyze design patterns: {str(e)}") from e
    
    def get_pattern_recommendations(self, uml_diagram: str) -> str:
        if not uml_diagram or not uml_diagram.strip():
            raise ValueError("PlantUML diagram cannot be empty")
        try:
            prompt = self._create_recommendations_prompt(uml_diagram)
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to get pattern recommendations: {str(e)}") from e
    
    def convert_to_improved_uml(self, uml_diagram: str) -> str:
        if not uml_diagram or not uml_diagram.strip():
            raise ValueError("PlantUML diagram cannot be empty")
        try:
            prompt = self._create_improved_uml_prompt(uml_diagram)
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate improved UML diagram: {str(e)}") from e
    
    def convert_to_code(self, improved_uml_diagram: str) -> str:
        if not improved_uml_diagram or not improved_uml_diagram.strip():
            raise ValueError("PlantUML diagram cannot be empty")
        try:
            prompt = self._create_code_from_uml_prompt(improved_uml_diagram)
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to convert UML to code: {str(e)}") from e