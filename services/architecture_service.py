import ast
import json
import sys 
from pathlib import Path
import re
import os 
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.llm_factory import create_sambanova_llm

class ArchitectureVisitor(ast.NodeVisitor):
    def __init__(self):
        self.structure = []
        self.current_class = None  # State to track which class we are inside
        self.global_functions = [] # Store standalone functions here

    def visit_ClassDef(self, node):
        """
        Enters a class. Sets up the dictionary.
        """
        self.current_class = {
            "name": node.name,
            "type": "class",
            "bases": [self._get_id(b) for b in node.bases],
            "methods": [],
            "attributes": [] 
        }
        self.generic_visit(node)
        # for item in node.body:
        #     self.visit(item)
        
        self.structure.append(self.current_class)
        self.current_class = None

    def visit_FunctionDef(self, node):
        args = [arg.arg for arg in node.args.args if arg.arg != 'self']
        return_type = self._get_id(node.returns) if node.returns else "Unknown"
        method_info = {
            "name": node.name,
            "args": args,
            "returns": return_type
        }
        if self.current_class:
            self.current_class["methods"].append(method_info)
        else:
            self.global_functions.append(method_info)
          
        self.generic_visit(node)

    def visit_Module(self, node):
        """
        Entry point. Visits all nodes, then packages global functions.
        """
        self.generic_visit(node)
        if self.global_functions:
            self.structure.append({
                "name": "Module_Functions",
                "type": "module", # Special tag for the converter
                "bases": [],
                "attributes": [],
                "methods": self.global_functions
            })
    

    def visit_AnnAssign(self, node):
        """
        Captures: self.variable: Type = ...
        Key for dependency injection detection.
        """
        if self.current_class and self._is_self_attribute(node.target):
            attr_name = node.target.attr
            attr_type = self._get_id(node.annotation) 
            
            self.current_class["attributes"].append({
                "name": attr_name,
                "type": attr_type
            })

    def visit_Assign(self, node):
        # 1. Sanity check: Are we in a class?
        if not self.current_class: return

        # Check if targeting self.attribute
        target_attr = None
        for target in node.targets:
            if self._is_self_attribute(target):
                target_attr = target.attr
                break
        
        if not target_attr: return

        # Logic: Identify Type or mark as Unknown
        inferred_type = "Unknown" # Default to trigger AI

        if isinstance(node.value, ast.Call):
            # It's a constructor (e.g., RedisCache())
            inferred_type = self._get_id(node.value.func)
            if inferred_type in ["str", "int", "list", "dict", "len", "super"]:
                return # Ignore primitives
        elif isinstance(node.value, ast.Constant):
            return 
                
        self.current_class["attributes"].append({
            "name": target_attr,
            "type": inferred_type
        })
        


    def _is_self_attribute(self, node) -> bool:
        """
        Helper: Checks if AST node represents 'self.variable'
        """
        return (isinstance(node, ast.Attribute) and 
                isinstance(node.value, ast.Name) and 
                node.value.id == 'self')

    def _get_id(self, node) -> str:
        """
        Recursive helper to extract type strings from AST nodes.
        """
        if node is None:
            return "None"

        # 1. Simple Type (int, str)
        if isinstance(node, ast.Name):
            return node.id
            
        # 2. Attributes (e.g., typing.List)
        elif isinstance(node, ast.Attribute):
            # Recursively get the value to support 'module.submodule.Type'
            val = self._get_id(node.value)
            return f"{val}.{node.attr}" if val else node.attr
            
        # 3. Complex Type (List[int])
        elif isinstance(node, ast.Subscript):
            container_name = self._get_id(node.value)
            inner_name = self._get_id(node.slice)
            return f"{container_name}[{inner_name}]"
        
        # 4. Tuples (Crucial for Dict[str, int])
        elif isinstance(node, ast.Tuple):
            # We must iterate over all elements in the tuple
            elements = [self._get_id(e) for e in node.elts]
            return ", ".join(elements)

        # 5. Constant (Python 3.9+ often puts simple types here if distinct)
        elif isinstance(node, ast.Constant):
            return str(node.value)

        return "Unknown"



class FastTypeEnricher:
    """takes the JSON from the Visitor, finds the "Unknown" types, and asks SambaNova to fix them."""

    def __init__(self , llm):
        self.llm = llm
        
    
    def enrich(self, code_context: str, structure: list[dict]) -> list[dict]:
        """
        Scans structure for missing types and asks Llama 3 to infer them.
        """
        missing_vars = []
        
        # 1. Find the gaps (Unknowns or inferred 'Any')
        for cls in structure:
            for attr in cls["attributes"]:
                if attr["type"] in ["Unknown", "Any"]:
                    # We need context: ClassName.AttributeName
                    missing_vars.append(f"{cls['name']}.{attr['name']}")
        
        if not missing_vars:
            return structure 
        # 2. Call SambaNova (The Surgical Strike)
        print(f"⚡ Fast System: Inferring types for {len(missing_vars)} variables...")
        
        prompt = f"""
        Act as a Python Static Analysis Engine.
        The following variables have missing type hints. Infer them based on the code.
        
        Variables to infer: {missing_vars}
        
        Rules:
        1. Return ONLY a valid JSON object.
        2. Keys must be "ClassName.AttributeName".
        3. Values must be the PEP 484 type (e.g., "List[str]", "UserRepository").
        4. If the type is primitive (str, int) or ambiguous, use "Any".
        
        Code Context:
        ```python
        {code_context[:4000]} 
        ```
        """
        
        try:
            messages = [
                SystemMessage(content="You are a JSON-only code analysis tool. Output valid JSON only."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # 3. Patch the Structure
            content = response.content
            # Clean up potential markdown blocks (```json ... ```)
            if "```" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            updates = json.loads(content)
            self._apply_patches(structure, updates)
        except Exception as e:
            error_msg = str(e)
            # Check if it's a model not found error
            if "404" in error_msg or "Model not found" in error_msg:
                print(f"⚠️ Model not available on SambaNova. Skipping type enrichment.")
                print(f"   (Tip: Check available models or use Nebius provider)")
            else:
                print(f"⚠️ Enrichment failed: {e}")
            
        return structure

    def _apply_patches(self, structure: list[dict], updates: dict):
        """
        Applies the inferred types back into the Visitor's structure.
        Complexity: O(N*M) but N (classes) and M (attributes) are small.
        """
        for key, inferred_type in updates.items():
            try:
                if "." not in key: continue
                
                class_name, attr_name = key.split(".", 1)
                
                # Find the class
                target_class = next((c for c in structure if c["name"] == class_name), None)
                if target_class:
                    # Find the attribute inside the class
                    for attr in target_class["attributes"]:
                        if attr["name"] == attr_name:
                            print(f"   ✅ Patched {key} -> {inferred_type}")
                            attr["type"] = inferred_type
                            break
            except Exception:
                continue



class DeterministicPlantUMLConverter:
    def convert(self, structure_json: list[dict]) -> str:
        if not structure_json: return ""
        known_classes = {cls["name"] for cls in structure_json}
        lines = ["@startuml", "skinparam linetype ortho"]

        # Draw Classes
        for cls in structure_json:
            lines.append(f"class {cls['name']} {{")
            for attr in cls["attributes"]:
                lines.append(f"  + {attr['name']} : {attr['type']}")
            lines.append("}")

        # Draw Arrows (The Test Subject)
        for cls in structure_json:
            # Inheritance
            for base in cls["bases"]:
                # Clean generics like Repository[Product] -> Repository
                base_clean = base.split("[")[0] 
                if base_clean in known_classes:
                    lines.append(f"{base_clean} <|-- {cls['name']}")

            # Dependencies
            for attr in cls["attributes"]:
                for target in known_classes:
                    if target == cls["name"]: continue
                    
                    # [CRITICAL REGEX]
                    # \b ensures we match "Product" but NOT "ProductionConfig"
                    pattern = fr"\b{re.escape(target)}\b"
                    
                    if re.search(pattern, attr["type"]):
                        lines.append(f"{cls['name']} o-- {target} : {attr['name']}")
                        break 

        lines.append("@enduml")
        return "\n".join(lines)




test_code = """
from typing import List, Dict, Optional, Union, Any, TypeVar, Generic
from abc import ABC, abstractmethod
import datetime

# [COMPLEXITY 1] Generics & Abstractions
T = TypeVar("T")

class Repository(Generic[T], ABC):
    @abstractmethod
    def save(self, entity: T) -> None:
        pass

class LoggableMixin:
    def log(self, msg: str):
        print(f"[LOG] {msg}")

# [COMPLEXITY 2] Domain Models
class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price

class User:
    def __init__(self, uid: str):
        self.uid = uid

# [COMPLEXITY 3] Naming Collision Trap for Regex
# Your regex must NOT draw an arrow from 'Product' to 'ProductionConfig'
class ProductionConfig:
    def __init__(self):
        self.env = "PROD"
        self.retries = 3

# [COMPLEXITY 4] Infrastructure Layer
class PostgresConnection:
    def connect(self):
        pass

class RedisCache:
    def set(self, key, val):
        pass

# [COMPLEXITY 5] Implementation with Mixins
class SqlProductRepository(Repository[Product], LoggableMixin):
    def __init__(self, db_conn):
        # [TRAP] Untyped Dependency!
        # The AI Enricher should infer that 'db_conn' is 'PostgresConnection' 
        # based on usage or naming convention.
        self.db = db_conn 
        
    def save(self, entity: Product) -> None:
        self.log(f"Saving {entity.name}")

# [COMPLEXITY 6] The Service Layer (The Spiderweb)
class ECommerceService:
    def __init__(self, repo: SqlProductRepository, config: ProductionConfig):
        # Typed Dependency (Easy for Visitor)
        self.repository: SqlProductRepository = repo
        self.config: ProductionConfig = config
        
        # [TRAP] Constructor Inference
        # Visitor should see this is a 'RedisCache'
        self.cache = RedisCache()
        
        # [TRAP] Complex Nested Type
        # A Dictionary mapping User IDs to a List of Products
        self.cart_state: Dict[str, List[Product]] = {}
        
        # [TRAP] Forward Reference (String literal)
        # Common in Django/FastAPI. Visitor needs to handle string 'User'
        self.current_admin: Optional['User'] = None
        
        # [TRAP] Union Type
        self.last_error: Union[ValueError, ConnectionError, None] = None

    def checkout(self, user_id: str) -> bool:
        return True
"""


if __name__ == "__main__":
    # try:
    #     tree = ast.parse(test_code)
    #     visitor = ArchitectureVisitor()
    #     visitor.visit(tree)
    #     print("=== Initial Structure ===")
    #     print(json.dumps(visitor.structure, indent=2))
        
    #     print(f"\n{50 * '='}== Enriched Structure {50 * '='}==")
        
    #     llm = create_sambanova_llm( temperature=0.0)
    #     enricher = FastTypeEnricher(llm)
    #     enriched = enricher.enrich(test_code, visitor.structure)
    #     print(json.dumps(enriched, indent=2))

    #     print(f"\n{50 * '='}== PlantUML Output {50 * '='}==")
    #     converter = DeterministicPlantUMLConverter()
    #     plantuml_output = converter.convert(enriched)
    #     print(plantuml_output)

        
    #     print("\n✅ Analysis completed successfully")
        
    # except Exception as e:
    #     print(f"❌ Error during analysis: {e}")
    #     import traceback
    #     traceback.print_exc()
    code = """
import os 
class MyClass:
    def method(self): pass

def my_global_function(x: int):
    pass

def another_function():
    return True
            """

    print("--- DEBUGGING VISITOR ---")
    tree = ast.parse(code)
    visitor = ArchitectureVisitor()
    visitor.visit(tree)

    print(f"Global Functions Found: {len(visitor.global_functions)}")
    for f in visitor.global_functions:
        print(f" - {f['name']}")

    print("\nStructure Output:")
    print(json.dumps(visitor.structure, indent=2))




