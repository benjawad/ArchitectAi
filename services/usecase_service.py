import ast
import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Actor:
    """Represents an actor in use case diagram"""
    name: str
    type: str = "human"  # human, system, external
    description: str = ""


@dataclass
class UseCase:
    """Represents a use case"""
    name: str
    description: str = ""
    actor_names: List[str] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    extends: List[str] = field(default_factory=list)
    source_class: str = ""
    source_method: str = ""
    module: str = ""  # NEW: Track which module this belongs to


@dataclass  
class UseCaseModel:
    """Complete use case model for a system"""
    system_name: str
    actors: List[Actor] = field(default_factory=list)
    use_cases: List[UseCase] = field(default_factory=list)


@dataclass
class ModularUseCaseModel:
    """
    🆕 NEW: Organizes use cases by module
    Allows generating separate diagrams per module
    """
    project_name: str
    modules: Dict[str, UseCaseModel] = field(default_factory=dict)
    shared_actors: List[Actor] = field(default_factory=list)


# ============================================================
# AST VISITOR - Extracts raw data from code
# ============================================================

class UseCaseVisitor(ast.NodeVisitor):
    """
    AST Visitor that extracts potential use cases from Python code.
    🔧 TRACKS MODULE/FILE INFORMATION
    """
    
    SERVICE_PATTERNS = [
        r'.*Service$', r'.*Controller$', r'.*Handler$', 
        r'.*API$', r'.*View$', r'.*Endpoint$', r'.*Router$',
        r'.*Manager$', r'.*UseCase$', r'.*Command$',
        r'.*Processor$', r'.*Worker$', r'.*Agent$',
        r'.*Helper$', r'.*Util$', r'.*Provider$',
        r'.*Visitor$', r'.*Analyzer$', r'.*Generator$'
    ]
    
    ENDPOINT_DECORATORS = [
        'get', 'post', 'put', 'delete', 'patch',
        'route', 'api_view',
        'command', 'cli',
        'task', 'job',
        'websocket', 'subscribe',
        'tool', 'mcp'
    ]
    
    def __init__(self, include_all_classes: bool = True, module_name: str = "main"):
        """
        Args:
            include_all_classes: If True, includes ALL classes
            module_name: Name of the module/file being analyzed
        """
        self.potential_use_cases: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None
        self.current_class_docstring: str = ""
        self.imports: List[str] = []
        self.decorators_found: List[str] = []
        self.include_all_classes = include_all_classes
        self.module_name = module_name  # NEW: Track module
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        class_name = node.name
        
        is_service_class = any(
            re.match(pattern, class_name) 
            for pattern in self.SERVICE_PATTERNS
        )
        
        bases = [self._get_name(b) for b in node.bases]
        has_service_base = any(
            any(re.match(p, base) for p in self.SERVICE_PATTERNS)
            for base in bases if base
        )
        
        should_process = is_service_class or has_service_base or self.include_all_classes
        
        if should_process:
            self.current_class = class_name
            self.current_class_docstring = ast.get_docstring(node) or ""
            self.generic_visit(node)
            self.current_class = None
            self.current_class_docstring = ""
        else:
            old_class = self.current_class
            self.current_class = class_name
            self.generic_visit(node)
            self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        if node.name.startswith('_') and not node.name.startswith('__'):
            self.generic_visit(node)
            return
            
        decorators = self._extract_decorators(node)
        http_method = None
        route_path = None
        is_endpoint = False
        
        for dec in decorators:
            dec_lower = dec['name'].lower()
            if dec_lower in self.ENDPOINT_DECORATORS:
                is_endpoint = True
                http_method = dec_lower
                if dec['args']:
                    route_path = dec['args'][0]
            if '.' in dec['name']:
                method_part = dec['name'].split('.')[-1].lower()
                if method_part in self.ENDPOINT_DECORATORS:
                    is_endpoint = True
                    http_method = method_part
                    if dec['args']:
                        route_path = dec['args'][0]
        
        use_case_data = {
            'name': self._method_to_use_case_name(node.name),
            'method_name': node.name,
            'class_name': self.current_class or "Module",
            'module': self.module_name,  # NEW: Track module
            'docstring': ast.get_docstring(node) or "",
            'parameters': [arg.arg for arg in node.args.args if arg.arg != 'self'],
            'decorators': decorators,
            'is_endpoint': is_endpoint,
            'http_method': http_method,
            'route_path': route_path,
            'return_type': self._get_name(node.returns) if node.returns else None,
        }
        
        if is_endpoint or self.current_class:
            if not (node.name.startswith('__') and node.name != '__init__'):
                self.potential_use_cases.append(use_case_data)
            
        self.generic_visit(node)
    
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def _extract_decorators(self, node) -> List[Dict[str, Any]]:
        decorators = []
        for dec in node.decorator_list:
            dec_info = {'name': '', 'args': []}
            
            if isinstance(dec, ast.Name):
                dec_info['name'] = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_info['name'] = self._get_full_attr(dec)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    dec_info['name'] = dec.func.id
                elif isinstance(dec.func, ast.Attribute):
                    dec_info['name'] = self._get_full_attr(dec.func)
                for arg in dec.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        dec_info['args'].append(arg.value)
                        
            if dec_info['name']:
                decorators.append(dec_info)
                self.decorators_found.append(dec_info['name'])
                
        return decorators
    
    def _get_full_attr(self, node) -> str:
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def _get_name(self, node) -> Optional[str]:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._get_full_attr(node)
        if isinstance(node, ast.Constant):
            return str(node.value)
        return None
    
    def _method_to_use_case_name(self, method_name: str) -> str:
        for prefix in ['get_', 'create_', 'update_', 'delete_', 'do_', 'handle_', 'process_', 'execute_']:
            if method_name.startswith(prefix):
                method_name = method_name[len(prefix):]
                break
        
        words = method_name.split('_')
        return ' '.join(word.capitalize() for word in words)
    
    def get_framework_context(self) -> str:
        imports_lower = [i.lower() for i in self.imports]
        
        if any('fastapi' in i for i in imports_lower):
            return "FastAPI"
        if any('flask' in i for i in imports_lower):
            return "Flask"
        if any('django' in i for i in imports_lower):
            return "Django"
        if any('mcp' in i for i in imports_lower):
            return "MCP Server"
        if any('gradio' in i for i in imports_lower):
            return "Gradio App"
        return "Generic Python"


# ============================================================
# 🆕 MODULAR ANALYZER - Groups use cases by module
# ============================================================

class ModularUseCaseAnalyzer:
    """
    Analyzes multiple files and organizes use cases by module.
    Generates separate diagrams for each module.
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.enricher = UseCaseEnricher(llm) if llm else None
    
    def analyze_project(
        self, 
        file_contents: Dict[str, str],  # filename -> code
        enrich: bool = True
    ) -> ModularUseCaseModel:
        """
        Analyze entire project and organize by modules.
        
        Args:
            file_contents: Dict mapping filename to code content
            enrich: Whether to use LLM enrichment
            
        Returns:
            ModularUseCaseModel with separate diagrams per module
        """
        # Group use cases by module
        modules_use_cases = defaultdict(list)
        all_use_cases = []
        
        for filename, code in file_contents.items():
            try:
                # Extract module name from filename
                module_name = self._extract_module_name(filename)
                
                # Analyze this file
                tree = ast.parse(code)
                visitor = UseCaseVisitor(
                    include_all_classes=True,
                    module_name=module_name
                )
                visitor.visit(tree)
                
                # Group by module
                for uc in visitor.potential_use_cases:
                    modules_use_cases[module_name].append(uc)
                    all_use_cases.append(uc)
                    
            except Exception as e:
                logging.warning(f"⚠️ Failed to analyze {filename}: {e}")
                continue
        
        # Identify shared actors across all modules
        shared_actors = self._identify_shared_actors(all_use_cases)
        
        # Create UseCaseModel for each module
        modular_model = ModularUseCaseModel(
            project_name="Project",
            shared_actors=shared_actors
        )
        
        for module_name, use_cases in modules_use_cases.items():
            if not use_cases:
                continue
            
            # Create model for this module
            if enrich and self.enricher:
                model = self.enricher.enrich(
                    use_cases,
                    "",  # We don't need full code context
                    "Generic Python"
                )
                model.system_name = f"{module_name} Module"
            else:
                model = self._create_basic_model(use_cases, module_name)
            
            modular_model.modules[module_name] = model
        
        return modular_model
    
    def _extract_module_name(self, filename: str) -> str:
        """Extract clean module name from filename"""
        # Remove .py extension
        name = filename.replace('.py', '')
        # Remove path separators
        name = name.replace('/', '_').replace('\\', '_')
        # Clean up
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Capitalize
        return ' '.join(word.capitalize() for word in name.split('_'))
    
    def _identify_shared_actors(self, all_use_cases: List[Dict]) -> List[Actor]:
        """Identify actors that appear across multiple modules"""
        actors = [
            Actor(name="User", type="human", description="System user"),
            Actor(name="Developer", type="human", description="Developer using the system")
        ]
        
        # Check for common patterns
        if any("llm" in uc['method_name'].lower() for uc in all_use_cases):
            actors.append(Actor(name="LLM", type="external", description="AI Language Model"))
        
        if any(uc.get('is_endpoint') for uc in all_use_cases):
            actors.append(Actor(name="API Client", type="external", description="External API consumer"))
        
        return actors
    
    def _create_basic_model(self, use_cases: List[Dict], module_name: str) -> UseCaseModel:
        """Create basic model without LLM enrichment"""
        actors = [Actor(name="User", type="human")]
        
        use_case_objects = [
            UseCase(
                name=uc['name'],
                description=uc['docstring'][:100] if uc['docstring'] else "",
                actor_names=["User"],
                source_class=uc['class_name'],
                source_method=uc['method_name'],
                module=uc['module']
            )
            for uc in use_cases
        ]
        
        return UseCaseModel(
            system_name=f"{module_name} Module",
            actors=actors,
            use_cases=use_case_objects
        )


# ============================================================
# LLM ENRICHER (Same as before)
# ============================================================

class UseCaseEnricher:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    def enrich(self, raw_use_cases: List[Dict], code_context: str, framework: str) -> UseCaseModel:
        if not raw_use_cases:
            return UseCaseModel(system_name="System")
        
        logging.info(f"🎭 Enriching {len(raw_use_cases)} use cases...")
        
        summary = self._prepare_summary(raw_use_cases)
        
        prompt = f"""
Act as a Software Architect. Analyze these code elements and create a Use Case model.

## Elements:
{json.dumps(summary, indent=2)}

## Output (JSON ONLY):
{{
    "system_name": "Module Name",
    "actors": [
        {{"name": "User", "type": "human", "description": "..."}}
    ],
    "use_cases": [
        {{
            "name": "Use Case Name",
            "description": "...",
            "actor_names": ["User"],
            "includes": [],
            "extends": []
        }}
    ]
}}

Return ONLY valid JSON.
"""
        
        try:
            messages = [
                SystemMessage(content="You are a JSON-only assistant. Output valid JSON only."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = self._clean_json(response.content)
            data = json.loads(content)
            
            return self._parse_model(data)
            
        except Exception as e:
            logging.warning(f"⚠️ Enrichment failed: {e}")
            return self._create_basic_model(raw_use_cases)
    
    def _prepare_summary(self, raw_use_cases: List[Dict]) -> List[Dict]:
        return [
            {
                'method': uc['method_name'],
                'class': uc['class_name'],
                'description': uc['docstring'][:200] if uc['docstring'] else "",
                'is_endpoint': uc['is_endpoint'],
            }
            for uc in raw_use_cases
        ]
    
    def _clean_json(self, content: str) -> str:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return content.strip()
    
    def _parse_model(self, data: Dict) -> UseCaseModel:
        actors = [
            Actor(
                name=a.get('name', 'Unknown'),
                type=a.get('type', 'human'),
                description=a.get('description', '')
            )
            for a in data.get('actors', [])
        ]
        
        use_cases = [
            UseCase(
                name=uc.get('name', ''),
                description=uc.get('description', ''),
                actor_names=uc.get('actor_names', []),
                includes=uc.get('includes', []),
                extends=uc.get('extends', [])
            )
            for uc in data.get('use_cases', [])
        ]
        
        return UseCaseModel(
            system_name=data.get('system_name', 'System'),
            actors=actors,
            use_cases=use_cases
        )
    
    def _create_basic_model(self, raw_use_cases: List[Dict]) -> UseCaseModel:
        actors = [Actor(name="User", type="human")]
        
        use_cases = [
            UseCase(
                name=uc['name'],
                actor_names=["User"]
            )
            for uc in raw_use_cases
        ]
        
        return UseCaseModel(
            system_name="System",
            actors=actors,
            use_cases=use_cases
        )


# ============================================================
# PLANTUML CONVERTER
# ============================================================

class UseCasePlantUMLConverter:
    def convert(self, model: UseCaseModel) -> str:
        """Generate PlantUML use case diagram"""
        lines = [
            "@startuml",
            f"title {model.system_name}",
            "",
            "skinparam actorStyle awesome",
            "skinparam usecaseBackgroundColor #f0f8ff",
            "skinparam usecaseBorderColor #4169e1",
            "left to right direction",
            ""
        ]
        
        # Actors
        lines.append("' -- Actors --")
        for actor in model.actors:
            if actor.type == "system":
                lines.append(f'actor "{actor.name}" as {self._safe_id(actor.name)} <<system>>')
            elif actor.type == "external":
                lines.append(f'actor "{actor.name}" as {self._safe_id(actor.name)} <<external>>')
            else:
                lines.append(f'actor "{actor.name}" as {self._safe_id(actor.name)}')
        
        lines.append("")
        
        # System boundary
        lines.append(f'rectangle "{model.system_name}" {{')
        
        for uc in model.use_cases:
            uc_id = self._safe_id(uc.name)
            lines.append(f'    usecase "{uc.name}" as {uc_id}')
        
        lines.append("}")
        lines.append("")
        
        # Relationships
        lines.append("' -- Relationships --")
        for uc in model.use_cases:
            uc_id = self._safe_id(uc.name)
            for actor_name in uc.actor_names:
                actor_id = self._safe_id(actor_name)
                if any(a.name == actor_name for a in model.actors):
                    lines.append(f"{actor_id} --> {uc_id}")
        
        # Include/Extend
        if any(uc.includes for uc in model.use_cases):
            lines.append("")
            for uc in model.use_cases:
                uc_id = self._safe_id(uc.name)
                for inc in uc.includes:
                    inc_id = self._safe_id(inc)
                    if any(u.name == inc for u in model.use_cases):
                        lines.append(f"{uc_id} ..> {inc_id} : <<include>>")
        
        lines.append("")
        lines.append("@enduml")
        
        return "\n".join(lines)
    
    def _safe_id(self, name: str) -> str:
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        safe = re.sub(r'_+', '_', safe)
        return f"UC_{safe}" if safe[0].isdigit() else safe


# ============================================================
# MAIN SERVICE
# ============================================================

class UseCaseDiagramService:
    """
    Enhanced service that can generate:
    1. Single diagram (original behavior)
    2. Multiple diagrams per module (NEW)
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.visitor = None
        self.enricher = UseCaseEnricher(llm) if llm else None
        self.converter = UseCasePlantUMLConverter()
        self.modular_analyzer = ModularUseCaseAnalyzer(llm)
    
    def generate(self, code: str, enrich: bool = True, include_all_classes: bool = True) -> str:
        """Generate single diagram (original behavior)"""
        try:
            tree = ast.parse(code)
            self.visitor = UseCaseVisitor(include_all_classes=include_all_classes)
            self.visitor.visit(tree)
            
            raw_use_cases = self.visitor.potential_use_cases
            framework = self.visitor.get_framework_context()
            
            logging.info(f"✓ Found {len(raw_use_cases)} use cases ({framework})")
            
            if not raw_use_cases:
                return self._empty_diagram(framework)
            
            if enrich and self.enricher:
                model = self.enricher.enrich(raw_use_cases, code[:4000], framework)
            else:
                model = self._create_basic_model(raw_use_cases, framework)
            
            return self.converter.convert(model)
            
        except Exception as e:
            logging.error(f"❌ Use case generation failed: {e}")
            import traceback
            traceback.print_exc()
            return f"' ❌ Error: {e}"
    
    def generate_modular(
        self,
        file_contents: Dict[str, str],
        enrich: bool = True
    ) -> Dict[str, str]:
        """
        🆕 NEW: Generate separate diagram for each module
        
        Returns:
            Dict mapping module name to PlantUML diagram
        """
        try:
            modular_model = self.modular_analyzer.analyze_project(
                file_contents,
                enrich=enrich
            )
            
            diagrams = {}
            
            for module_name, model in modular_model.modules.items():
                puml = self.converter.convert(model)
                diagrams[module_name] = puml
            
            logging.info(f"✓ Generated {len(diagrams)} modular diagrams")
            return diagrams
            
        except Exception as e:
            logging.error(f"❌ Modular generation failed: {e}")
            return {"error": f"' ❌ Error: {e}"}
    
    def _create_basic_model(self, raw_use_cases: List[Dict], framework: str) -> UseCaseModel:
        actors = [Actor(name="User", type="human")]
        
        use_cases = [
            UseCase(
                name=uc['name'],
                actor_names=["User"],
                source_class=uc['class_name'],
                source_method=uc['method_name']
            )
            for uc in raw_use_cases
        ]
        
        system_name = raw_use_cases[0]['class_name'] if raw_use_cases else "System"
        
        return UseCaseModel(
            system_name=f"{system_name} System",
            actors=actors,
            use_cases=use_cases
        )
    
    def _empty_diagram(self, framework: str) -> str:
        return f"""@startuml
title No Use Cases Found

note as N1
  No public methods detected in {framework} code.
end note

@enduml"""