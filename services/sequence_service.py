"""
Multi-Module Sequence Diagram Service
Generates SEPARATE sequence diagrams for each module/service.

NEW FEATURES:
- ModularSequenceModel: Groups sequence flows by module
- ModularSequenceAnalyzer: Organizes entry points by file/module
- generate_modular(): Returns Dict[module_name, puml_diagram]

This solves the "one massive sequence diagram" problem by creating
focused diagrams per module (3-8 sequences each).
"""

import ast
import json
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Participant:
    """A participant in the sequence diagram (class/module/actor)"""
    name: str
    alias: str = ""
    type: str = "participant"
    order: int = 0


@dataclass
class Message:
    """A message (method call) between participants"""
    from_participant: str
    to_participant: str
    method_name: str
    arguments: List[str] = field(default_factory=list)
    return_value: Optional[str] = None
    is_async: bool = False
    is_self_call: bool = False
    sequence_number: int = 0
    
    condition: Optional[str] = None
    loop_context: Optional[str] = None


@dataclass
class SequenceBlock:
    """Represents control flow blocks (alt, opt, loop, par)"""
    block_type: str
    condition: str = ""
    messages: List[Message] = field(default_factory=list)
    else_messages: List[Message] = field(default_factory=list)


@dataclass
class SequenceModel:
    """Complete sequence diagram model"""
    title: str
    participants: List[Participant] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    blocks: List[SequenceBlock] = field(default_factory=list)
    entry_point: str = ""
    module: str = ""  # 🆕 NEW: Which module this belongs to


# 🆕 NEW: MODULAR SEQUENCE MODEL
@dataclass
class ModularSequenceModel:
    """
    Organizes sequence diagrams by module.
    
    Structure:
    {
        "Services Architecture": SequenceModel(...),
        "Agent Module": SequenceModel(...),
        "Core Llm": SequenceModel(...)
    }
    """
    sequences_by_module: Dict[str, SequenceModel] = field(default_factory=dict)
    entry_points_by_module: Dict[str, List[str]] = field(default_factory=dict)


# ============================================================
# AST VISITOR - Traces method calls
# ============================================================

class CallGraphVisitor(ast.NodeVisitor):
    """
    AST Visitor that builds a call graph for sequence diagrams.
    
    🆕 NEW: Now tracks module_name for each method
    """
    
    def __init__(self, module_name: str = ""):
        # 🆕 NEW: Track which module/file this belongs to
        self.module_name = module_name
        
        # Class structure
        self.classes: Dict[str, Dict] = {}
        self.current_class: Optional[str] = None
        self.current_method: Optional[str] = None
        
        # Call graph
        self.call_sequences: Dict[str, List[Dict]] = defaultdict(list)
        
        # Dependencies
        self.dependencies: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Control flow
        self.control_flow_stack: List[Dict] = []
        
        # Imports
        self.imports: Dict[str, str] = {}
        
        # 🆕 NEW: Track which module each method belongs to
        self.method_modules: Dict[str, str] = {}
        
    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = f"{module}.{alias.name}"
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Enter a class definition"""
        self.current_class = node.name
        self.classes[node.name] = {
            'methods': [],
            'attributes': {},
            'bases': [self._get_name(b) for b in node.bases],
            'module': self.module_name  # 🆕 NEW
        }
        self.generic_visit(node)
        self.current_class = None
    
    def visit_FunctionDef(self, node):
        """Visit method/function definitions"""
        if self.current_class:
            self.classes[self.current_class]['methods'].append(node.name)
            
        self.current_method = f"{self.current_class}.{node.name}" if self.current_class else node.name
        self.call_sequences[self.current_method] = []
        
        # 🆕 NEW: Track module for this method
        self.method_modules[self.current_method] = self.module_name
        
        self.generic_visit(node)
        self.current_method = None
    
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def visit_AnnAssign(self, node):
        """Capture typed attributes: self.service: UserService"""
        if self.current_class and self._is_self_attr(node.target):
            attr_name = node.target.attr
            attr_type = self._get_name(node.annotation)
            self.classes[self.current_class]['attributes'][attr_name] = attr_type
            self.dependencies[self.current_class][attr_name] = attr_type
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Capture untyped attributes: self.cache = RedisCache()"""
        if self.current_class:
            for target in node.targets:
                if self._is_self_attr(target):
                    attr_name = target.attr
                    attr_type = self._infer_type(node.value)
                    if attr_type and attr_type not in ['str', 'int', 'list', 'dict', 'None']:
                        self.classes[self.current_class]['attributes'][attr_name] = attr_type
                        self.dependencies[self.current_class][attr_name] = attr_type
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """THE CORE: Trace method calls for sequence diagram"""
        if not self.current_method:
            self.generic_visit(node)
            return
        
        call_info = self._analyze_call(node)
        if call_info:
            if self.control_flow_stack:
                call_info['control_flow'] = self.control_flow_stack[-1].copy()
            
            self.call_sequences[self.current_method].append(call_info)
        
        self.generic_visit(node)
    
    def visit_Await(self, node):
        """Mark async calls"""
        if isinstance(node.value, ast.Call):
            call_info = self._analyze_call(node.value)
            if call_info:
                call_info['is_async'] = True
                if self.control_flow_stack:
                    call_info['control_flow'] = self.control_flow_stack[-1].copy()
                self.call_sequences[self.current_method].append(call_info)
        self.generic_visit(node)
    
    def visit_If(self, node):
        """Track if/else control flow"""
        condition = self._get_condition_str(node.test)
        
        self.control_flow_stack.append({'type': 'alt', 'condition': condition, 'branch': 'if'})
        for stmt in node.body:
            self.visit(stmt)
        self.control_flow_stack.pop()
        
        if node.orelse:
            self.control_flow_stack.append({'type': 'alt', 'condition': condition, 'branch': 'else'})
            for stmt in node.orelse:
                self.visit(stmt)
            self.control_flow_stack.pop()
    
    def visit_For(self, node):
        """Track for loops"""
        iter_str = self._get_name(node.iter) or "items"
        target_str = self._get_name(node.target) or "item"
        
        self.control_flow_stack.append({
            'type': 'loop', 
            'condition': f"for {target_str} in {iter_str}"
        })
        self.generic_visit(node)
        self.control_flow_stack.pop()
    
    def visit_While(self, node):
        """Track while loops"""
        condition = self._get_condition_str(node.test)
        
        self.control_flow_stack.append({'type': 'loop', 'condition': f"while {condition}"})
        self.generic_visit(node)
        self.control_flow_stack.pop()
    
    def visit_Try(self, node):
        """Track try/except blocks"""
        self.control_flow_stack.append({'type': 'opt', 'condition': 'try'})
        for stmt in node.body:
            self.visit(stmt)
        self.control_flow_stack.pop()
        
        for handler in node.handlers:
            exc_type = self._get_name(handler.type) if handler.type else "Exception"
            self.control_flow_stack.append({'type': 'alt', 'condition': f'except {exc_type}', 'branch': 'except'})
            for stmt in handler.body:
                self.visit(stmt)
            self.control_flow_stack.pop()
    
    def _analyze_call(self, node: ast.Call) -> Optional[Dict]:
        """Analyze a call node and extract sequence information"""
        func = node.func
        
        # Case 1: self.method()
        if isinstance(func, ast.Attribute) and self._is_self(func.value):
            return {
                'target': 'self',
                'target_type': self.current_class,
                'method': func.attr,
                'args': self._extract_args(node),
                'is_self_call': True,
                'is_async': False
            }
        
        # Case 2: self.dependency.method()
        if isinstance(func, ast.Attribute):
            value = func.value
            
            if isinstance(value, ast.Attribute) and self._is_self(value.value):
                dep_name = value.attr
                dep_type = self._resolve_dependency_type(dep_name)
                return {
                    'target': dep_name,
                    'target_type': dep_type,
                    'method': func.attr,
                    'args': self._extract_args(node),
                    'is_self_call': False,
                    'is_async': False
                }
            
            if isinstance(value, ast.Name):
                var_name = value.id
                return {
                    'target': var_name,
                    'target_type': self._infer_variable_type(var_name),
                    'method': func.attr,
                    'args': self._extract_args(node),
                    'is_self_call': False,
                    'is_async': False
                }
        
        # Case 3: ClassName.method()
        if isinstance(func, ast.Attribute):
            return {
                'target': self._get_name(func.value),
                'target_type': self._get_name(func.value),
                'method': func.attr,
                'args': self._extract_args(node),
                'is_self_call': False,
                'is_async': False
            }
        
        # Case 4: Direct function call
        if isinstance(func, ast.Name):
            return {
                'target': func.id,
                'target_type': func.id,
                'method': '__call__',
                'args': self._extract_args(node),
                'is_self_call': False,
                'is_async': False
            }
        
        return None
    
    def _extract_args(self, node: ast.Call) -> List[str]:
        """Extract argument names/values from call"""
        args = []
        for arg in node.args[:3]:
            args.append(self._get_name(arg) or "...")
        for kw in node.keywords[:2]:
            args.append(f"{kw.arg}=...")
        return args
    
    def _resolve_dependency_type(self, attr_name: str) -> str:
        """Resolve type of self.attr"""
        if self.current_class and self.current_class in self.dependencies:
            return self.dependencies[self.current_class].get(attr_name, attr_name.title())
        return attr_name.title()
    
    def _infer_variable_type(self, var_name: str) -> str:
        """Try to infer type from variable name conventions"""
        if var_name in self.imports:
            return self.imports[var_name].split('.')[-1]
        if var_name.endswith('_service'):
            return var_name.replace('_service', '').title() + 'Service'
        if var_name.endswith('_repo') or var_name.endswith('_repository'):
            return var_name.replace('_repo', '').replace('_repository', '').title() + 'Repository'
        return var_name.title()
    
    def _infer_type(self, node) -> Optional[str]:
        """Infer type from assignment value"""
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        return None
    
    def _is_self(self, node) -> bool:
        """Check if node is 'self'"""
        return isinstance(node, ast.Name) and node.id == 'self'
    
    def _is_self_attr(self, node) -> bool:
        """Check if node is 'self.something'"""
        return (isinstance(node, ast.Attribute) and 
                isinstance(node.value, ast.Name) and 
                node.value.id == 'self')
    
    def _get_name(self, node) -> Optional[str]:
        """Extract name from AST node"""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            val = self._get_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        if isinstance(node, ast.Constant):
            return repr(node.value)[:20]
        if isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        return None
    
    def _get_condition_str(self, node) -> str:
        """Convert condition AST to readable string"""
        if isinstance(node, ast.Compare):
            left = self._get_name(node.left) or "x"
            ops = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.Gt: ">", 
                   ast.Is: "is", ast.IsNot: "is not", ast.In: "in"}
            op = ops.get(type(node.ops[0]), "?")
            right = self._get_name(node.comparators[0]) or "y"
            return f"{left} {op} {right}"
        if isinstance(node, ast.BoolOp):
            op = "and" if isinstance(node.op, ast.And) else "or"
            return f" {op} ".join(self._get_condition_str(v) for v in node.values[:2])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"not {self._get_name(node.operand)}"
        if isinstance(node, ast.Call):
            return f"{self._get_name(node.func)}()"
        return self._get_name(node) or "condition"


# ============================================================
# SEQUENCE BUILDER
# ============================================================

class SequenceBuilder:
    """Builds a SequenceModel from the CallGraphVisitor's output"""
    
    STEREOTYPES = {
        'service': 'control',
        'controller': 'boundary', 
        'handler': 'boundary',
        'repository': 'entity',
        'repo': 'entity',
        'dao': 'entity',
        'cache': 'database',
        'db': 'database',
        'client': 'boundary',
        'api': 'boundary',
        'gateway': 'boundary',
    }
    
    def build(self, visitor: CallGraphVisitor, entry_method: str) -> SequenceModel:
        """Build sequence model starting from entry_method"""
        model = SequenceModel(title=entry_method.split('.')[-1])
        model.entry_point = entry_method
        model.module = visitor.method_modules.get(entry_method, visitor.module_name)  # 🆕 NEW
        
        visited_methods: Set[str] = set()
        participant_set: Set[str] = set()
        messages: List[Message] = []
        
        if '.' in entry_method:
            entry_class = entry_method.split('.')[0]
            participant_set.add(entry_class)
        
        self._traverse_calls(
            visitor, entry_method, 
            visited_methods, participant_set, messages,
            sequence_counter=[0]
        )
        
        model.participants = self._build_participants(participant_set, visitor)
        model.messages = messages
        
        return model
    
    def _traverse_calls(
        self, 
        visitor: CallGraphVisitor,
        method: str,
        visited: Set[str],
        participants: Set[str],
        messages: List[Message],
        sequence_counter: List[int],
        depth: int = 0
    ):
        """Recursively traverse call graph"""
        if method in visited or depth > 10:
            return
        visited.add(method)
        
        calls = visitor.call_sequences.get(method, [])
        current_class = method.split('.')[0] if '.' in method else "Module"
        
        for call in calls:
            sequence_counter[0] += 1
            
            target = call['target_type'] or call['target']
            
            if target.lower() in ['str', 'int', 'list', 'dict', 'print', 'len', 'range']:
                continue
            
            participants.add(target)
            
            msg = Message(
                from_participant=current_class,
                to_participant=target,
                method_name=call['method'],
                arguments=call['args'],
                is_async=call.get('is_async', False),
                is_self_call=call.get('is_self_call', False),
                sequence_number=sequence_counter[0]
            )
            
            if 'control_flow' in call:
                cf = call['control_flow']
                msg.condition = cf.get('condition')
                msg.loop_context = cf.get('condition') if cf['type'] == 'loop' else None
            
            messages.append(msg)
            
            called_method = f"{target}.{call['method']}"
            if called_method in visitor.call_sequences:
                self._traverse_calls(
                    visitor, called_method,
                    visited, participants, messages,
                    sequence_counter, depth + 1
                )
    
    def _build_participants(
        self, 
        participant_names: Set[str], 
        visitor: CallGraphVisitor
    ) -> List[Participant]:
        """Build participant list with types and ordering"""
        participants = []
        
        for i, name in enumerate(sorted(participant_names)):
            ptype = self._infer_stereotype(name)
            participants.append(Participant(
                name=name,
                alias=self._safe_alias(name),
                type=ptype,
                order=i
            ))
        
        return participants
    
    def _infer_stereotype(self, name: str) -> str:
        """Infer PlantUML stereotype from class name"""
        name_lower = name.lower()
        for pattern, stereotype in self.STEREOTYPES.items():
            if pattern in name_lower:
                return stereotype
        return 'participant'
    
    def _safe_alias(self, name: str) -> str:
        """Create safe alias for PlantUML"""
        return re.sub(r'[^a-zA-Z0-9]', '', name)


# ============================================================
# 🆕 NEW: MODULAR SEQUENCE ANALYZER
# ============================================================

class ModularSequenceAnalyzer:
    """
    Analyzes project and groups sequence diagrams by module.
    
    Key difference from single-file: Returns MULTIPLE diagrams.
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.visitors_by_module: Dict[str, CallGraphVisitor] = {}
        self.global_visitor = CallGraphVisitor()
    
    def analyze_project(self, file_contents: Dict[str, str]) -> ModularSequenceModel:
        """
        Analyze multiple files and create separate diagrams per module.
        
        Args:
            file_contents: {"services/order.py": code, "agent/workflow.py": code}
        
        Returns:
            ModularSequenceModel with separate diagrams per module
        """
        modular_model = ModularSequenceModel()
        
        # Step 1: Analyze each file individually
        for filepath, code in file_contents.items():
            module_name = self._extract_module_name(filepath)
            
            try:
                tree = ast.parse(code)
                visitor = CallGraphVisitor(module_name=module_name)
                visitor.visit(tree)
                
                self.visitors_by_module[module_name] = visitor
                
                # Merge into global visitor for cross-module calls
                self.global_visitor.classes.update(visitor.classes)
                self.global_visitor.dependencies.update(visitor.dependencies)
                
                for method, calls in visitor.call_sequences.items():
                    self.global_visitor.call_sequences[method].extend(calls)
                    self.global_visitor.method_modules[method] = module_name
                
                # Track entry points per module
                entry_points = [
                    method for method in visitor.call_sequences.keys()
                    if not method.startswith('_') and visitor.call_sequences[method]
                ]
                
                if entry_points:
                    modular_model.entry_points_by_module[module_name] = entry_points
                    
            except Exception as e:
                logging.warning(f"⚠️ Failed to analyze {filepath}: {e}")
        
        return modular_model
    
    def generate_diagrams(
        self, 
        modular_model: ModularSequenceModel,
        enrich: bool = True
    ) -> Dict[str, str]:
        """
        Generate separate PlantUML diagrams for each module.
        
        Returns:
            {"Services Order": "@startuml...", "Agent Workflow": "@startuml..."}
        """
        diagrams = {}
        builder = SequenceBuilder()
        enricher = SequenceEnricher(self.llm) if self.llm and enrich else None
        converter = SequencePlantUMLConverter()
        
        for module_name, entry_points in modular_model.entry_points_by_module.items():
            if not entry_points:
                continue
            
            # Pick the most complex entry point for this module
            best_entry = self._pick_best_entry_point(entry_points)
            
            try:
                # Build sequence model
                model = builder.build(self.global_visitor, best_entry)
                model.module = module_name
                
                if not model.messages:
                    continue
                
                # Enrich if LLM available
                if enricher:
                    try:
                        model = enricher.enrich(model, "")
                    except:
                        pass
                
                # Convert to PlantUML
                puml = converter.convert(model)
                diagrams[module_name] = puml
                
            except Exception as e:
                logging.warning(f"⚠️ Failed to generate diagram for {module_name}: {e}")
        
        if not diagrams:
            return {"error": "⚠️ No sequence diagrams generated. No method calls found."}
        
        return diagrams
    
    def _extract_module_name(self, filepath: str) -> str:
        """
        Convert filepath to human-readable module name.
        
        Examples:
            "services/architecture_service.py" -> "Services Architecture Service"
            "agent/agent_graph.py" -> "Agent Agent Graph"
            "core/llm_factory.py" -> "Core Llm Factory"
        """
        path_obj = Path(filepath)
        
        # Remove .py extension
        name = path_obj.stem
        
        # Get parent directory
        if path_obj.parent and path_obj.parent.name != '.':
            parent = path_obj.parent.name
            name = f"{parent}_{name}"
        
        # Clean and capitalize
        name = name.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def _pick_best_entry_point(self, entry_points: List[str]) -> str:
        """Pick the entry point with most calls (most interesting sequence)"""
        best = entry_points[0]
        best_count = len(self.global_visitor.call_sequences.get(best, []))
        
        for entry in entry_points[1:]:
            count = len(self.global_visitor.call_sequences.get(entry, []))
            if count > best_count:
                best = entry
                best_count = count
        
        return best


# ============================================================
# LLM ENRICHER
# ============================================================

class SequenceEnricher:
    """Uses LLM to improve sequence diagram readability"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    def enrich(self, model: SequenceModel, code_context: str) -> SequenceModel:
        """Enrich the sequence model with LLM inference"""
        if not model.messages:
            return model
        
        logging.info(f"🎬 Enriching sequence diagram with {len(model.messages)} messages...")
        
        summary = {
            'title': model.title,
            'participants': [p.name for p in model.participants],
            'messages': [
                {
                    'from': m.from_participant,
                    'to': m.to_participant,
                    'method': m.method_name,
                    'args': m.arguments[:3]
                }
                for m in model.messages[:20]
            ]
        }
        
        prompt = f"""
Act as a Software Architect reviewing a sequence diagram.

## Current Sequence:
{json.dumps(summary, indent=2)}

## Your Task:
1. **Improve Title**: Give a business-friendly title for this interaction
2. **Order Participants**: Suggest optimal left-to-right ordering (actors first, then controllers, services, repositories, databases)
3. **Add Return Values**: For key methods, suggest what they likely return

## Output Format (JSON ONLY):
{{
    "title": "Customer Places Order",
    "participant_order": ["Customer", "OrderController", "OrderService", "OrderRepository", "Database"],
    "return_values": {{
        "OrderService.create_order": "Order",
        "OrderRepository.save": "order_id"
    }},
    "notes": ["Optional: any important notes to add to diagram"]
}}

Return ONLY valid JSON.
"""
        
        try:
            messages = [
                SystemMessage(content="You are a JSON-only sequence diagram assistant."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = self._clean_json(response.content)
            data = json.loads(content)
            
            model.title = data.get('title', model.title)
            
            if 'participant_order' in data:
                order_map = {name: i for i, name in enumerate(data['participant_order'])}
                for p in model.participants:
                    p.order = order_map.get(p.name, 99)
                model.participants.sort(key=lambda p: p.order)
            
            if 'return_values' in data:
                for msg in model.messages:
                    key = f"{msg.to_participant}.{msg.method_name}"
                    if key in data['return_values']:
                        msg.return_value = data['return_values'][key]
            
            return model
            
        except Exception as e:
            logging.warning(f"⚠️ Sequence enrichment failed: {e}")
            return model
    
    def _clean_json(self, content: str) -> str:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return content.strip()


# ============================================================
# PLANTUML CONVERTER
# ============================================================

class SequencePlantUMLConverter:
    """Converts SequenceModel to PlantUML sequence diagram syntax"""
    
    STEREOTYPE_STYLES = {
        'actor': 'actor',
        'boundary': 'boundary',
        'control': 'control',
        'entity': 'entity',
        'database': 'database',
        'participant': 'participant'
    }
    
    def convert(self, model: SequenceModel) -> str:
        """Generate PlantUML sequence diagram"""
        lines = [
            "@startuml",
            f"title {model.title}",
            "",
            "' -- Styling --",
            "skinparam sequenceArrowThickness 2",
            "skinparam participantPadding 20",
            "skinparam boxPadding 10",
            "skinparam sequenceGroupBorderColor #666666",
            "autonumber",
            ""
        ]
        
        lines.append("' -- Participants --")
        for p in sorted(model.participants, key=lambda x: x.order):
            ptype = self.STEREOTYPE_STYLES.get(p.type, 'participant')
            alias = p.alias or self._safe_id(p.name)
            lines.append(f'{ptype} "{p.name}" as {alias}')
        
        lines.append("")
        lines.append("' -- Sequence --")
        
        active_blocks: List[Dict] = []
        
        for msg in model.messages:
            self._handle_control_flow(lines, msg, active_blocks)
            
            from_alias = self._safe_id(msg.from_participant)
            to_alias = self._safe_id(msg.to_participant)
            
            if msg.is_async:
                arrow = "->>"
            elif msg.is_self_call:
                arrow = "->"
            else:
                arrow = "->"
            
            args_str = ", ".join(msg.arguments[:3]) if msg.arguments else ""
            method_str = f"{msg.method_name}({args_str})"
            
            if msg.is_self_call or from_alias == to_alias:
                lines.append(f"{from_alias} -> {from_alias}: {method_str}")
            else:
                lines.append(f"{from_alias} {arrow} {to_alias}: {method_str}")
            
            if not msg.is_self_call and from_alias != to_alias:
                lines.append(f"activate {to_alias}")
            
            if msg.return_value:
                lines.append(f"{to_alias} --> {from_alias}: {msg.return_value}")
            
            if not msg.is_self_call and from_alias != to_alias:
                lines.append(f"deactivate {to_alias}")
        
        for _ in active_blocks:
            lines.append("end")
        
        lines.append("")
        lines.append("@enduml")
        
        return "\n".join(lines)
    
    def _handle_control_flow(
        self, 
        lines: List[str], 
        msg: Message, 
        active_blocks: List[Dict]
    ):
        """Insert control flow markers (alt, loop, opt)"""
        if msg.loop_context and not any(b.get('condition') == msg.loop_context for b in active_blocks):
            lines.append(f"loop {msg.loop_context}")
            active_blocks.append({'type': 'loop', 'condition': msg.loop_context})
    
    def _safe_id(self, name: str) -> str:
        """Create safe PlantUML identifier"""
        return re.sub(r'[^a-zA-Z0-9]', '', name)


# ============================================================
# MAIN SERVICE - Single File
# ============================================================

class SequenceDiagramService:
    """Main service to generate Sequence diagrams from Python code"""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.visitor = None
        self.builder = SequenceBuilder()
        self.enricher = SequenceEnricher(llm) if llm else None
        self.converter = SequencePlantUMLConverter()
    
    def generate(
        self, 
        code: str, 
        entry_method: Optional[str] = None,
        enrich: bool = True
    ) -> str:
        """Generate PlantUML sequence diagram from Python code"""
        try:
            tree = ast.parse(code)
            self.visitor = CallGraphVisitor()
            self.visitor.visit(tree)
            
            logging.info(f"✓ Analyzed {len(self.visitor.classes)} classes, "
                        f"{len(self.visitor.call_sequences)} methods")
            
            if not entry_method:
                entry_method = self._find_entry_point()
            
            if not entry_method:
                return self._empty_diagram("No entry method found")
            
            logging.info(f"✓ Entry point: {entry_method}")
            
            model = self.builder.build(self.visitor, entry_method)
            
            if not model.messages:
                return self._empty_diagram(f"No calls found in {entry_method}")
            
            if enrich and self.enricher and self.llm:
                model = self.enricher.enrich(model, code[:4000])
            
            return self.converter.convert(model)
            
        except SyntaxError as e:
            return f"' ❌ Syntax Error: {e}"
        except Exception as e:
            logging.error(f"❌ Sequence generation failed: {e}")
            import traceback
            traceback.print_exc()
            return f"' ❌ Error: {e}"
    
    # 🆕 NEW: MULTI-MODULE GENERATION
    def generate_modular(
        self,
        file_contents: Dict[str, str],
        enrich: bool = True
    ) -> Dict[str, str]:
        """
        🆕 Generate MULTIPLE sequence diagrams, one per module.
        
        Args:
            file_contents: {"services/order.py": code, ...}
            enrich: Use LLM enrichment
        
        Returns:
            {"Services Order": "@startuml...", "Agent Workflow": "@startuml..."}
        """
        analyzer = ModularSequenceAnalyzer(llm=self.llm)
        modular_model = analyzer.analyze_project(file_contents)
        return analyzer.generate_diagrams(modular_model, enrich=enrich)
    
    def list_available_methods(self) -> List[str]:
        """Return list of methods that can be used as entry points"""
        if not self.visitor:
            return []
        return list(self.visitor.call_sequences.keys())
    
    def _find_entry_point(self) -> Optional[str]:
        """Auto-detect a good entry point method"""
        if not self.visitor:
            return None
        
        candidates = [
            (method, len(calls))
            for method, calls in self.visitor.call_sequences.items()
            if calls and not method.startswith('_')
        ]
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _empty_diagram(self, reason: str) -> str:
        return f"""@startuml
title Sequence Diagram
note over Caller: {reason}
@enduml"""


# ============================================================
# PROJECT-LEVEL ANALYSIS (Original)
# ============================================================

class ProjectSequenceAnalyzer:
    """Analyzes multiple files to build cross-file sequence diagrams"""
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.global_visitor = CallGraphVisitor()
    
    def analyze_files(self, file_contents: Dict[str, str]) -> CallGraphVisitor:
        """Analyze multiple files and merge their call graphs"""
        for filename, code in file_contents.items():
            try:
                tree = ast.parse(code)
                file_visitor = CallGraphVisitor()
                file_visitor.visit(tree)
                
                self.global_visitor.classes.update(file_visitor.classes)
                self.global_visitor.dependencies.update(file_visitor.dependencies)
                
                for method, calls in file_visitor.call_sequences.items():
                    self.global_visitor.call_sequences[method].extend(calls)
                    
            except Exception as e:
                logging.warning(f"⚠️ Failed to analyze {filename}: {e}")
        
        return self.global_visitor
    
    def generate_diagram(self, entry_method: str, enrich: bool = True) -> str:
        """Generate sequence diagram from merged analysis"""
        service = SequenceDiagramService(self.llm)
        service.visitor = self.global_visitor
        
        model = service.builder.build(self.global_visitor, entry_method)
        
        if enrich and service.enricher:
            model = service.enricher.enrich(model, "")
        
        return service.converter.convert(model)