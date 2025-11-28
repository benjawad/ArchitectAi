"""
Design Pattern Detection & Recommendation System

Detects existing patterns and suggests where to apply new ones.
Uses AST for deterministic detection + AI for confidence scoring.
"""

import ast
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


@dataclass
class PatternDetection:
    """Represents a detected design pattern"""
    pattern: str
    location: str  # "ClassName" or "ClassName.method"
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # What indicates this pattern
    justification: str  # AI-generated explanation
    code_snippet: Optional[str] = None


@dataclass
class PatternRecommendation:
    """Represents a suggested pattern to apply"""
    pattern: str
    location: str
    reason: str  # Why this pattern would help
    benefit: str  # What improvement it brings
    complexity_reduction: int  # Estimated % improvement
    implementation_hint: str  # How to implement
    before_uml: Optional[str] = None  # Current structure as UML
    after_uml: Optional[str] = None  # Recommended structure as UML


class PatternDetectorAST(ast.NodeVisitor):
    """
    Deterministic pattern detection using AST analysis.
    Fast, no AI required for initial detection.
    """
    
    def __init__(self, code: str):
        self.code = code
        self.detections: List[PatternDetection] = []
        self.classes: Dict[str, Dict] = {}
        self.current_class = None
    
    def analyze(self) -> List[PatternDetection]:
        """Run all pattern detections"""
        try:
            tree = ast.parse(self.code)
            self.visit(tree)
            
            # Run pattern detectors
            self._detect_singleton()
            self._detect_factory()
            self._detect_strategy()
            self._detect_observer()
            self._detect_builder()
            self._detect_adapter()
            
            return self.detections
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []
    
    def visit_ClassDef(self, node):
        """Collect class information"""
        class_info = {
            "name": node.name,
            "bases": [self._get_name(b) for b in node.bases],
            "methods": [],
            "class_vars": [],
            "decorators": [self._get_name(d) for d in node.decorator_list],
            "has_new": False,
            "has_init": False,
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_info["methods"].append({
                    "name": item.name,
                    "args": [arg.arg for arg in item.args.args],
                    "decorators": [self._get_name(d) for d in item.decorator_list],
                    "returns_self": self._returns_self(item),
                })
                if item.name == "__new__":
                    class_info["has_new"] = True
                if item.name == "__init__":
                    class_info["has_init"] = True
            
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info["class_vars"].append(target.id)
        
        self.classes[node.name] = class_info
        self.generic_visit(node)
    
    def _detect_singleton(self):
        """Detect Singleton pattern"""
        for name, info in self.classes.items():
            evidence = []
            score = 0.0
            
            # Check for __new__ override (strong indicator)
            if info["has_new"]:
                evidence.append("Overrides __new__ method")
                score += 0.4
            
            # Check for _instance class variable
            instance_vars = [v for v in info["class_vars"] 
                           if "instance" in v.lower() or v.startswith("_")]
            if instance_vars:
                evidence.append(f"Has instance variable: {instance_vars[0]}")
                score += 0.3
            
            # Check for get_instance or getInstance method
            get_methods = [m for m in info["methods"] 
                          if "instance" in m["name"].lower()]
            if get_methods:
                evidence.append(f"Has get_instance method: {get_methods[0]['name']}")
                score += 0.3
            
            # Check for private constructor pattern
            if any(m["name"] == "__init__" and m["decorators"] for m in info["methods"]):
                evidence.append("Protected constructor")
                score += 0.2
            
            if score >= 0.5:
                self.detections.append(PatternDetection(
                    pattern="Singleton",
                    location=name,
                    confidence=min(score, 1.0),
                    evidence=evidence,
                    justification="",  # Will be filled by AI
                    code_snippet=self._extract_class_code(name)
                ))
    
    def _detect_factory(self):
        """Detect Factory pattern"""
        for name, info in self.classes.items():
            evidence = []
            score = 0.0
            
            # Check for "Factory" in name
            if "factory" in name.lower():
                evidence.append("Class name contains 'Factory'")
                score += 0.3
            
            # Check for create/make methods
            create_methods = [m for m in info["methods"] 
                            if any(keyword in m["name"].lower() 
                                  for keyword in ["create", "make", "build", "get"])]
            if create_methods:
                evidence.append(f"Has creation methods: {[m['name'] for m in create_methods]}")
                score += 0.4
            
            # Check if methods return different types (polymorphism)
            # This is a heuristic - proper detection needs type analysis
            if len(create_methods) >= 2:
                evidence.append("Multiple factory methods suggest polymorphic creation")
                score += 0.3
            
            if score >= 0.5:
                self.detections.append(PatternDetection(
                    pattern="Factory",
                    location=name,
                    confidence=min(score, 1.0),
                    evidence=evidence,
                    justification="",
                    code_snippet=self._extract_class_code(name)
                ))
    
    def _detect_strategy(self):
        """Detect Strategy pattern"""
        # Look for interface-like classes (ABC) and multiple implementations
        abc_classes = [name for name, info in self.classes.items() 
                      if "ABC" in info["bases"] or "abc" in str(info["bases"]).lower()]
        
        for abc_name in abc_classes:
            # Find classes that inherit from this ABC
            implementations = [name for name, info in self.classes.items() 
                             if abc_name in info["bases"]]
            
            if len(implementations) >= 2:
                evidence = [
                    f"Abstract class: {abc_name}",
                    f"Implementations: {implementations}",
                    "Multiple interchangeable implementations"
                ]
                
                self.detections.append(PatternDetection(
                    pattern="Strategy",
                    location=abc_name,
                    confidence=0.85,
                    evidence=evidence,
                    justification="",
                    code_snippet=self._extract_class_code(abc_name)
                ))
    
    def _detect_observer(self):
        """Detect Observer pattern"""
        for name, info in self.classes.items():
            evidence = []
            score = 0.0
            
            # Look for subscribe/notify methods
            observer_methods = [m for m in info["methods"] 
                              if any(keyword in m["name"].lower() 
                                    for keyword in ["subscribe", "notify", "attach", 
                                                   "detach", "observer", "listener"])]
            
            if observer_methods:
                evidence.append(f"Observer methods: {[m['name'] for m in observer_methods]}")
                score += 0.5
            
            # Look for list of observers
            observer_lists = [v for v in info["class_vars"] 
                            if any(keyword in v.lower() 
                                  for keyword in ["observer", "listener", "subscriber"])]
            
            if observer_lists:
                evidence.append(f"Observer collection: {observer_lists}")
                score += 0.4
            
            if score >= 0.6:
                self.detections.append(PatternDetection(
                    pattern="Observer",
                    location=name,
                    confidence=min(score, 1.0),
                    evidence=evidence,
                    justification="",
                    code_snippet=self._extract_class_code(name)
                ))
    
    def _detect_builder(self):
        """Detect Builder pattern"""
        for name, info in self.classes.items():
            evidence = []
            score = 0.0
            
            # Check for "Builder" in name
            if "builder" in name.lower():
                evidence.append("Class name contains 'Builder'")
                score += 0.3
            
            # Check for fluent interface (methods returning self)
            fluent_methods = [m for m in info["methods"] if m["returns_self"]]
            if len(fluent_methods) >= 3:
                evidence.append(f"Fluent interface: {len(fluent_methods)} methods return self")
                score += 0.5
            
            # Check for build() method
            if any(m["name"] == "build" for m in info["methods"]):
                evidence.append("Has build() method")
                score += 0.3
            
            if score >= 0.5:
                self.detections.append(PatternDetection(
                    pattern="Builder",
                    location=name,
                    confidence=min(score, 1.0),
                    evidence=evidence,
                    justification="",
                    code_snippet=self._extract_class_code(name)
                ))
    
    def _detect_adapter(self):
        """Detect Adapter pattern"""
        for name, info in self.classes.items():
            evidence = []
            score = 0.0
            
            # Check for "Adapter" or "Wrapper" in name
            if any(keyword in name.lower() for keyword in ["adapter", "wrapper"]):
                evidence.append(f"Class name suggests adapter: {name}")
                score += 0.4
            
            # Check for composition (has instance variable of another class)
            # This is heuristic - needs better type analysis
            if info["has_init"] and len(info["methods"]) > 2:
                evidence.append("Has composition and delegates methods")
                score += 0.3
            
            # Check if class has same method names as potential adaptee
            # (requires cross-class analysis, simplified here)
            if len(info["methods"]) >= 3:
                score += 0.2
            
            if score >= 0.5:
                self.detections.append(PatternDetection(
                    pattern="Adapter",
                    location=name,
                    confidence=min(score, 1.0),
                    evidence=evidence,
                    justification="",
                    code_snippet=self._extract_class_code(name)
                ))
    
    # Helper methods
    def _get_name(self, node) -> str:
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def _returns_self(self, func_node) -> bool:
        """Check if function returns self"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                if node.value.id == "self":
                    return True
        return False
    
    def _extract_class_code(self, class_name: str) -> str:
        """Extract source code for a specific class"""
        try:
            lines = self.code.split("\n")
            tree = ast.parse(self.code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 10
                    return "\n".join(lines[start:end])
        except:
            pass
        return ""


class PatternRecommender:
    """
    Analyzes code structure to recommend where patterns would help.
    Uses code smells and structural analysis.
    """
    
    def __init__(self, llm=None):
        self.recommendations: List[PatternRecommendation] = []
        self.llm = llm
    
    def analyze(self, structure: List[Dict], code: str) -> List[PatternRecommendation]:
        """Generate pattern recommendations"""
        self.recommendations = []
        
        self._recommend_strategy(structure, code)
        self._recommend_factory(structure, code)
        self._recommend_singleton(structure, code)
        self._recommend_observer(structure, code)
        
        return self.recommendations
    
    def generate_recommendation_uml(self, recommendation: PatternRecommendation, structure: List[Dict], code: str) -> tuple[str, str]:
        """
        Generate before/after UML for a recommendation.
        Returns: (before_uml, after_uml)
        """
        if self.llm:
            return self._generate_ai_uml(recommendation, structure, code)
        else:
            return self._generate_template_uml(recommendation, structure)
    
    def _generate_template_uml(self, rec: PatternRecommendation, structure: List[Dict]) -> tuple[str, str]:
        """Generate template-based UML for recommendations"""
        
        if rec.pattern == "Strategy":
            before = self._strategy_before_uml(rec, structure)
            after = self._strategy_after_uml(rec)
            return before, after
        
        elif rec.pattern == "Factory":
            before = self._factory_before_uml(rec, structure)
            after = self._factory_after_uml(rec)
            return before, after
        
        elif rec.pattern == "Singleton":
            before = self._singleton_before_uml(rec, structure)
            after = self._singleton_after_uml(rec)
            return before, after
        
        elif rec.pattern == "Observer":
            before = self._observer_before_uml(rec, structure)
            after = self._observer_after_uml(rec)
            return before, after
        
        return "", ""
    
    def _strategy_before_uml(self, rec: PatternRecommendation, structure: List[Dict]) -> str:
        """Generate BEFORE UML for Strategy pattern recommendation"""
        class_name = rec.location
        
        return f"""@startuml
title Before: {class_name} with Conditionals
skinparam classAttributeIconSize 0

class {class_name} {{
  + process(type, data)
  - handle_type_a()
  - handle_type_b()
  - handle_type_c()
}}

note right of {class_name}
  ❌ Problem: Multiple if/else branches
  ❌ Hard to add new types
  ❌ Violates Open/Closed Principle
end note

@enduml"""
    
    def _strategy_after_uml(self, rec: PatternRecommendation) -> str:
        """Generate AFTER UML for Strategy pattern recommendation"""
        class_name = rec.location
        
        return f"""@startuml
title After: Strategy Pattern Applied
skinparam classAttributeIconSize 0

interface ProcessingStrategy {{
  + process(data)
}}

class {class_name} {{
  - strategy: ProcessingStrategy
  + set_strategy(s: ProcessingStrategy)
  + execute(data)
}}

class StrategyA {{
  + process(data)
}}

class StrategyB {{
  + process(data)
}}

class StrategyC {{
  + process(data)
}}

ProcessingStrategy <|.. StrategyA
ProcessingStrategy <|.. StrategyB
ProcessingStrategy <|.. StrategyC
{class_name} o-- ProcessingStrategy

note right of ProcessingStrategy
  ✅ Easy to add new strategies
  ✅ Each strategy is independent
  ✅ Follows Open/Closed Principle
  ✅ ~{rec.complexity_reduction}% complexity reduction
end note

@enduml"""
    
    def _factory_before_uml(self, rec: PatternRecommendation, structure: List[Dict]) -> str:
        """Generate BEFORE UML for Factory pattern recommendation"""
        
        return f"""@startuml
title Before: Scattered Object Creation
skinparam classAttributeIconSize 0

class ClientCode {{
  + method1()
  + method2()
  + method3()
}}

class ProductA {{
}}

class ProductB {{
}}

class ProductC {{
}}

ClientCode ..> ProductA : creates directly
ClientCode ..> ProductB : creates directly
ClientCode ..> ProductC : creates directly

note right of ClientCode
  ❌ Object creation scattered everywhere
  ❌ Hard to change instantiation logic
  ❌ Tight coupling to concrete classes
end note

@enduml"""
    
    def _factory_after_uml(self, rec: PatternRecommendation) -> str:
        """Generate AFTER UML for Factory pattern recommendation"""
        
        return f"""@startuml
title After: Factory Pattern Applied
skinparam classAttributeIconSize 0

class ProductFactory {{
  + create_product(type: str): Product
}}

interface Product {{
  + operation()
}}

class ProductA {{
  + operation()
}}

class ProductB {{
  + operation()
}}

class ProductC {{
  + operation()
}}

class ClientCode {{
  - factory: ProductFactory
  + use_product(type: str)
}}

Product <|.. ProductA
Product <|.. ProductB
Product <|.. ProductC
ProductFactory ..> Product : creates
ClientCode o-- ProductFactory

note right of ProductFactory
  ✅ Centralized creation logic
  ✅ Easy to add new products
  ✅ Client decoupled from concrete classes
  ✅ ~{rec.complexity_reduction}% complexity reduction
end note

@enduml"""
    
    def _singleton_before_uml(self, rec: PatternRecommendation, structure: List[Dict]) -> str:
        """Generate BEFORE UML for Singleton pattern recommendation"""
        
        return f"""@startuml
title Before: Multiple Instances Possible
skinparam classAttributeIconSize 0

class SharedResource {{
  + config: Dict
  + connect()
  + disconnect()
}}

class ClientA {{
  - resource: SharedResource
}}

class ClientB {{
  - resource: SharedResource
}}

class ClientC {{
  - resource: SharedResource
}}

ClientA ..> SharedResource : creates new instance
ClientB ..> SharedResource : creates new instance
ClientC ..> SharedResource : creates new instance

note right of SharedResource
  ❌ Multiple instances created
  ❌ Inconsistent state
  ❌ Resource waste
end note

@enduml"""
    
    def _singleton_after_uml(self, rec: PatternRecommendation) -> str:
        """Generate AFTER UML for Singleton pattern recommendation"""
        
        return """@startuml
title After: Singleton Pattern Applied
skinparam classAttributeIconSize 0

class SharedResource {{
  - {static} _instance: SharedResource
  - config: Dict
  + {static} get_instance(): SharedResource
  + connect()
  + disconnect()
}}

class ClientA {{
}}

class ClientB {{
}}

class ClientC {{
}}

ClientA ..> SharedResource : uses singleton
ClientB ..> SharedResource : uses singleton
ClientC ..> SharedResource : uses singleton

note right of SharedResource
  ✅ Single instance guaranteed
  ✅ Global access point
  ✅ Consistent state
  ✅ ~{rec.complexity_reduction}% complexity reduction
end note

@enduml"""
    
    def _observer_before_uml(self, rec: PatternRecommendation, structure: List[Dict]) -> str:
        """Generate BEFORE UML for Observer pattern recommendation"""
        
        return f"""@startuml
title Before: Tight Coupling for Notifications
skinparam classAttributeIconSize 0

class EventSource {{
  + notify_a()
  + notify_b()
  + notify_c()
}}

class ListenerA {{
  + update()
}}

class ListenerB {{
  + update()
}}

class ListenerC {{
  + update()
}}

EventSource --> ListenerA
EventSource --> ListenerB
EventSource --> ListenerC

note right of EventSource
  ❌ Tightly coupled to all listeners
  ❌ Hard to add/remove listeners
  ❌ Manual notification logic
end note

@enduml"""
    
    def _observer_after_uml(self, rec: PatternRecommendation) -> str:
        """Generate AFTER UML for Observer pattern recommendation"""
        
        return f"""@startuml
title After: Observer Pattern Applied
skinparam classAttributeIconSize 0

interface Observer {{
  + update(event)
}}

class Subject {{
  - observers: List<Observer>
  + attach(o: Observer)
  + detach(o: Observer)
  + notify()
}}

class ListenerA {{
  + update(event)
}}

class ListenerB {{
  + update(event)
}}

class ListenerC {{
  + update(event)
}}

Observer <|.. ListenerA
Observer <|.. ListenerB
Observer <|.. ListenerC
Subject o-- Observer

note right of Subject
  ✅ Loosely coupled
  ✅ Dynamic listener management
  ✅ Automatic notifications
  ✅ ~{rec.complexity_reduction}% complexity reduction
end note

@enduml"""
    
    def _generate_ai_uml(self, rec: PatternRecommendation, structure: List[Dict], code: str) -> tuple[str, str]:
        """Generate AI-powered UML for recommendations"""
        try:
            prompt = f"""
Generate two PlantUML class diagrams for a design pattern recommendation.

Pattern to apply: {rec.pattern}
Location: {rec.location}
Reason: {rec.reason}

Current code structure:
{json.dumps(structure[:3], indent=2)}

Generate:
1. BEFORE diagram: Show current problematic structure
2. AFTER diagram: Show improved structure with {rec.pattern} pattern

Output as JSON:
{{
  "before": "@startuml...@enduml",
  "after": "@startuml...@enduml"
}}

Keep diagrams simple (4-6 classes max). Include notes explaining problems/benefits.
OUTPUT ONLY VALID JSON.
"""
            
            messages = [
                SystemMessage(content="You are a UML diagram expert. Output only valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Clean JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            return result.get("before", ""), result.get("after", "")
            
        except Exception as e:
            logger.warning(f"AI UML generation failed: {e}, using templates")
            return self._generate_template_uml(rec, structure)
    
    def _recommend_strategy(self, structure: List[Dict], code: str):
        """Recommend Strategy pattern for complex conditionals"""
        # Look for classes with many if-else statements
        for cls in structure:
            if cls.get("type") == "module":
                continue
            
            # Count conditionals in methods (simplified heuristic)
            conditional_count = code.count("if ") + code.count("elif ")
            
            if conditional_count > 5:  # Threshold
                self.recommendations.append(PatternRecommendation(
                    pattern="Strategy",
                    location=cls["name"],
                    reason="Multiple conditional branches in methods",
                    benefit="Replace conditionals with polymorphism, easier to extend",
                    complexity_reduction=30,
                    implementation_hint="Create strategy interface, extract each branch into separate strategy class"
                ))
    
    def _recommend_factory(self, structure: List[Dict], code: str):
        """Recommend Factory for object creation logic"""
        for cls in structure:
            if cls.get("type") == "module":
                continue
            
            # Look for multiple object instantiations
            if "(" in code and code.count("= ") > 3:
                self.recommendations.append(PatternRecommendation(
                    pattern="Factory",
                    location=cls["name"],
                    reason="Multiple object instantiations scattered in code",
                    benefit="Centralize object creation, easier to modify construction logic",
                    complexity_reduction=20,
                    implementation_hint="Create factory class with create() method for each type"
                ))
    
    def _recommend_singleton(self, structure: List[Dict], code: str):
        """Recommend Singleton for shared resources"""
        # Look for global variables or module-level instances
        if "global " in code or code.count("_instance") > 0:
            self.recommendations.append(PatternRecommendation(
                pattern="Singleton",
                location="Global scope",
                reason="Global state or shared resource management",
                benefit="Control access to shared resource, lazy initialization",
                complexity_reduction=15,
                implementation_hint="Implement __new__ method to control instantiation"
            ))
    
    def _recommend_observer(self, structure: List[Dict], code: str):
        """Recommend Observer for event handling"""
        # Look for callback patterns or manual notification
        if "callback" in code.lower() or "notify" in code.lower():
            self.recommendations.append(PatternRecommendation(
                pattern="Observer",
                location="Event system",
                reason="Manual event notification or callback management",
                benefit="Decouple event producers from consumers, easier to add listeners",
                complexity_reduction=25,
                implementation_hint="Create Subject class with attach/detach/notify methods"
            ))


class PatternEnricher:
    """
    Uses AI to enrich pattern detections with:
    - Confidence scoring
    - Human-readable justification
    - Implementation quality assessment
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def enrich_detections(self, detections: List[PatternDetection], code: str) -> List[PatternDetection]:
        """Add AI-generated justifications to detections"""
        if not self.llm or not detections:
            return detections
        
        enriched = []
        
        for detection in detections:
            try:
                justification = self._generate_justification(detection, code)
                detection.justification = justification
                enriched.append(detection)
            except Exception as e:
                logger.warning(f"Failed to enrich {detection.pattern}: {e}")
                detection.justification = f"Pattern detected based on: {', '.join(detection.evidence)}"
                enriched.append(detection)
        
        return enriched
    
    def _generate_justification(self, detection: PatternDetection, code: str) -> str:
        """Generate AI explanation for pattern detection"""
        prompt = f"""
Analyze this design pattern detection and provide a clear justification.

Pattern: {detection.pattern}
Location: {detection.location}
Evidence: {detection.evidence}

Code snippet:
```python
{detection.code_snippet or "N/A"}
```

Provide a 1-2 sentence justification explaining:
1. Why this is identified as {detection.pattern} pattern
2. What specific code structure confirms it

Keep it concise and technical. Output ONLY the justification text, no preamble.
"""
        
        try:
            messages = [
                SystemMessage(content="You are a design pattern expert. Provide concise technical justifications."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"AI justification failed: {e}")
            return f"Pattern detected based on: {', '.join(detection.evidence)}"


class PatternDetectionService:
    """
    Main service orchestrating pattern detection and recommendation.
    Combines deterministic AST analysis with optional AI enrichment.
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.detector = None
        self.recommender = PatternRecommender()
        self.enricher = PatternEnricher(llm) if llm else None
    
    def analyze_code(self, code: str, enrich: bool = True) -> Dict[str, Any]:
        """
        Analyze code for patterns and recommendations.
        
        Returns:
            {
                "detections": [PatternDetection, ...],
                "recommendations": [PatternRecommendation, ...],
                "summary": {"total_patterns": int, "total_recommendations": int}
            }
        """
        logger.info("🔍 Starting pattern analysis...")
        
        # Step 1: Deterministic detection
        self.detector = PatternDetectorAST(code)
        detections = self.detector.analyze()
        
        logger.info(f"Found {len(detections)} pattern instances")
        
        # Step 2: AI enrichment (optional)
        if enrich and self.enricher and detections:
            logger.info("✨ Enriching with AI justifications...")
            detections = self.enricher.enrich_detections(detections, code)
        
        # Step 3: Generate recommendations
        # For recommendations, we need structure data
        # Use a simple parser for now
        structure = self._simple_structure_parse(code)
        recommendations = self.recommender.analyze(structure, code)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return {
            "detections": [asdict(d) for d in detections],
            "recommendations": [asdict(r) for r in recommendations],
            "summary": {
                "total_patterns": len(detections),
                "total_recommendations": len(recommendations),
                "patterns_found": list(set(d.pattern for d in detections))
            }
        }
    
    def _simple_structure_parse(self, code: str) -> List[Dict]:
        """Quick structure extraction for recommendations"""
        try:
            tree = ast.parse(code)
            structure = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure.append({
                        "name": node.name,
                        "type": "class",
                        "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    })
            
            return structure
        except:
            return []
    
    def format_report(self, analysis: Dict) -> str:
        """Generate beautifully formatted markdown report"""
        lines = []
        
        # Header
        lines.append("# 🏛️ Design Pattern Analysis Report")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Executive Summary Box
        total_patterns = analysis['summary']['total_patterns']
        total_recs = analysis['summary']['total_recommendations']
        
        lines.append("## 📋 Executive Summary")
        lines.append("")
        lines.append("| Metric | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| 🎯 Patterns Detected | **{total_patterns}** |")
        lines.append(f"| 💡 Recommendations | **{total_recs}** |")
        if analysis['summary']['patterns_found']:
            patterns_str = ", ".join(analysis['summary']['patterns_found'])
            lines.append(f"| 🏷️ Pattern Types | {patterns_str} |")
        lines.append("")
        
        # Detected Patterns Section
        lines.append("---")
        lines.append("")
        lines.append("## 🔍 Detected Design Patterns")
        lines.append("")
        
        if analysis["detections"]:
            for idx, det in enumerate(analysis["detections"], 1):
                # Pattern header with icon
                pattern_icons = {
                    "Singleton": "🔒",
                    "Factory": "🏭",
                    "Strategy": "🎯",
                    "Observer": "👀",
                    "Builder": "🔨",
                    "Adapter": "🔌"
                }
                icon = pattern_icons.get(det['pattern'], "📐")
                
                lines.append(f"### {idx}. {icon} {det['pattern']} Pattern")
                lines.append("")
                
                # Info table
                lines.append("| Property | Value |")
                lines.append("|----------|-------|")
                lines.append(f"| **Location** | `{det['location']}` |")
                lines.append(f"| **Confidence** | {det['confidence']:.0%} {'🟢' if det['confidence'] >= 0.8 else '🟡' if det['confidence'] >= 0.6 else '🟠'} |")
                lines.append("")
                
                # Evidence
                lines.append("**📌 Evidence:**")
                lines.append("")
                for ev in det['evidence']:
                    lines.append(f"- ✓ {ev}")
                lines.append("")
                
                # AI Justification
                if det['justification']:
                    lines.append("**🤖 AI Analysis:**")
                    lines.append("")
                    lines.append(f"> {det['justification']}")
                    lines.append("")
                
                # Code snippet if available
                if det.get('code_snippet'):
                    lines.append("<details>")
                    lines.append("<summary>📝 View Code Snippet</summary>")
                    lines.append("")
                    lines.append("```python")
                    lines.append(det['code_snippet'][:500])  # Limit length
                    if len(det['code_snippet']) > 500:
                        lines.append("# ... (truncated)")
                    lines.append("```")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        else:
            lines.append("> ℹ️ No design patterns were detected in the analyzed code.")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Recommendations Section
        lines.append("## 💡 Pattern Recommendations")
        lines.append("")
        
        if analysis["recommendations"]:
            lines.append("The following design patterns are recommended to improve code quality:")
            lines.append("")
            
            for idx, rec in enumerate(analysis["recommendations"], 1):
                # Pattern header with icon
                pattern_icons = {
                    "Singleton": "🔒",
                    "Factory": "🏭",
                    "Strategy": "🎯",
                    "Observer": "👀",
                    "Builder": "🔨",
                    "Adapter": "🔌"
                }
                icon = pattern_icons.get(rec['pattern'], "📐")
                
                lines.append(f"### {idx}. {icon} Apply {rec['pattern']} Pattern")
                lines.append("")
                
                # Recommendation details
                lines.append("| Aspect | Details |")
                lines.append("|--------|---------|")
                lines.append(f"| **📍 Location** | `{rec['location']}` |")
                lines.append(f"| **⚠️ Problem** | {rec['reason']} |")
                lines.append(f"| **✅ Benefit** | {rec['benefit']} |")
                lines.append(f"| **📉 Complexity Reduction** | ~{rec['complexity_reduction']}% |")
                lines.append("")
                
                # Implementation guide
                lines.append("**🔧 Implementation Guide:**")
                lines.append("")
                lines.append(f"> {rec['implementation_hint']}")
                lines.append("")
                
                # Visual indicator for impact
                impact = rec['complexity_reduction']
                if impact >= 30:
                    impact_label = "🟢 **High Impact**"
                elif impact >= 20:
                    impact_label = "🟡 **Medium Impact**"
                else:
                    impact_label = "🟠 **Low Impact**"
                
                lines.append(f"**Impact Level:** {impact_label}")
                lines.append("")
                
                lines.append("---")
                lines.append("")
        else:
            lines.append("> ✨ Your code is well-structured! No immediate pattern recommendations.")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Footer with tips
        lines.append("## 📚 Additional Resources")
        lines.append("")
        lines.append("**Design Pattern Categories:**")
        lines.append("")
        lines.append("- 🎨 **Creational**: Singleton, Factory, Builder - Object creation mechanisms")
        lines.append("- 🏗️ **Structural**: Adapter, Decorator, Facade - Object composition")  
        lines.append("- 🔄 **Behavioral**: Strategy, Observer, Command - Object interaction")
        lines.append("")
        
        lines.append("**Best Practices:**")
        lines.append("")
        lines.append("1. ✅ Apply patterns when they solve a specific problem")
        lines.append("2. ⚠️ Avoid over-engineering with unnecessary patterns")
        lines.append("3. 📖 Document pattern usage for team understanding")
        lines.append("4. 🧪 Test pattern implementations thoroughly")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by ArchitectAI Pattern Detection System*")
        
        return "\n".join(lines)


# Convenience function for direct use
def detect_patterns(code: str, llm=None, enrich: bool = True) -> Dict[str, Any]:
    """
    Quick pattern detection function.
    
    Args:
        code: Python source code to analyze
        llm: Optional LLM for enrichment
        enrich: Whether to use AI for justifications
    
    Returns:
        Analysis dictionary with detections and recommendations
    """
    service = PatternDetectionService(llm=llm)
    return service.analyze_code(code, enrich=enrich)