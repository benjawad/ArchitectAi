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
    Uses AI to generate custom before/after UML diagrams based on actual code.
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
        Generate before/after UML for a recommendation using AI analysis of actual code.
        Returns: (before_uml, after_uml)
        """
        if self.llm:
            logger.info(f"🤖 Using AI to analyze {recommendation.pattern} pattern for {recommendation.location}")
            return self._generate_ai_uml(recommendation, structure, code)
        else:
            logger.info(f"📊 Using structure-based UML for {recommendation.pattern} pattern")
            return self._generate_structure_based_uml(recommendation, structure, code)
    
    def _generate_structure_based_uml(self, rec: PatternRecommendation, structure: List[Dict], code: str) -> tuple[str, str]:
        """Generate UML from actual code structure when AI is not available"""
        
        # Extract actual class from structure
        target_class = None
        for item in structure:
            if item.get("name") == rec.location or rec.location in item.get("name", ""):
                target_class = item
                break
        
        if not target_class:
            # Fallback to first class if location not found
            target_class = structure[0] if structure else {"name": rec.location, "methods": []}
        
        class_name = target_class.get("name", rec.location)
        methods = target_class.get("methods", [])
        
        # Generate BEFORE diagram with actual code
        before = f"""@startuml
title Before: {class_name} - Current Implementation
skinparam classAttributeIconSize 0

class {class_name} {{
"""
        # Add actual methods from the class
        for method in methods[:6]:  # Show up to 6 methods
            before += f"  + {method}()\n"
        
        before += f"""}}

note right of {class_name}
  ❌ Current Issue:
  {rec.reason}
  
  ❌ Problems:
  - Hard to extend
  - High complexity
  - Maintenance burden
  
  Location: {rec.location}
end note

@enduml"""
        
        # Generate AFTER diagram with pattern applied
        after = f"""@startuml
title After: {rec.pattern} Pattern Applied to {class_name}
skinparam classAttributeIconSize 0

{self._generate_pattern_structure(rec.pattern, class_name, methods)}

note bottom
  ✅ Benefits:
  {rec.benefit}
  
  ✅ Improvements:
  - ~{rec.complexity_reduction}% complexity reduction
  - Better separation of concerns
  - Easier to extend and maintain
  
  Pattern: {rec.pattern}
  Implementation: {rec.implementation_hint}
end note

@enduml"""
        
        return before, after
    
    def _generate_pattern_structure(self, pattern: str, class_name: str, methods: List[str]) -> str:
        """Generate pattern-specific structure using actual class name"""
        
        if pattern == "Strategy":
            return f"""interface Strategy {{
  + execute()
}}

class {class_name} {{
  - strategy: Strategy
  + set_strategy(s: Strategy)
  + execute()
}}

class StrategyA {{
  + execute()
}}

class StrategyB {{
  + execute()
}}

Strategy <|.. StrategyA
Strategy <|.. StrategyB
{class_name} o-- Strategy"""
        
        elif pattern == "Factory":
            return f"""interface Product {{
  + operation()
}}

class {class_name}Factory {{
  + create_product(type: str): Product
}}

class ProductA {{
  + operation()
}}

class ProductB {{
  + operation()
}}

Product <|.. ProductA
Product <|.. ProductB
{class_name}Factory ..> Product
{class_name}Factory ..> ProductA
{class_name}Factory ..> ProductB"""
        
        elif pattern == "Singleton":
            return f"""class {class_name} <<Singleton>> {{
  - {{static}} _instance: {class_name}
  - __init__()
  + {{static}} get_instance(): {class_name}
"""
            + "\n".join(f"  + {m}()" for m in methods[:4]) + "\n}"
        
        elif pattern == "Observer":
            return f"""interface Observer {{
  + update(event)
}}

class {class_name} {{
  - observers: List<Observer>
  + attach(o: Observer)
  + detach(o: Observer)
  + notify()
}}

class ObserverA {{
  + update(event)
}}

class ObserverB {{
  + update(event)
}}

Observer <|.. ObserverA
Observer <|.. ObserverB
{class_name} o-- Observer"""
        
        else:
            # Generic pattern structure
            return f"""class {class_name}Improved {{
"""
            + "\n".join(f"  + {m}()" for m in methods[:4]) + "\n}"
    
    def _generate_ai_uml(self, rec: PatternRecommendation, structure: List[Dict], code: str) -> tuple[str, str]:
        """
        🤖 AI-POWERED UML GENERATION
        
        This is the key method that makes diagrams show YOUR actual code,
        not generic templates.
        """
        try:
            # Extract relevant code section for the target class
            relevant_code = self._extract_relevant_code(rec.location, code)
            
            # Get actual class details from structure
            target_class = None
            for item in structure:
                if item.get("name") == rec.location:
                    target_class = item
                    break
            
            # Prepare context for AI
            context = {
                "class_name": rec.location,
                "pattern": rec.pattern,
                "reason": rec.reason,
                "benefit": rec.benefit,
                "complexity_reduction": rec.complexity_reduction,
                "code_snippet": relevant_code[:1500],
                "class_details": target_class if target_class else {},
                "all_classes": [item.get("name") for item in structure if item.get("type") == "class"][:10]
            }
            
            prompt = f"""You are an expert software architect analyzing REAL Python code.

**ACTUAL CODE TO ANALYZE:**
```python
{context['code_snippet']}
```

**CODE CONTEXT:**
- Target class: `{context['class_name']}`
- All classes in project: {', '.join(context['all_classes'])}
- Current problem: {context['reason']}
- Recommended pattern: {context['pattern']}

**YOUR TASK:**
Generate TWO PlantUML class diagrams showing before/after applying {context['pattern']} pattern.

**CRITICAL REQUIREMENTS:**
1. Use ACTUAL class names from the code above (e.g., `{context['class_name']}`, not "ClientCode")
2. Use ACTUAL method names you see in the code
3. Show REAL relationships between classes
4. BEFORE diagram: Show current problematic structure
5. AFTER diagram: Show improved structure with {context['pattern']} pattern

**DIAGRAM GUIDELINES:**
- Keep simple: 4-6 classes maximum
- Use actual names (no "ProductA", "ServiceA" generic names)
- Add notes explaining problems (BEFORE) and benefits (AFTER)
- Include complexity reduction: ~{context['complexity_reduction']}%

**OUTPUT FORMAT (CRITICAL):**
Return ONLY this JSON structure, nothing else:
{{{{
  "before": "@startuml\\ntitle Before: [use actual class name]\\nskinparam classAttributeIconSize 0\\n...\\n@enduml",
  "after": "@startuml\\ntitle After: {context['pattern']} Pattern Applied\\nskinparam classAttributeIconSize 0\\n...\\n@enduml"
}}}}

DO NOT include ```json or ``` markers.
DO NOT add any explanation text.
OUTPUT ONLY THE JSON OBJECT.
"""
            
            messages = [
                SystemMessage(content="You are a UML expert. Analyze actual code and generate precise diagrams using real class/method names. Output only valid JSON, no markdown."),
                HumanMessage(content=prompt)
            ]
            
            logger.info(f"🤖 Sending code to AI for analysis...")
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Clean up response - remove markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Find JSON object boundaries
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
            
            # Parse JSON
            result = json.loads(content)
            before = result.get("before", "")
            after = result.get("after", "")
            
            # Validate UML syntax
            if "@startuml" not in before or "@enduml" not in before:
                logger.warning("❌ AI returned invalid BEFORE UML")
                raise ValueError("Invalid BEFORE UML from AI")
            
            if "@startuml" not in after or "@enduml" not in after:
                logger.warning("❌ AI returned invalid AFTER UML")
                raise ValueError("Invalid AFTER UML from AI")
            
            logger.info(f"✅ AI successfully generated custom UML for {rec.pattern} pattern")
            logger.info(f"   BEFORE diagram: {len(before)} chars")
            logger.info(f"   AFTER diagram: {len(after)} chars")
            
            return before, after
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI response as JSON: {e}")
            logger.error(f"   Response preview: {content[:200] if 'content' in locals() else 'N/A'}")
            return self._generate_structure_based_uml(rec, structure, code)
            
        except Exception as e:
            logger.error(f"❌ AI UML generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("⚠️ Falling back to structure-based generation")
            return self._generate_structure_based_uml(rec, structure, code)
    
    def _extract_relevant_code(self, class_name: str, code: str) -> str:
        """Extract the code section relevant to the class being analyzed"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    lines = code.split("\n")
                    start = node.lineno - 1
                    # Get reasonable chunk of code (not too much)
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 30
                    end = min(end, start + 40)  # Max 40 lines
                    return "\n".join(lines[start:end])
        except Exception as e:
            logger.warning(f"Failed to extract class code: {e}")
        
        # Fallback: search for class definition manually
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if f"class {class_name}" in line:
                return "\n".join(lines[i:min(i+30, len(lines))])
        
        # Last resort: return first chunk of code
        return code[:800]
    
    # Keep the existing _recommend_* methods unchanged
    def _recommend_strategy(self, structure: List[Dict], code: str):
        """Recommend Strategy pattern for complex conditionals"""
        for cls in structure:
            if cls.get("type") == "module":
                continue
            
            conditional_count = code.count("if ") + code.count("elif ")
            
            if conditional_count > 5:
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