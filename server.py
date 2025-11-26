import sys
import ast
import logging
import traceback
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import json

from services.code_generation_service import CodeGenerator
from services.project_service import ProjectAnalyzer
from services.refactoring_service import RefactoringAdvisor 
import os 
from dotenv import load_dotenv

from services.sequence_service import ProjectSequenceAnalyzer, SequenceDiagramService
from services.usecase_service import UseCaseDiagramService

load_dotenv()

# --- 🛡️ PROTOCOL PROTECTION & LOGGING SETUP 🛡️ ---

os.environ.setdefault("MODAL_TOKEN_ID", os.getenv("MODAL_TOKEN_ID", ""))
os.environ.setdefault("MODAL_TOKEN_SECRET", os.getenv("MODAL_TOKEN_SECRET", ""))

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(message)s'
)

for lib in ["httpx", "httpcore", "asyncio"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# 1. Initialize MCP Server
mcp = FastMCP("ArchitectAI", dependencies=["langchain-openai", "langchain-core"])

# 2. Import Core Logic
try:
    from core.llm_factory import create_sambanova_llm, create_nebius_llm, create_openai_llm
    from services.filesystem_service import FileSystemVisitor, TreeFormatter
    from services.architecture_service import (
        ArchitectureVisitor, FastTypeEnricher, DeterministicPlantUMLConverter
    )
    logging.info("✅ All services imported successfully")
except ImportError as e:
    logging.error(f"❌ Critical Import Error: {e}")
    raise

try:
    import modal
    modal_function = modal.Function.from_name("architect-ai-surgeon", "safe_refactor_and_test")
    logging.info("☁️  Modal Function imported successfully.")
except ImportError:
    logging.warning("⚠️ Could not import modal_executor function. Cloud features disabled.")
    modal_function = None
# --- 🛡️ SANDBOX CONFIGURATION 🛡️ ---
def _find_project_root(start_path: Path) -> Path:
    """
    Searches upwards for a marker that indicates the project root
    (like .git, pyproject.toml, or requirements.txt).
    Falls back to CWD if nothing is found.
    """
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        # Check for common project markers
        if (parent / ".git").exists() or \
           (parent / "pyproject.toml").exists() or \
           (parent / "requirements.txt").exists():
            return parent
            
    # Fallback: If no marker found, use the Current Working Directory (where you ran the command)
    return Path.cwd().resolve()

# 1. Start searching from the location of this script
_script_location = Path(__file__).parent
# 2. Determine the actual project root dynamically
SANDBOX_ROOT = _find_project_root(_script_location)

logging.info(f"🌍 Project Root Detected at: {SANDBOX_ROOT}")

BLOCKED_PATHS = {
    SANDBOX_ROOT / ".venv",
    SANDBOX_ROOT / "venv",
    SANDBOX_ROOT / ".git",
    SANDBOX_ROOT / ".env",
    SANDBOX_ROOT / "__pycache__",
}

# --- 🔧 SINGLETON LLM CLIENT 🔧 ---
class LLMClientSingleton:
    """
    Singleton pattern for LLM client.
    Ensures only ONE instance of the LLM client is created and reused.
    """
    _instance = None
    _llm_client = None
    _current_provider = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logging.info("🏗️  Creating LLM Singleton instance...")
        return cls._instance
    
    def get_client(self, preferred_provider: str = "sambanova", temperature: float = 0.0):
        """
        Get or initialize the LLM client.
        Only creates a new client if provider changes.
        """
        # Return cached client if provider hasn't changed
        if self._llm_client is not None and self._current_provider == preferred_provider:
            logging.debug(f"♻️  Reusing cached {preferred_provider} client")
            return self._llm_client
        
        # Create new client if provider changed
        logging.info(f"🔄 Initializing {preferred_provider} LLM client...")
        
        strategies = [
            ("sambanova", create_sambanova_llm),
            ("nebius", create_nebius_llm),
            ("openai", create_openai_llm)
        ]
        
        if preferred_provider == "nebius":
            strategies = [
                ("nebius", create_nebius_llm),
                ("openai", create_openai_llm),
                ("sambanova", create_sambanova_llm)
            ]
        elif preferred_provider == "openai":
            strategies = [
                ("openai", create_openai_llm),
                ("nebius", create_nebius_llm),
                ("sambanova", create_sambanova_llm)
            ]

        for name, factory in strategies:
            try:
                logging.info(f"🔌 Attempting to connect to {name}...")
                self._llm_client = factory(temperature=temperature)
                self._current_provider = name
                logging.info(f"✅ Connected to {name} (cached for reuse)")
                return self._llm_client
            except Exception as e:
                logging.warning(f"⚠️  {name} failed: {str(e)[:100]}...")
        
        logging.error("❌ No LLM provider available!")
        self._llm_client = None
        self._current_provider = None
        return None
    
    def reset(self):
        """Force reset the cached client (useful for testing)."""
        logging.info("🔄 Resetting LLM Singleton...")
        self._llm_client = None
        self._current_provider = None

# Global singleton instance
_llm_singleton = LLMClientSingleton()

def _validate_path(user_path: str, operation: str = "read") -> tuple[bool, Path, str]:
    """
    Validates that a user-provided path is within the sandbox.
    
    Returns:
        (is_valid, resolved_path, error_message)
    """
    try:
        requested = Path(user_path).resolve()
        sandbox_root = SANDBOX_ROOT.resolve()
        
        try:
            requested.relative_to(sandbox_root)
        except ValueError:
            return False, requested, f"❌ Access Denied: Path '{user_path}' is outside project directory"
        
        if requested in BLOCKED_PATHS or any(requested.is_relative_to(bp) for bp in BLOCKED_PATHS if bp.exists()):
            return False, requested, f"❌ Access Denied: Cannot access '{requested.name}' (protected directory)"
        
        if not requested.exists():
            return False, requested, f"❌ Path not found: '{user_path}'"
        
        if operation == "read":
            if requested.is_dir():
                return False, requested, f"❌ '{user_path}' is a directory. Use list_project_structure() instead"
            if not requested.is_file():
                return False, requested, f"❌ '{user_path}' is not a regular file"
        
        elif operation == "list":
            if not requested.is_dir():
                return False, requested, f"❌ '{user_path}' is not a directory"
        
        logging.debug(f"✓ Path validation passed: {requested}")
        return True, requested, ""
        
    except Exception as e:
        return False, Path(user_path), f"❌ Path validation error: {str(e)}"

# --- MCP TOOLS ---

@mcp.tool()
def generate_architecture_diagram(code: str, enrich: bool = True, provider: str = "sambanova") -> str:
    """
    Analyzes Python code and generates a PlantUML Class Diagram.
    Uses singleton LLM client for efficiency.
    """
    try:
        # 1. Static Analysis (Fast & Deterministic)
        try:
            tree = ast.parse(code)
            visitor = ArchitectureVisitor()
            visitor.visit(tree)
            logging.info(f"✓ Static analysis: {len(visitor.structure)} classes found")
        except SyntaxError as se:
            return f"❌ Syntax Error in code: {se}"

        # 2. AI Enrichment (Optional & Hybrid) - Using Singleton
        if enrich and visitor.structure:
            # 🔧 GET SINGLETON CLIENT
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
            
            if llm:
                try:
                    enricher = FastTypeEnricher(llm)
                    visitor.structure = enricher.enrich(code, visitor.structure)
                    logging.info("✓ Type enrichment complete")
                except Exception as e:
                    logging.error(f"⚠️ Enrichment failed (skipping): {e}")
            else:
                logging.warning("⚠️ Skipping enrichment: No LLM provider available")

        # 3. Visualization (Deterministic)
        converter = DeterministicPlantUMLConverter()
        result = converter.convert(visitor.structure)
        logging.info("✓ Diagram generated successfully")
        return result

    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        traceback.print_exc(file=sys.stderr)
        return f"❌ Processing failed: {str(e)}"

@mcp.tool()
def list_project_structure(path: str = ".", style: str = "tree") -> str:
    """
    Deterministically maps the folder structure.
    
    Args:
        path: The root directory to list.
        style: 'tree' for visual string, 'json' for structured data (easier for coding).
    
    🛡️ Sandboxed: Can only list directories within the project.
    """
    try:
        # 🛡️ SANDBOX CHECK
        is_valid, resolved_path, error_msg = _validate_path(path, operation="list")
        if not is_valid:
            logging.warning(f"🚫 Sandbox violation attempt: {error_msg}")
            return error_msg
        
        # 1. Get Raw Data (Dictionary)
        visitor = FileSystemVisitor()
        tree_data = visitor.visit(str(resolved_path))
        
        # 2. Format Output based on Style
        if style == "json":
            result = json.dumps(tree_data, indent=2)
        else:
            formatter = TreeFormatter()
            result = formatter.format(tree_data)
        
        logging.info(f"✓ Listed directory: {resolved_path} (style={style})")
        return result
        
    except Exception as e:
        logging.error(f"❌ Directory listing failed: {e}")
        return f"Error listing directory: {e}"
@mcp.tool()
def read_file(path: str) -> str:
    """
    Reads the full content of a specific file. 
    Use this to inspect code before generating diagrams.
    🛡️ Sandboxed: Can only read files within the project.
    """
    try:
        is_valid, resolved_path, error_msg = _validate_path(path, operation="read")
        if not is_valid:
            logging.warning(f"🚫 Sandbox violation attempt: {error_msg}")
            return error_msg
        
        content = resolved_path.read_text(encoding='utf-8', errors='replace')
        logging.info(f"✓ Read file: {resolved_path.name} ({len(content)} chars)")
        return content

    except Exception as e:
        logging.error(f"❌ Read failed: {e}")
        return f"❌ Error reading file: {str(e)}"

@mcp.tool()
def generate_full_project_diagram(path: str = ".", enrich: bool = False, provider: str = "sambanova") -> str:
    """
    Analyzes the ENTIRE project directory and generates a massive Class Diagram.
    
    Args:
        path: Root directory to analyze.
        enrich: If True, uses AI to infer types (Can be slow for big projects).
    """
    try:
        # 1. Security Check
        is_valid, resolved_path, error_msg = _validate_path(path, operation="list")
        if not is_valid:
            return error_msg
            
        # 2. Analyze Project (The Aggregation)
        analyzer = ProjectAnalyzer(resolved_path)
        full_structure = analyzer.analyze()
        
        if not full_structure:
            return "⚠️ No Python code found in this directory."

        # 3. AI Enrichment (Optional Batch Processing)
        # Warning: For huge projects, we might want to limit this or do it in chunks.
        if enrich:
            # We construct a "Virtual" code context combining all files? 
            # OR we just skip enrichment for the full map to save tokens.
            # For now, let's SKIP enrichment for the full view to be fast.
            logging.info("ℹ️ Skipping AI enrichment for full project to ensure speed.")
            pass 

        # 4. Convert to PlantUML
        converter = DeterministicPlantUMLConverter()
        puml_code = converter.convert(full_structure)
        
        logging.info(f"✓ Generated full project diagram ({len(full_structure)} items)")
        return puml_code

    except Exception as e:
        logging.error(f"❌ Project analysis failed: {e}")
        return f"Error analyzing project: {str(e)}"


@mcp.tool()
def propose_architecture_refactoring(path: str = ".") -> str:
    """
    Analyzes the project and returns a JSON proposal with 'proposed_uml'.
    Use this to visualize changes BEFORE applying them.
    """
    try:
        # 1. Scan Project
        is_valid, resolved_path, error_msg = _validate_path(path, operation="list")
        if not is_valid: return error_msg
        
        analyzer = ProjectAnalyzer(resolved_path)
        structure = analyzer.analyze()
        
        if not structure: return "Error: No code found."

        # 2. Generate Proposal
        advisor = RefactoringAdvisor()
        proposal = advisor.propose_improvement(structure)
        
        # Return as pretty string for the Agent/User to read
        return json.dumps(proposal, indent=2)

    except Exception as e:
        return f"Error generating proposal: {e}"
    

@mcp.tool()
def apply_refactoring(file_path: str, instruction: str) -> str:
    """
    APPLIES CHANGES TO THE FILE. 
    WARNING: This overwrites the file. Use only on 'ai_architect - Copy'.
    
    Args:
        file_path: Relative path (e.g., 'core/llm_factory.py').
        instruction: What to do (e.g., 'Refactor to Singleton pattern').
    """
    try:
        # 1. Security Check        
        target_file = SANDBOX_ROOT / file_path
        
        if not target_file.exists():
            return f"❌ File not found: {file_path}"

        original_code = target_file.read_text(encoding='utf-8')
        
     
        generator = CodeGenerator()
        new_code = generator.generate_refactored_code(original_code, instruction, file_path)
        
        saved_path = generator.save_code(file_path, new_code, SANDBOX_ROOT)
        
        return f"✅ File updated successfully: {saved_path}"

    except Exception as e:
        return f"❌ Error: {e}"
    

@mcp.tool()
def apply_refactoring_safely(
    file_path: str, 
    instruction: str,
    test_file: str = None
) -> str:
    """
    Safely refactors code with automatic testing.
    Uses Modal sandbox if available, falls back to local LLM if not.
    
    Args:
        file_path: Path to file to refactor
        instruction: Refactoring instructions
        test_file: Optional path to test file
    """
    try:
        # 1. Read original code
        target = SANDBOX_ROOT / file_path
        if not target.exists():
            return f"❌ File not found: {file_path}"
        
        original_code = target.read_text(encoding='utf-8')
        
        # 2. Read tests if provided
        test_code = None
        if test_file:
            test_path = SANDBOX_ROOT / test_file
            if test_path.exists():
                test_code = test_path.read_text(encoding='utf-8')
        
        # 3. Try Modal sandbox first
        if modal_function is not None:
            try:
                logging.info("🚀 Sending to Modal sandbox...")
                result = modal_function.remote(
                    original_code,
                    instruction,
                    test_code
                )
                
                # 4. Check results
                if not result["success"]:
                    return f"❌ Refactoring failed: {result['error']}"
                
                if not result["test_results"]["passed"]:
                    return f"""
⚠️ Refactoring succeeded but tests FAILED:
{result['test_results']['output']}

Code was NOT saved. Review and fix tests first.
"""
                
                # 5. Save only if tests passed
                target.write_text(result["new_code"], encoding='utf-8')
                
                return f"""
✅ Refactoring completed successfully in Modal sandbox!
📊 Tests: PASSED ✓
💾 File saved: {file_path}

Test output:
{result['test_results']['output']}
"""
            except Exception as modal_error:
                logging.warning(f"⚠️ Modal sandbox failed: {str(modal_error)[:100]}... Falling back to local refactoring.")
        
        # 4. Fallback: Local LLM refactoring (no testing)
        logging.info("🔧 Using local LLM for refactoring (testing disabled)...")
        llm = _llm_singleton.get_client(preferred_provider="openai", temperature=0.0)
        
        if llm is None:
            return "❌ No LLM available for refactoring (Modal and local LLM both failed)"
        
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_msg = SystemMessage(content=
            "You are a Senior Python Refactoring Engineer. "
            "Rewrite the provided code to meet the user's architectural instructions "
            "while preserving the original business logic."
        )
        
        user_msg = HumanMessage(content=f"""
Refactor this Python code according to these instructions:

**Instructions:** {instruction}

**Original Code:**
```python
{original_code}
```

Return ONLY the refactored code, no explanations.
""")
        
        try:
            response = llm.invoke([system_msg, user_msg])
            new_code = response.content
            
            # Clean up markdown if present
            if "```python" in new_code:
                new_code = new_code.split("```python")[1].split("```")[0].strip()
            elif "```" in new_code:
                new_code = new_code.split("```")[1].split("```")[0].strip()
            
            # Save the refactored code
            target.write_text(new_code, encoding='utf-8')
            
            return f"""
✅ Refactoring completed successfully (Local LLM)!
💾 File saved: {file_path}
⚠️ Note: Testing was skipped (Modal not available). Please verify manually.

Refactored code preview:
```python
{new_code[:500]}...
```
"""
        except Exception as llm_error:
            return f"❌ Local LLM refactoring failed: {str(llm_error)}"
        
    except Exception as e:
        logging.error(f"❌ Refactoring error: {e}")
        return f"❌ Error: {e}"

@mcp.tool()
def generate_usecase_diagram(code: str, enrich: bool = True, provider: str = "sambanova") -> str:
    """
    Analyzes Python code and generates a PlantUML Use Case Diagram.
    
    Detects:
    - Service classes (patterns: *Service, *Controller, *Handler, *API)
    - FastAPI/Flask endpoints (@app.get, @router.post, etc.)
    - Public methods as potential use cases
    
    Args:
        code: Python source code to analyze
        enrich: If True, uses AI to infer actors and relationships
        provider: LLM provider ("sambanova", "nebius", "openai")
    
    Returns:
        PlantUML use case diagram string
    """
    try:
        # Get LLM client (using singleton)
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        # Generate diagram
        service = UseCaseDiagramService(llm=llm)
        result = service.generate(code, enrich=enrich)
        
        logging.info("✓ Use case diagram generated successfully")
        return result
        
    except Exception as e:
        logging.error(f"❌ Use case diagram failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return f"' ❌ Error: {str(e)}"

@mcp.tool()
def generate_full_project_usecase_diagram(path: str = ".", enrich: bool = True, provider: str = "sambanova") -> str:
    """
    Analyzes the ENTIRE project directory and generates a Use Case Diagram.
    
    Scans all .py files for services, controllers, and endpoints.
    
    Args:
        path: Root directory to analyze
        enrich: If True, uses AI to infer actors
        provider: LLM provider
    
    Returns:
        PlantUML use case diagram string
    """
    try:
        # Security check
        is_valid, resolved_path, error_msg = _validate_path(path, operation="list")
        if not is_valid:
            return error_msg
        
        # Collect all Python files
        combined_code = []
        file_count = 0
        
        for file_path in resolved_path.rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules"] for p in parts):
                continue
            try:
                code = file_path.read_text(encoding='utf-8', errors='replace')
                combined_code.append(f"# === File: {file_path.name} ===\n{code}")
                file_count += 1
            except Exception:
                continue
        
        if not combined_code:
            return "' ⚠️ No Python files found in directory"
        
        # Get LLM client
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        # Generate diagram from combined code
        service = UseCaseDiagramService(llm=llm)
        # Limit code size to avoid token limits
        full_code = "\n\n".join(combined_code)[:50000]
        result = service.generate(full_code, enrich=enrich)
        
        logging.info(f"✓ Project use case diagram generated ({file_count} files)")
        return result
        
    except Exception as e:
        logging.error(f"❌ Project use case diagram failed: {e}")
        return f"' ❌ Error: {str(e)}"

@mcp.tool()
def generate_sequence_diagram(
    code: str, 
    entry_method: str = None,
    enrich: bool = True, 
    provider: str = "sambanova"
) -> str:
    """
    Analyzes Python code and generates a PlantUML Sequence Diagram.
    
    Traces method calls through:
    - self.dependency.method() patterns
    - Control flow (if/else, loops, try/except)
    - Async operations
    
    Args:
        code: Python source code to analyze
        entry_method: Starting point (e.g., "OrderService.create_order")
                     If None, auto-detects the method with most calls
        enrich: If True, uses AI to improve readability
        provider: LLM provider ("sambanova", "nebius", "openai")
    
    Returns:
        PlantUML sequence diagram string
    """
    try:
        # Get LLM client
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        # Generate diagram
        service = SequenceDiagramService(llm=llm)
        result = service.generate(code, entry_method=entry_method, enrich=enrich)
        
        logging.info(f"✓ Sequence diagram generated (entry: {entry_method or 'auto'})")
        return result
        
    except Exception as e:
        logging.error(f"❌ Sequence diagram failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return f"' ❌ Error: {str(e)}"


@mcp.tool()
def generate_full_project_sequence_diagram(
    path: str = ".",
    entry_method: str = None,
    enrich: bool = False,
    provider: str = "sambanova"
) -> str:
    """
    Analyzes ENTIRE project and generates a cross-file Sequence Diagram.
    
    Scans all .py files and traces method calls across modules.
    
    Args:
        path: Root directory to analyze
        entry_method: Starting point (e.g., "OrderService.create_order")
        enrich: If True, uses AI to improve readability
        provider: LLM provider
    
    Returns:
        PlantUML sequence diagram string
    """
    try:
        # Security check
        is_valid, resolved_path, error_msg = _validate_path(path, operation="list")
        if not is_valid:
            return error_msg
        
        # Collect all Python files
        file_contents = {}
        for file_path in resolved_path.rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", ".git"] for p in parts):
                continue
            try:
                file_contents[file_path.name] = file_path.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue
        
        if not file_contents:
            return "' ⚠️ No Python files found in directory"
        
        # Get LLM client
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        # Analyze project
        analyzer = ProjectSequenceAnalyzer(llm=llm)
        analyzer.analyze_files(file_contents)
        
        # Auto-detect entry if not provided
        if not entry_method:
            candidates = [
                (m, len(c)) 
                for m, c in analyzer.global_visitor.call_sequences.items()
                if c and not m.startswith('_')
            ]
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                entry_method = candidates[0][0]
        
        if not entry_method:
            # List available methods
            methods = list(analyzer.global_visitor.call_sequences.keys())[:10]
            return f"' ⚠️ No entry method specified. Available: {methods}"
        
        result = analyzer.generate_diagram(entry_method, enrich=enrich)
        
        logging.info(f"✓ Project sequence diagram generated ({len(file_contents)} files)")
        return result
        
    except Exception as e:
        logging.error(f"❌ Project sequence diagram failed: {e}")
        return f"' ❌ Error: {str(e)}"

@mcp.tool()
def list_sequence_entry_points(code: str) -> str:
    """
    Lists all available methods that can be used as sequence diagram entry points.
    
    Use this to discover what methods exist before generating a sequence diagram.
    
    Args:
        code: Python source code to analyze
    
    Returns:
        JSON list of available entry methods with their call counts
    """
    try:
        from services.sequence_service import CallGraphVisitor
        
        tree = ast.parse(code)
        visitor = CallGraphVisitor()
        visitor.visit(tree)
        
        # Build list with call counts
        entry_points = []
        for method, calls in visitor.call_sequences.items():
            if method.startswith('_'):
                continue
            entry_points.append({
                'method': method,
                'calls_count': len(calls),
                'calls_to': list(set(
                    c['target_type'] for c in calls if c.get('target_type')
                ))[:5]
            })
        
        # Sort by call count (most interesting methods first)
        entry_points.sort(key=lambda x: x['calls_count'], reverse=True)
        
        logging.info(f"✓ Found {len(entry_points)} potential entry points")
        return json.dumps({
            'total_methods': len(entry_points),
            'entry_points': entry_points[:20]
        }, indent=2)
        
    except SyntaxError as se:
        return json.dumps({'error': f'Syntax error: {se}'})
    except Exception as e:
        return json.dumps({'error': str(e)})


@mcp.tool()
def list_project_entry_points(path: str = ".") -> str:
    """
    Lists all available methods across the ENTIRE project for sequence diagrams.
    
    Args:
        path: Root directory to analyze
    
    Returns:
        JSON list of available entry methods with their call counts
    """
    try:
        from services.sequence_service import CallGraphVisitor
        
        # Security check
        is_valid, resolved_path, error_msg = _validate_path(path, operation="list")
        if not is_valid:
            return json.dumps({'error': error_msg})
        
        # Analyze all files
        global_visitor = CallGraphVisitor()
        file_count = 0
        
        for file_path in resolved_path.rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__"] for p in parts):
                continue
            try:
                code = file_path.read_text(encoding='utf-8', errors='replace')
                tree = ast.parse(code)
                file_visitor = CallGraphVisitor()
                file_visitor.visit(tree)
                
                # Merge
                global_visitor.classes.update(file_visitor.classes)
                for method, calls in file_visitor.call_sequences.items():
                    global_visitor.call_sequences[method].extend(calls)
                file_count += 1
            except Exception:
                continue
        
        # Build entry points list
        entry_points = []
        for method, calls in global_visitor.call_sequences.items():
            if method.startswith('_') or '.visit_' in method:
                continue
            entry_points.append({
                'method': method,
                'calls_count': len(calls),
                'calls_to': list(set(
                    c['target_type'] for c in calls if c.get('target_type')
                ))[:5]
            })
        
        entry_points.sort(key=lambda x: x['calls_count'], reverse=True)
        
        logging.info(f"✓ Scanned {file_count} files, found {len(entry_points)} entry points")
        return json.dumps({
            'files_analyzed': file_count,
            'total_methods': len(entry_points),
            'entry_points': entry_points[:30]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({'error': str(e)})


# --- ENTRY POINT ---

if __name__ == "__main__":
    logging.info("🚀 MCP Server 'ArchitectAI' starting...")
    logging.info(f"📍 Sandbox Root: {SANDBOX_ROOT}")
    logging.info(f"🚫 Protected Paths: {len(BLOCKED_PATHS)}")
    logging.info("🔧 LLM Client: Singleton Pattern (created once, reused)")
    logging.info("📊 Available Diagram Types:")
    logging.info("   • Class Diagrams      → generate_architecture_diagram()")
    logging.info("   • Use Case Diagrams   → generate_usecase_diagram()")
    logging.info("   • Sequence Diagrams   → generate_sequence_diagram()")
    logging.info("Available providers: sambanova, nebius, openai")

    
    try:
        mcp.run()
    except KeyboardInterrupt:
        logging.info("🛑 Server stopped manually.")
    except Exception as e:
        logging.critical(f"❌ Fatal server error: {e}")
        sys.exit(1)