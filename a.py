import gradio as gr
import ast
import logging
import io
import os
import json
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from plantuml import PlantUML

from services.pattern_detector import PatternDetectionService, PatternRecommendation
from services.sequence_service import CallGraphVisitor, ProjectSequenceAnalyzer, SequenceDiagramService
from services.usecase_service import UseCaseDiagramService

# --- SETUP PATHS ---
sys.path.insert(0, str(Path(__file__).parent))

# --- IMPORTS ---
from services.architecture_service import (
    ArchitectureVisitor,
    DeterministicPlantUMLConverter,
    FastTypeEnricher
)
from services.project_service import ProjectAnalyzer
from services.refactoring_service import RefactoringAdvisor
from core.llm_factory import create_gemini_llm, create_openai_llm, create_sambanova_llm, create_nebius_llm

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
PLANTUML_SERVER_URL = 'http://www.plantuml.com/plantuml/img/'
plantuml_client = PlantUML(url=PLANTUML_SERVER_URL)

# --- SINGLETON LLM CLIENT ---
class LLMClientSingleton:
    """Singleton pattern for LLM client to avoid repeated connections"""
    _instance = None
    _llm_client = None
    _current_provider = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self, preferred_provider: str = "openai", temperature: float = 0.0):
        """Get or create LLM client with provider fallback"""
        if self._llm_client is not None and self._current_provider == preferred_provider:
            return self._llm_client
        
        # Define provider strategies with fallbacks
        strategies = {
            "openai": [create_openai_llm, create_sambanova_llm, create_nebius_llm],
            "sambanova": [create_sambanova_llm, create_openai_llm, create_nebius_llm],
            "nebius": [create_nebius_llm, create_openai_llm, create_sambanova_llm], 
            "gemini": [create_gemini_llm, create_sambanova_llm, create_nebius_llm], 
        }
        
        factories = strategies.get(preferred_provider, strategies["nebius"])
        names = ["openai", "sambanova", "nebius", "gemini"]
        
        for factory, name in zip(factories, names):
            try:
                self._llm_client = factory(temperature=temperature)
                self._current_provider = name
                logger.info(f"✅ Connected to {name}")
                return self._llm_client
            except Exception as e:
                logger.warning(f"⚠️ {name} failed: {str(e)[:100]}")
        
        logger.error("❌ All LLM providers failed")
        return None

# Global singleton instance
_llm_singleton = LLMClientSingleton()

# --- HELPER FUNCTIONS ---
def render_plantuml(puml_text: str) -> tuple:
    """Render PlantUML text to image"""
    if not puml_text:
        return None, None
    
    try:
        image_bytes = plantuml_client.processes(puml_text)
        image = Image.open(io.BytesIO(image_bytes))
        return puml_text, image
    except Exception as e:
        logger.error(f"PlantUML render error: {e}")
        return f"{puml_text}\n\n⚠️ Render Error: {e}", None

def safe_cleanup(path):
    """Safely remove temporary directory"""
    try:
        if path and Path(path).exists():
            shutil.rmtree(path)
    except Exception as e:
        logger.warning(f"Cleanup failed for {path}: {e}")

def extract_file_list(zip_path):
    """Extract list of .py files from uploaded ZIP for dropdown"""
    if not zip_path:
        return gr.update(choices=[]), gr.update(choices=[])
    
    try:
        py_files = []
        test_files = ["None (Skip tests)"]  # Default option
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                filename = file_info.filename
                
                # Skip directories, non-Python files, and common junk
                if (file_info.is_dir() or 
                    not filename.endswith('.py') or
                    '__pycache__' in filename or
                    '/.venv/' in filename or
                    '/venv/' in filename):
                    continue
                
                py_files.append(filename)
                
                # Separate test files
                if 'test' in filename.lower():
                    test_files.append(filename)
        
        py_files.sort()
        
        if not py_files:
            return (
                gr.update(choices=["⚠️ No Python files found"], value=None),
                gr.update(choices=test_files, value="None (Skip tests)")
            )
        
        return (
            gr.update(choices=py_files, value=py_files[0]),
            gr.update(choices=test_files, value="None (Skip tests)")
        )
        
    except zipfile.BadZipFile:
        return (
            gr.update(choices=["❌ Invalid ZIP file"], value=None),
            gr.update(choices=["None (Skip tests)"], value="None (Skip tests)")
        )
    except Exception as e:
        logger.error(f"ZIP extraction error: {e}")
        return (
            gr.update(choices=[f"❌ Error: {str(e)[:50]}"], value=None),
            gr.update(choices=["None (Skip tests)"], value="None (Skip tests)")
        )

# --- TAB 1: SINGLE FILE ANALYSIS ---
def process_code_snippet(code_snippet: str, enrich_types: bool = False):
    """Analyze single Python code snippet and generate UML diagram"""
    if not code_snippet.strip():
        return "⚠️ Please enter some code.", None, gr.update(visible=True, value="⚠️ No Input")
    
    try:
        # Parse code with AST
        tree = ast.parse(code_snippet)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        if not visitor.structure:
            return "⚠️ No classes/functions found.", None, gr.update(visible=True, value="⚠️ No Structure")
        
        # Optional AI type enrichment
        if enrich_types:
            try:
                llm = _llm_singleton.get_client(preferred_provider="openai", temperature=0.0)
                if llm:
                    enricher = FastTypeEnricher(llm)
                    visitor.structure = enricher.enrich(code_snippet, visitor.structure)
                    logger.info("✓ Type enrichment complete")
            except Exception as e:
                logger.warning(f"Type enrichment failed: {e}")
        
        # Convert to PlantUML
        converter = DeterministicPlantUMLConverter()
        puml_text = converter.convert(visitor.structure)
        text, image = render_plantuml(puml_text)
        
        return text, image, gr.update(visible=True, value="✅ Analysis Complete!")
        
    except SyntaxError as se:
        return f"❌ Syntax Error: {se}", None, gr.update(visible=True, value="❌ Syntax Error")
    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        return f"❌ Error: {e}", None, gr.update(visible=True, value="❌ Failed")

# --- TAB 2: PROJECT MAP ---

def process_pattern_detection_zip(zip_path, enrich: bool = True, provider: str = "openai", progress=gr.Progress()):
    """Analyze entire project for design patterns"""
    if not zip_path:
        return "⚠️ Please upload a ZIP file first.", gr.update(visible=True, value="⚠️ No File"), gr.update(visible=False)
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        
        progress(0.2, desc="📦 Extracting ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        progress(0.4, desc="🔍 Scanning Python files...")
        
        # Collect all Python files
        all_code = []
        file_count = 0
        
        for file_path in Path(temp_dir).rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules"] for p in parts):
                continue
            try:
                code = file_path.read_text(encoding='utf-8', errors='replace')
                all_code.append(f"# === File: {file_path.name} ===\n{code}")
                file_count += 1
            except Exception:
                continue
        
        if not all_code:
            return "⚠️ No Python files found.", gr.update(visible=True, value="⚠️ No Files"), gr.update(visible=False)
        
        progress(0.6, desc=f"🏛️ Analyzing {file_count} files for patterns...")
        
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        service = PatternDetectionService(llm=llm)
        combined_code = "\n\n".join(all_code)
        result = service.analyze_code(combined_code, enrich=enrich)
        
        progress(0.9, desc="📝 Generating report...")
        report = service.format_report(result)
        
        progress(1.0, desc="✅ Complete!")
        
        status_msg = f"✅ Analyzed {file_count} files • Found {result['summary']['total_patterns']} patterns • {result['summary']['total_recommendations']} recommendations"
        
        return (
            report,
            gr.update(visible=True, value=status_msg),
            gr.update(visible=True)  # Show results section
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logging.error(f"Pattern detection error: {error_detail}")
        return (
            f"❌ Error: {e}\n\nDetails:\n{error_detail}",
            gr.update(visible=True, value=f"❌ Failed"),
            gr.update(visible=True)
        )
    finally:
        safe_cleanup(temp_dir)
        
def process_zip_upload(zip_path, progress=gr.Progress()):
    """Extract ZIP and analyze entire project structure"""
    if not zip_path:
        return "⚠️ Please upload a ZIP file.", None, gr.update(visible=True, value="⚠️ No File")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        
        progress(0.2, desc="📦 Extracting ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        progress(0.5, desc="🔍 Analyzing project...")
        analyzer = ProjectAnalyzer(Path(temp_dir))
        full_structure = analyzer.analyze()
        
        if not full_structure:
            return "⚠️ No Python code found.", None, gr.update(visible=True, value="⚠️ No Code")
        
        progress(0.8, desc="🎨 Generating diagram...")
        converter = DeterministicPlantUMLConverter()
        puml_text = converter.convert(full_structure)
        text, image = render_plantuml(puml_text)
        
        progress(1.0, desc="✅ Complete!")
        return text, image, gr.update(visible=True, value=f"✅ Found {len(full_structure)} components")
        
    except zipfile.BadZipFile:
        return "❌ Invalid ZIP file.", None, gr.update(visible=True, value="❌ Bad ZIP")
    except Exception as e:
        logger.error(f"Project analysis error: {e}")
        return f"❌ Error: {e}", None, gr.update(visible=True, value="❌ Failed")
    finally:
        safe_cleanup(temp_dir)


# -- TAB 3 : USE CASE DIAGRAM ---

def process_usecase_snippet(code_snippet: str, enrich: bool = True, provider: str = "sambanova"):
    """TAB 1B: Single File Use Case Diagram"""
    if not code_snippet.strip():
        return "⚠️ Please enter some code.", None, gr.update(visible=False)
    
    try:
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        service = UseCaseDiagramService(llm=llm)
        puml_text = service.generate(code_snippet, enrich=enrich, include_all_classes=True)
        text, image = render_plantuml(puml_text)
        
        return text, image, gr.update(visible=True, value="✅ Use Case Diagram Complete!")
        
    except Exception as e:
        return f"❌ Error: {e}", None, gr.update(visible=True, value=f"❌ Error")

def process_folder_usecase(folder_path: str, enrich: bool = True, provider: str = "sambanova", progress=gr.Progress()):
    """TAB 2B: Project Use Case Diagram - SINGLE COMBINED"""
    path_obj = Path(folder_path)
    
    if not path_obj.exists() or not path_obj.is_dir():
        return "❌ Invalid path.", None, gr.update(visible=True, value="❌ Invalid Path")
    
    try:
        progress(0.3, desc="Scanning Python files...")
        
        combined_code = []
        file_count = 0
        
        for file_path in path_obj.rglob("*.py"):
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
            return "⚠️ No Python files found.", None, gr.update(visible=True, value="⚠️ No Files")
        
        progress(0.6, desc=f"Analyzing {file_count} files...")
        
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        service = UseCaseDiagramService(llm=llm)
        full_code = "\n\n".join(combined_code)[:50000]
        puml_text = service.generate(full_code, enrich=enrich, include_all_classes=True)
        
        progress(0.9, desc="Rendering diagram...")
        text, image = render_plantuml(puml_text)
        
        progress(1.0, desc="Complete!")
        return text, image, gr.update(visible=True, value=f"✅ Analyzed {file_count} files")
        
    except Exception as e:
        return f"❌ Error: {e}", None, gr.update(visible=True, value=f"❌ Failed")

def process_folder_usecase_multi(folder_path: str, enrich: bool = True, provider: str = "sambanova", progress=gr.Progress()):
    """TAB 3: Project Use Case Diagrams - MULTIPLE BY MODULE"""
    path_obj = Path(folder_path)
    
    if not path_obj.exists() or not path_obj.is_dir():
        return "❌ Invalid path.", [], [], None, "", gr.update(visible=True, value="❌ Invalid Path")
    
    try:
        progress(0.2, desc="Scanning Python files...")
        
        file_contents = {}
        file_count = 0
        
        for file_path in path_obj.rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules"] for p in parts):
                continue
            try:
                code = file_path.read_text(encoding='utf-8', errors='replace')
                rel_path = file_path.relative_to(path_obj)
                file_contents[str(rel_path)] = code
                file_count += 1
            except Exception:
                continue
        
        if not file_contents:
            return "⚠️ No Python files found.", [], [], None, "", gr.update(visible=True, value="⚠️ No Files")
        
        progress(0.5, desc=f"Analyzing {file_count} files by module...")
        
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        service = UseCaseDiagramService(llm=llm)
        
        if hasattr(service, 'generate_modular'):
            diagrams_dict = service.generate_modular(file_contents, enrich=enrich)
        else:
            return "⚠️ Please update usecase_service.py with multi-module support", [], [], None, "", gr.update(visible=True, value="⚠️ Update Required")
        
        if "error" in diagrams_dict:
            return diagrams_dict["error"], [], [], None, "", gr.update(visible=True, value="❌ Failed")
        
        progress(0.8, desc="Rendering diagrams...")
        
        diagram_outputs = []
        
        for module_name, puml_text in diagrams_dict.items():
            if "error" in module_name.lower():
                continue
            text, image = render_plantuml(puml_text)
            
            if image:
                diagram_outputs.append({
                    "module": module_name,
                    "image": image,
                    "puml": puml_text
                })
        
        if not diagram_outputs:
            return "⚠️ No diagrams generated.", [], [], None, "", gr.update(visible=True, value="⚠️ No Diagrams")
        
        progress(1.0, desc="Complete!")
        
        summary = f"✅ Generated {len(diagram_outputs)} Use Case diagrams:\n\n"
        summary += "\n".join([f"📊 {d['module']}" for d in diagram_outputs])
        
        module_names = [d["module"] for d in diagram_outputs]
        
        first_img = diagram_outputs[0]["image"] if diagram_outputs else None
        first_puml = diagram_outputs[0]["puml"] if diagram_outputs else ""
        
        return (
            summary,
            diagram_outputs,
            gr.update(choices=module_names, value=module_names[0] if module_names else None),
            first_img,
            first_puml,
            gr.update(visible=True, value=f"✅ {len(diagram_outputs)} Modules")
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logging.error(f"Multi-diagram error: {error_detail}")
        return f"❌ Error: {e}\n\nDetails:\n{error_detail}", [], [], None, "", gr.update(visible=True, value=f"❌ Failed")

def process_folder_usecase_multi_zip(zip_path, enrich: bool = True, provider: str = "sambanova", progress=gr.Progress()):
    """TAB 3: Multi-Module Use Cases from ZIP file"""
    
    # ✅ FIX: Check if zip_path is provided and is a valid file
    if not zip_path:
        return "⚠️ Please upload a ZIP file.", [], [], None, "", gr.update(visible=True, value="⚠️ No File")
    
    # ✅ FIX: Check if the file exists and is a valid ZIP
    zip_file = Path(zip_path)
    if not zip_file.exists():
        return "❌ File not found.", [], [], None, "", gr.update(visible=True, value="❌ File Not Found")
    
    if not zip_file.suffix.lower() == '.zip':
        return "❌ Please upload a ZIP file.", [], [], None, "", gr.update(visible=True, value="❌ Invalid File Type")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        logging.info(f"📁 Created temp directory: {temp_dir}")
        
        progress(0.1, desc="📦 Extracting ZIP...")
        
        # ✅ FIX: Add error handling for ZIP extraction
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            logging.info(f"✅ Extracted ZIP to: {temp_dir}")
        except zipfile.BadZipFile:
            return "❌ Invalid or corrupted ZIP file.", [], [], None, "", gr.update(visible=True, value="❌ Bad ZIP")
        
        progress(0.2, desc="🔍 Scanning Python files...")
        
        file_contents = {}
        file_count = 0
        
        # ✅ FIX: Use temp_dir (not zip_path) to scan files
        for file_path in Path(temp_dir).rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules", "__MACOSX"] for p in parts):
                continue
            try:
                code = file_path.read_text(encoding='utf-8', errors='replace')
                rel_path = file_path.relative_to(temp_dir)
                file_contents[str(rel_path)] = code
                file_count += 1
                logging.info(f"📄 Found file: {rel_path}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to read {file_path}: {e}")
                continue
        
        if not file_contents:
            return f"⚠️ No Python files found in ZIP. Extracted to: {temp_dir}", [], [], None, "", gr.update(visible=True, value="⚠️ No Files")
        
        logging.info(f"✅ Found {file_count} Python files")
        progress(0.5, desc=f"🎯 Analyzing {file_count} files by module...")
        
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        service = UseCaseDiagramService(llm=llm)
        
        if not hasattr(service, 'generate_modular'):
            return "⚠️ Please update usecase_service.py with multi-module support", [], [], None, "", gr.update(visible=True, value="⚠️ Update Required")
        
        diagrams_dict = service.generate_modular(file_contents, enrich=enrich)
        
        if "error" in diagrams_dict:
            return diagrams_dict["error"], [], [], None, "", gr.update(visible=True, value="❌ Failed")
        
        progress(0.8, desc="🎨 Rendering diagrams...")
        
        diagram_outputs = []
        
        for module_name, puml_text in diagrams_dict.items():
            if "error" in module_name.lower():
                continue
            text, image = render_plantuml(puml_text)
            
            if image:
                diagram_outputs.append({
                    "module": module_name,
                    "image": image,
                    "puml": puml_text
                })
        
        if not diagram_outputs:
            return "⚠️ No diagrams generated. Check if your code has classes/methods.", [], [], None, "", gr.update(visible=True, value="⚠️ No Diagrams")
        
        progress(1.0, desc="✅ Complete!")
        
        summary = f"✅ Generated {len(diagram_outputs)} Use Case diagrams from {file_count} files:\n\n"
        summary += "\n".join([f"📊 {d['module']}" for d in diagram_outputs])
        
        module_names = [d["module"] for d in diagram_outputs]
        first_img = diagram_outputs[0]["image"] if diagram_outputs else None
        first_puml = diagram_outputs[0]["puml"] if diagram_outputs else ""
        
        return (
            summary,
            diagram_outputs,
            gr.update(choices=module_names, value=module_names[0] if module_names else None),
            first_img,
            first_puml,
            gr.update(visible=True, value=f"✅ {len(diagram_outputs)} Modules")
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logging.error(f"❌ Multi-diagram error: {error_detail}")
        return f"❌ Error: {e}\n\nDetails:\n{error_detail}", [], [], None, "", gr.update(visible=True, value=f"❌ Failed")
    finally:
        safe_cleanup(temp_dir)



# --- TAB 4: sequence diagrams ---
def process_folder_sequence_multi_zip(zip_path, enrich: bool = False, provider: str = "sambanova", progress=gr.Progress()):
    """TAB 4: Multi-Module Sequences from ZIP file"""
    
    # ✅ FIX: Check if zip_path is provided and is a valid file
    if not zip_path:
        return "⚠️ Please upload a ZIP file.", [], [], None, "", gr.update(visible=True, value="⚠️ No File")
    
    # ✅ FIX: Check if the file exists and is a valid ZIP
    zip_file = Path(zip_path)
    if not zip_file.exists():
        return "❌ File not found.", [], [], None, "", gr.update(visible=True, value="❌ File Not Found")
    
    if not zip_file.suffix.lower() == '.zip':
        return "❌ Please upload a ZIP file.", [], [], None, "", gr.update(visible=True, value="❌ Invalid File Type")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        logging.info(f"📁 Created temp directory: {temp_dir}")
        
        progress(0.1, desc="📦 Extracting ZIP...")
        
        # ✅ FIX: Add error handling for ZIP extraction
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            logging.info(f"✅ Extracted ZIP to: {temp_dir}")
        except zipfile.BadZipFile:
            return "❌ Invalid or corrupted ZIP file.", [], [], None, "", gr.update(visible=True, value="❌ Bad ZIP")
        
        progress(0.2, desc="🔍 Scanning Python files...")
        
        file_contents = {}
        file_count = 0
        
        # ✅ FIX: Use temp_dir (not zip_path) to scan files
        for file_path in Path(temp_dir).rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules", "__MACOSX"] for p in parts):
                continue
            try:
                code = file_path.read_text(encoding='utf-8', errors='replace')
                rel_path = file_path.relative_to(temp_dir)
                file_contents[str(rel_path)] = code
                file_count += 1
                logging.info(f"📄 Found file: {rel_path}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to read {file_path}: {e}")
                continue
        
        if not file_contents:
            return f"⚠️ No Python files found in ZIP. Extracted to: {temp_dir}", [], [], None, "", gr.update(visible=True, value="⚠️ No Files")
        
        logging.info(f"✅ Found {file_count} Python files")
        progress(0.5, desc=f"🎬 Analyzing {file_count} files by module...")
        
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        service = SequenceDiagramService(llm=llm)
        
        if not hasattr(service, 'generate_modular'):
            return "⚠️ Please update sequence_service.py with multi-module support", [], [], None, "", gr.update(visible=True, value="⚠️ Update Required")
        
        diagrams_dict = service.generate_modular(file_contents, enrich=enrich)
        
        if "error" in diagrams_dict:
            return diagrams_dict["error"], [], [], None, "", gr.update(visible=True, value="❌ Failed")
        
        progress(0.8, desc="🎨 Rendering diagrams...")
        
        diagram_outputs = []
        
        for module_name, puml_text in diagrams_dict.items():
            if "error" in module_name.lower():
                continue
            
            text, image = render_plantuml(puml_text)
            
            if image:
                diagram_outputs.append({
                    "module": module_name,
                    "image": image,
                    "puml": puml_text
                })
        
        if not diagram_outputs:
            return "⚠️ No diagrams generated. Check if your code has method calls.", [], [], None, "", gr.update(visible=True, value="⚠️ No Diagrams")
        
        progress(1.0, desc="✅ Complete!")
        
        summary = f"✅ Generated {len(diagram_outputs)} Sequence diagrams from {file_count} files:\n\n"
        summary += "\n".join([f"🎬 {d['module']}" for d in diagram_outputs])
        
        module_names = [d["module"] for d in diagram_outputs]
        first_img = diagram_outputs[0]["image"] if diagram_outputs else None
        first_puml = diagram_outputs[0]["puml"] if diagram_outputs else ""
        
        return (
            summary,
            diagram_outputs,
            gr.update(choices=module_names, value=module_names[0] if module_names else None),
            first_img,
            first_puml,
            gr.update(visible=True, value=f"✅ {len(diagram_outputs)} Modules")
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logging.error(f"❌ Multi-sequence error: {error_detail}")
        return f"❌ Error: {e}\n\nDetails:\n{error_detail}", [], [], None, "", gr.update(visible=True, value=f"❌ Failed")
    finally:
        safe_cleanup(temp_dir)

def process_sequence_snippet(code_snippet: str, entry_method: str = None, enrich: bool = True, provider: str = "sambanova"):
    """TAB 1C: Single File Sequence Diagram"""
    if not code_snippet.strip():
        return "⚠️ Please enter some code.", None, gr.update(visible=False), ""
    
    try:
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        service = SequenceDiagramService(llm=llm)
        puml_text = service.generate(code_snippet, entry_method=entry_method, enrich=enrich)
        text, image = render_plantuml(puml_text)
        
        tree = ast.parse(code_snippet)
        visitor = CallGraphVisitor()
        visitor.visit(tree)
        
        entry_points = [m for m in visitor.call_sequences.keys() if not m.startswith('_')]
        entry_info = f"Available entry points: {', '.join(entry_points[:10])}"
        
        return text, image, gr.update(visible=True, value="✅ Sequence Diagram Complete!"), entry_info
        
    except Exception as e:
        return f"❌ Error: {e}", None, gr.update(visible=True, value=f"❌ Error"), ""

def process_folder_sequence(folder_path: str, entry_method: str = None, enrich: bool = False, provider: str = "sambanova", progress=gr.Progress()):
    """TAB 2D: Project Sequence Diagram"""
    path_obj = Path(folder_path)
    
    if not path_obj.exists() or not path_obj.is_dir():
        return "❌ Invalid path.", None, gr.update(visible=True, value="❌ Invalid Path"), ""
    
    try:
        progress(0.3, desc="Scanning Python files...")
        
        file_contents = {}
        for file_path in path_obj.rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", ".git"] for p in parts):
                continue
            try:
                file_contents[file_path.name] = file_path.read_text(encoding='utf-8', errors='replace')
            except Exception:
                continue
        
        if not file_contents:
            return "⚠️ No Python files found.", None, gr.update(visible=True, value="⚠️ No Files"), ""
        
        progress(0.6, desc=f"Analyzing {len(file_contents)} files...")
        
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        analyzer = ProjectSequenceAnalyzer(llm=llm)
        analyzer.analyze_files(file_contents)
        
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
            methods = list(analyzer.global_visitor.call_sequences.keys())[:10]
            return f"⚠️ No entry method. Available: {methods}", None, gr.update(visible=True, value="⚠️ Specify Entry"), ""
        
        progress(0.8, desc="Generating sequence diagram...")
        puml_text = analyzer.generate_diagram(entry_method, enrich=enrich)
        text, image = render_plantuml(puml_text)
        
        entry_points = [m for m in analyzer.global_visitor.call_sequences.keys() if not m.startswith('_')]
        entry_info = f"Top methods: {', '.join(entry_points[:15])}"
        
        progress(1.0, desc="Complete!")
        return text, image, gr.update(visible=True, value=f"✅ Entry: {entry_method}"), entry_info
        
    except Exception as e:
        return f"❌ Error: {e}", None, gr.update(visible=True, value=f"❌ Failed"), ""

def process_folder_sequence_multi(folder_path: str, enrich: bool = False, provider: str = "sambanova", progress=gr.Progress()):
    """TAB 6: Generate MULTIPLE sequence diagrams, one per module"""
    path_obj = Path(folder_path)
    
    if not path_obj.exists() or not path_obj.is_dir():
        return "❌ Invalid path.", [], [], None, "", gr.update(visible=True, value="❌ Invalid Path")
    
    try:
        progress(0.2, desc="Scanning Python files...")
        
        file_contents = {}
        file_count = 0
        
        for file_path in path_obj.rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules"] for p in parts):
                continue
            try:
                code = file_path.read_text(encoding='utf-8', errors='replace')
                rel_path = file_path.relative_to(path_obj)
                file_contents[str(rel_path)] = code
                file_count += 1
            except Exception:
                continue
        
        if not file_contents:
            return "⚠️ No Python files found.", [], [], None, "", gr.update(visible=True, value="⚠️ No Files")
        
        progress(0.5, desc=f"Analyzing {file_count} files by module...")
        
        llm = None
        if enrich:
            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
        
        service = SequenceDiagramService(llm=llm)
        
        if not hasattr(service, 'generate_modular'):
            return "⚠️ Please update sequence_service.py with multi-module support", [], [], None, "", gr.update(visible=True, value="⚠️ Update Required")
        
        diagrams_dict = service.generate_modular(file_contents, enrich=enrich)
        
        if "error" in diagrams_dict:
            return diagrams_dict["error"], [], [], None, "", gr.update(visible=True, value="❌ Failed")
        
        progress(0.8, desc="Rendering diagrams...")
        
        diagram_outputs = []
        
        for module_name, puml_text in diagrams_dict.items():
            if "error" in module_name.lower():
                continue
            
            text, image = render_plantuml(puml_text)
            
            if image:
                diagram_outputs.append({
                    "module": module_name,
                    "image": image,
                    "puml": puml_text
                })
        
        if not diagram_outputs:
            return "⚠️ No diagrams generated.", [], [], None, "", gr.update(visible=True, value="⚠️ No Diagrams")
        
        progress(1.0, desc="Complete!")
        
        summary = f"✅ Generated {len(diagram_outputs)} Sequence diagrams:\n\n"
        summary += "\n".join([f"🎬 {d['module']}" for d in diagram_outputs])
        
        module_names = [d["module"] for d in diagram_outputs]
        
        first_img = diagram_outputs[0]["image"] if diagram_outputs else None
        first_puml = diagram_outputs[0]["puml"] if diagram_outputs else ""
        
        return (
            summary,
            diagram_outputs,
            gr.update(choices=module_names, value=module_names[0] if module_names else None),
            first_img,
            first_puml,
            gr.update(visible=True, value=f"✅ {len(diagram_outputs)} Modules")
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logging.error(f"Multi-sequence error: {error_detail}")
        return f"❌ Error: {e}\n\nDetails:\n{error_detail}", [], [], None, "", gr.update(visible=True, value=f"❌ Failed")

# --- TAB 5: AI PROPOSAL ---
def process_proposal_zip(zip_path, progress=gr.Progress()):
    """AI-powered architecture refactoring proposal"""
    if not zip_path:
        return "⚠️ Please upload a ZIP file.", None, None, gr.update(visible=True, value="⚠️ No File")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        
        progress(0.2, desc="📦 Extracting ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        progress(0.5, desc="🧠 AI analyzing architecture...")
        analyzer = ProjectAnalyzer(Path(temp_dir))
        structure = analyzer.analyze()
        
        if not structure:
            return "⚠️ No code found.", None, None, gr.update(visible=True, value="⚠️ No Code")
        
        advisor = RefactoringAdvisor()
        proposal = advisor.propose_improvement(structure)
        
        if "error" in proposal:
            return f"❌ AI Error: {proposal['error']}", None, None, gr.update(visible=True, value="❌ AI Failed")
        
        progress(0.8, desc="🎨 Generating proposed UML...")
        puml_code = proposal.get("proposed_uml", "")
        _, image = render_plantuml(puml_code)
        
        progress(1.0, desc="✅ Complete!")
        return (
            json.dumps(proposal, indent=2),
            puml_code,
            image,
            gr.update(visible=True, value="✅ Proposal Generated")
        )
        
    except Exception as e:
        logger.error(f"Proposal generation error: {e}")
        return f"❌ Error: {e}", None, None, gr.update(visible=True, value="❌ Failed")
    finally:
        safe_cleanup(temp_dir)

# --- TAB 6: MODAL REFACTORING ---
def run_modal_refactoring_zip(zip_path, file_path, instruction, test_path=None, progress=gr.Progress()):
    """Execute refactoring in Modal cloud sandbox"""
    # Validation
    if not zip_path:
        return "⚠️ Please upload a ZIP file.", gr.update(visible=False)
    if not file_path or file_path.startswith("❌") or file_path.startswith("⚠️"):
        return "⚠️ Please select a valid Python file.", gr.update(visible=False)
    if not instruction or not instruction.strip():
        return "⚠️ Please provide refactoring instructions.", gr.update(visible=False)
    
    # Handle test file selection
    if test_path == "None (Skip tests)":
        test_path = None
    
    temp_extract = None
    try:
        temp_extract = tempfile.mkdtemp()
        
        progress(0.2, desc="📦 Extracting project...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        
        progress(0.4, desc="📄 Locating file...")
        
        # FIX: Search for file in extracted directory
        target_file = None
        for root, dirs, files in os.walk(temp_extract):
            if file_path in root or file_path.endswith(os.path.basename(root)):
                continue
            potential_path = Path(root) / os.path.basename(file_path)
            if potential_path.exists():
                target_file = potential_path
                break
        
        # If not found, try direct path
        if not target_file:
            target_file = Path(temp_extract) / file_path
        
        if not target_file or not target_file.exists():
            # List what we actually have for debugging
            all_py_files = []
            for root, dirs, files in os.walk(temp_extract):
                for f in files:
                    if f.endswith('.py'):
                        rel_path = Path(root).relative_to(temp_extract) / f
                        all_py_files.append(str(rel_path))
            
            return f"❌ File not found: {file_path}\n\nAvailable files:\n" + "\n".join(all_py_files[:10]), gr.update(visible=False)
        
        progress(0.5, desc="📖 Reading file...")
        original_code = target_file.read_text(encoding='utf-8')
        
        # Read test file
        test_code = None
        if test_path:
            test_file = None
            for root, dirs, files in os.walk(temp_extract):
                potential_path = Path(root) / os.path.basename(test_path)
                if potential_path.exists():
                    test_file = potential_path
                    break
            
            if test_file and test_file.exists():
                test_code = test_file.read_text(encoding='utf-8')
        
        progress(0.7, desc="☁️ Sending to Modal...")
        
        # Call Modal
        try:
            from server import apply_refactoring_safely
            
            # Use just the filename for Modal
            file_name = os.path.basename(file_path)
            result = apply_refactoring_safely(file_name, instruction, test_path)
            
            progress(1.0, desc="✅ Complete!")
            return result, gr.update(visible=False)
            
        except ImportError:
            return "⚠️ Modal not configured.", gr.update(visible=False)
        except Exception as modal_error:
            logger.error(f"Modal error: {modal_error}")
            return f"❌ Modal failed: {str(modal_error)}", gr.update(visible=False)
            
    except Exception as e:
        logger.error(f"Refactoring error: {e}")
        return f"❌ Error: {str(e)}", gr.update(visible=False)
    finally:
        safe_cleanup(temp_extract)
# --- CUSTOM CSS ---
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1400px !important;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.main-header p {
    font-size: 1.1rem;
    opacity: 0.9;
}

.banner {
    padding: 1rem;
    border-radius: 8px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 1rem;
}

.info-card {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.diagram-container {
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 1rem;
    background: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.sponsors {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.sponsor-badge {
    background: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
"""

# --- GRADIO INTERFACE ---
with gr.Blocks(
    title="ArchitectAI - Autonomous Cloud Refactoring",
    fill_height=True
) as demo:
    
    # HEADER
    gr.HTML("""
        <div class="main-header">
            <div style="margin-bottom: 1.5rem;">
                <h1>$ ./architectai --run</h1>
                <p class="subtitle">→ architecture intelligence platform</p>
            </div>

            <div style="margin: 1.5rem 0;">
                <p class="tagline">» multi-module-uml | pattern-detection | evolution-tracking | ai-refactoring</p>
            </div>

            <div class="sponsors">
                <span class="sponsor-badge">◆ anthropic-mcp</span>
                <span class="sponsor-badge">◇ modal</span>
                <span class="sponsor-badge">◈ openai</span>
                <span class="sponsor-badge">◉ sambanova</span>
                <span class="sponsor-badge">◎ nebius</span>
            </div>
        </div>
    """)
    
    # TABS
    with gr.Tabs():
        
        # TAB 1: Single File
        with gr.Tab("📄 Single File Analysis" , id = 0):
            gr.HTML('<div class="info-card"><strong>💡 Quick Analysis:</strong> Paste Python code to generate instant UML class diagram with optional AI-powered type enrichment.</div>')

            with gr.Row():
                # LEFT COLUMN - Inputs
                with gr.Column(scale=1):
                    code_input = gr.Code(
                        language="python",
                        label="Python Code",
                        lines=20,
                        elem_classes=["code-input"]
                    )
                    enrich_checkbox = gr.Checkbox(
                        label="✨ AI Type Enrichment",
                        value=False,
                        info="Use AI to infer missing type hints"
                    )
                    analyze_btn = gr.Button(
                        "🚀 Analyze Code",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"]
                    )

                # RIGHT COLUMN - Results
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["output-card"]):
                        status_banner_1 = gr.Markdown(visible=False, elem_classes=["banner"])
                        img_output_1 = gr.Image(
                            label="📊 Class Diagram",
                            type="pil",
                            elem_classes=["diagram-container"]
                        )

                    with gr.Accordion("📝 PlantUML Source", open=False):
                        text_output_1 = gr.Code(language="markdown", lines=10)

            analyze_btn.click(
                fn=process_code_snippet,
                inputs=[code_input, enrich_checkbox],
                outputs=[text_output_1, img_output_1, status_banner_1]
            )
        

        # TAB 2: Project Map
        with gr.Tab("📂 Project Map", id=1):
            gr.HTML('<div class="info-card"><strong>🗺️ Full Project Analysis:</strong> Upload a ZIP file to visualize all classes, relationships, and design patterns in your Python project. Works best with 5-50 files.</div>')

            # Store the project structure for pattern detection
            stored_project_structure = gr.State(None)
            stored_project_code = gr.State(None)

            with gr.Row():
                # LEFT COLUMN - Upload
                with gr.Column(scale=1):
                    project_zip = gr.File(
                        label="📦 Upload Project (ZIP)",
                        file_types=[".zip"],
                        type="filepath",
                        elem_classes=["file-upload"]
                    )
                    scan_btn = gr.Button(
                        "🔍 Scan Project",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"]
                    )

                # RIGHT COLUMN - Results
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["output-card"]):
                        status_banner_2 = gr.Markdown(visible=False, elem_classes=["banner"])
                        img_output_2 = gr.Image(
                            label="🗺️ Architecture Map",
                            type="pil",
                            elem_classes=["diagram-container"]
                        )

                    with gr.Accordion("📝 PlantUML Source", open=False):
                        text_output_2 = gr.Code(language="markdown", lines=10)
            
            # Pattern Detection Section (appears after diagram generation)
            with gr.Row(visible=False) as pattern_section:
                with gr.Column(scale=1):
                    gr.HTML('<div class="info-card"><strong>🏛️ Design Pattern Analysis:</strong> Detect existing patterns and get AI-powered recommendations for architectural improvements.</div>')

                    with gr.Row():
                        pattern_enrich_toggle = gr.Checkbox(
                            label="✨ AI Enrichment",
                            value=True,
                            info="Generate detailed justifications (slower)"
                        )
                        pattern_provider_choice = gr.Dropdown(
                            choices=["openai", "sambanova", "nebius"],
                            value="openai",
                            label="LLM Provider",
                            scale=1
                        )

                    detect_patterns_btn = gr.Button(
                        "🔍 Detect Patterns & Recommendations",
                        variant="secondary",
                        size="lg"
                    )

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["output-card"]):
                        pattern_status = gr.Markdown(visible=False, elem_classes=["banner"])
            
            # Pattern Report Output
            with gr.Row(visible=False) as pattern_results_section:
                with gr.Column():
                    with gr.Group(elem_classes=["output-card"]):
                        pattern_report_output = gr.Markdown(
                            label="📊 Pattern Analysis Report",
                            value="*Waiting for analysis...*"
                        )

            # Pattern UML Visualizations
            with gr.Row(visible=False) as pattern_uml_section:
                gr.HTML('<div class="info-card"><strong>💡 Before & After:</strong> Visual comparison of current design vs. recommended pattern implementation.</div>')

            with gr.Row(visible=False) as pattern_selector_section:
                recommendation_dropdown = gr.Dropdown(
                    label="📋 Select Recommendation to Visualize",
                    choices=[],
                    interactive=True
                )

            with gr.Row(visible=False) as pattern_comparison_section:
                with gr.Column(scale=1):
                    gr.Markdown("#### ⚠️ Before (Current Structure)")
                    with gr.Group(elem_classes=["output-card"]):
                        pattern_before_img = gr.Image(
                            label="Current Design",
                            type="pil",
                            elem_classes=["diagram-container"]
                        )
                    with gr.Accordion("📝 PlantUML Code", open=False):
                        pattern_before_uml = gr.Code(language="markdown", lines=8)

                with gr.Column(scale=1):
                    gr.Markdown("#### ✅ After (Recommended Pattern)")
                    with gr.Group(elem_classes=["output-card"]):
                        pattern_after_img = gr.Image(
                            label="Improved Design",
                            type="pil",
                            elem_classes=["diagram-container"]
                        )
                    with gr.Accordion("📝 PlantUML Code", open=False):
                        pattern_after_uml = gr.Code(language="markdown", lines=8)

            def process_pattern_detection_from_structure(structure, code, enrich: bool = True, provider: str = "openai", progress=gr.Progress()):
                """
                Analyze patterns using already-parsed structure and code.
                Now with UML diagram generation for recommendations!
                """
                
                if not structure or not code:
                    logging.warning("⚠️ Pattern detection called without structure or code")
                    return (
                        "⚠️ Please generate the class diagram first.",
                        gr.update(visible=True, value="⚠️ No Data"),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(choices=[]),
                        None, None, "", ""
                    )
                
                try:
                    # Log what we received
                    logging.info(f"🔍 Pattern detection using stored project: {len(structure)} components, {len(code)} chars of code")
                    
                    progress(0.2, desc="🏛️ Analyzing patterns...")
                    
                    # Get LLM if enrichment enabled
                    llm = None
                    if enrich:
                        try:
                            llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
                            progress(0.4, desc="🤖 AI analyzing patterns...")
                        except Exception as e:
                            logger.warning(f"LLM initialization failed: {e}, proceeding without enrichment")
                    
                    # Run pattern detection
                    service = PatternDetectionService(llm=llm)
                    
                    progress(0.6, desc="🔍 Detecting patterns...")
                    result = service.analyze_code(code[:100000], enrich=enrich)
                    
                    progress(0.8, desc="📝 Generating report...")
                    report = service.format_report(result)
                    
                    # Generate UML for recommendations
                    recommendation_choices = []
                    first_before_uml = ""
                    first_after_uml = ""
                    first_before_img = None
                    first_after_img = None
                    
                    if result['recommendations']:
                        progress(0.9, desc="🎨 Generating UML diagrams...")
                        
                        # Store UML diagrams for all recommendations
                        for i, rec_dict in enumerate(result['recommendations']):
                            rec = PatternRecommendation(**rec_dict)
                            recommendation_choices.append(f"{i+1}. {rec.pattern} - {rec.location}")
                            
                            # Generate UML for first recommendation
                            if i == 0:
                                recommender = service.recommender
                                before_uml, after_uml = recommender.generate_recommendation_uml(rec, structure, code)
                                
                                first_before_uml = before_uml
                                first_after_uml = after_uml
                                
                                # Render UML to images
                                _, first_before_img = render_plantuml(before_uml)
                                _, first_after_img = render_plantuml(after_uml)
                    
                    progress(1.0, desc="✅ Complete!")
                    
                    # Create status message
                    patterns_count = result['summary']['total_patterns']
                    recs_count = result['summary']['total_recommendations']
                    files_analyzed = len(set(item.get('source_file', 'unknown') for item in structure))
                    
                    status_msg = f"✅ Analyzed {files_analyzed} files • Found {patterns_count} pattern(s) • {recs_count} recommendation(s)"
                    
                    show_uml = recs_count > 0
                    
                    return (
                        report,
                        gr.update(visible=True, value=status_msg),
                        gr.update(visible=True),  # Show report
                        gr.update(visible=show_uml),  # Show UML section if recommendations exist
                        gr.update(visible=show_uml),  # Show selector if recommendations exist
                        gr.update(visible=show_uml),  # Show comparison section if recommendations exist
                        gr.update(choices=recommendation_choices, value=recommendation_choices[0] if recommendation_choices else None),
                        first_before_img,
                        first_after_img,
                        first_before_uml,
                        first_after_uml
                    )
                    
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    logger.error(f"Pattern detection error: {error_detail}")
                    
                    return (
                        f"❌ Error during pattern detection:\n\n{str(e)}\n\n**Details:**\n```\n{error_detail[:500]}\n```",
                        gr.update(visible=True, value="❌ Analysis Failed"),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(choices=[]),
                        None, None, "", ""
                    )
                
            def update_recommendation_visualization(selected_rec, structure, code, enrich, provider):
                """Update UML diagrams when user selects different recommendation"""
                if not selected_rec or not structure:
                    return None, None, "", ""
                
                try:
                    # Parse selection (format: "1. Strategy - PaymentProcessor")
                    rec_index = int(selected_rec.split(".")[0]) - 1
                    
                    # Re-run analysis to get recommendations
                    llm = None
                    if enrich:
                        llm = _llm_singleton.get_client(preferred_provider=provider, temperature=0.0)
                    
                    service = PatternDetectionService(llm=llm)
                    result = service.analyze_code(code[:100000], enrich=False)  # Don't re-enrich
                    
                    if rec_index < len(result['recommendations']):
                        rec_dict = result['recommendations'][rec_index]
                        rec = PatternRecommendation(**rec_dict)
                        
                        # Generate UML
                        recommender = service.recommender
                        before_uml, after_uml = recommender.generate_recommendation_uml(rec, structure, code)
                        
                        # Render to images
                        _, before_img = render_plantuml(before_uml)
                        _, after_img = render_plantuml(after_uml)
                        
                        return before_img, after_img, before_uml, after_uml
                
                except Exception as e:
                    logger.error(f"Visualization update error: {e}")
                
                return None, None, "", ""

            # Event handlers
            def process_zip_and_store(zip_path, progress=gr.Progress()):
                """Process ZIP, generate diagram, and store data for pattern detection"""
                if not zip_path:
                    return (
                        "⚠️ Please upload a ZIP file.", 
                        None, 
                        gr.update(visible=True, value="⚠️ No File"),
                        gr.update(visible=False),
                        None,
                        None
                    )
                
                temp_dir = None
                try:
                    temp_dir = tempfile.mkdtemp()
                    
                    progress(0.2, desc="📦 Extracting ZIP...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    progress(0.5, desc="🔍 Analyzing project...")
                    analyzer = ProjectAnalyzer(Path(temp_dir))
                    full_structure = analyzer.analyze()
                    
                    if not full_structure:
                        return (
                            "⚠️ No Python code found.", 
                            None, 
                            gr.update(visible=True, value="⚠️ No Code"),
                            gr.update(visible=False),
                            None,
                            None
                        )
                    
                    # Collect raw code for pattern detection
                    all_code = []
                    file_count = 0
                    for file_path in Path(temp_dir).rglob("*.py"):
                        parts = file_path.parts
                        if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules"] for p in parts):
                            continue
                        try:
                            code = file_path.read_text(encoding='utf-8', errors='replace')
                            # Add file header for better pattern detection context
                            rel_path = file_path.relative_to(temp_dir)
                            all_code.append(f"# === File: {rel_path} ===\n{code}")
                            file_count += 1
                        except Exception:
                            continue
                    
                    combined_code = "\n\n".join(all_code)
                    
                    # Log info for debugging
                    logging.info(f"📊 Stored {file_count} Python files ({len(combined_code)} chars) for pattern detection")
                    
                    progress(0.8, desc="🎨 Generating diagram...")
                    converter = DeterministicPlantUMLConverter()
                    puml_text = converter.convert(full_structure)
                    text, image = render_plantuml(puml_text)
                    
                    progress(1.0, desc="✅ Complete!")
                    
                    return (
                        text, 
                        image, 
                        gr.update(visible=True, value=f"✅ Found {len(full_structure)} components • {file_count} files ready for pattern analysis"),
                        gr.update(visible=True),  # Show pattern detection section
                        full_structure,  # Store structure for pattern detection
                        combined_code   # Store code for pattern detection
                    )
                    
                except zipfile.BadZipFile:
                    return (
                        "❌ Invalid ZIP file.", 
                        None, 
                        gr.update(visible=True, value="❌ Bad ZIP"),
                        gr.update(visible=False),
                        None,
                        None
                    )
                except Exception as e:
                    logger.error(f"Project analysis error: {e}")
                    return (
                        f"❌ Error: {e}", 
                        None, 
                        gr.update(visible=True, value="❌ Failed"),
                        gr.update(visible=False),
                        None,
                        None
                    )
                finally:
                    safe_cleanup(temp_dir)
            
            scan_btn.click(
                fn=process_zip_and_store,
                inputs=project_zip,
                outputs=[
                    text_output_2, 
                    img_output_2, 
                    status_banner_2,
                    pattern_section,
                    stored_project_structure,  # Store for pattern detection
                    stored_project_code        # Store for pattern detection
                ]
            )
            
            detect_patterns_btn.click(
                fn=process_pattern_detection_from_structure,
                inputs=[stored_project_structure, stored_project_code, pattern_enrich_toggle, pattern_provider_choice],
                outputs=[
                    pattern_report_output,
                    pattern_status,
                    pattern_results_section,
                    pattern_uml_section,
                    pattern_selector_section,
                    pattern_comparison_section,
                    recommendation_dropdown,
                    pattern_before_img,
                    pattern_after_img,
                    pattern_before_uml,
                    pattern_after_uml
                ]
            )
            recommendation_dropdown.change(
                fn=update_recommendation_visualization,
                inputs=[recommendation_dropdown, stored_project_structure, stored_project_code, pattern_enrich_toggle, pattern_provider_choice],
                outputs=[pattern_before_img, pattern_after_img, pattern_before_uml, pattern_after_uml]
            )

        # TAB 3: MULTI-MODULE USE CASES
        
        with gr.Tab("📊 Multi-Module Use Cases", id=2):
            gr.Markdown("### Multiple Use Case Diagrams by Module\nGenerates **separate diagrams for each module** - much clearer than one massive diagram!")
            
            gr.HTML('<div class="info-card"><strong>🎯 Smart Organization:</strong> Each service/module gets its own diagram. Perfect for large projects with multiple components.</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    multi_zip_input = gr.File(
                            label="📦 Upload Project ZIP",
                            file_types=[".zip"],
                            type="filepath"
                        )
                    
                    with gr.Row():
                        multi_provider = gr.Dropdown(choices=["sambanova", "nebius", "openai"], value="sambanova", label="LLM Provider")
                        multi_enrich = gr.Checkbox(label="✨ AI Enrichment", value=True, info="Better actor detection")
                    
                    multi_scan_btn = gr.Button("🔍 Generate Module Diagrams", variant="primary", size="lg", elem_classes=["primary-button"])
                
                with gr.Column(scale=1):
                    multi_status_banner = gr.Markdown(visible=False, elem_classes=["banner"])
                    multi_summary = gr.Textbox(label="Generated Diagrams", lines=5, interactive=False)
            
            gr.Markdown("### 📊 Diagrams by Module")
            
            multi_gallery = gr.State([])
            multi_module_selector = gr.Dropdown(label="Select Module to View", choices=[], interactive=True)
            
            with gr.Group():
                multi_diagram_img = gr.Image(label="Module Use Case Diagram", type="pil", elem_classes=["diagram-container"])
            
            with gr.Accordion("📝 PlantUML Source", open=False):
                multi_diagram_puml = gr.Code(language="markdown", label="PlantUML Code", lines=10)
            
            def update_diagram_viewer(diagrams_data, selected_module):
                if not diagrams_data or not selected_module:
                    return None, ""
                for d in diagrams_data:
                    if d["module"] == selected_module:
                        return d["image"], d["puml"]
                return None, ""
            
            multi_scan_btn.click(fn=process_folder_usecase_multi_zip, inputs=[multi_zip_input, multi_enrich, multi_provider], outputs=[multi_summary, multi_gallery, multi_module_selector, multi_diagram_img, multi_diagram_puml, multi_status_banner])
            multi_module_selector.change(fn=update_diagram_viewer, inputs=[multi_gallery, multi_module_selector], outputs=[multi_diagram_img, multi_diagram_puml])
        
        # TAB 4: MULTI-MODULE SEQUENCES
        with gr.Tab("🎬 Multi-Module Sequences", id=3):
            gr.Markdown("### Multiple Sequence Diagrams by Module\nGenerates **separate sequence diagrams for each module** - much clearer than one massive diagram!")
            
            gr.HTML('<div class="info-card"><strong>🎯 Smart Organization:</strong> Each service/module gets its own sequence flow. Perfect for understanding complex interactions across multiple components.</div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    seq_multi_zip_input = gr.File(
                        label="📦 Upload Project ZIP",
                        file_types=[".zip"],
                        type="filepath"
                    )

                    with gr.Row():
                        seq_multi_provider = gr.Dropdown(choices=["sambanova", "nebius", "openai"], value="sambanova", label="LLM Provider")
                        seq_multi_enrich = gr.Checkbox(label="✨ AI Enrichment", value=False, info="Slow but better names")
                    
                    seq_multi_scan_btn = gr.Button("🔍 Generate Module Sequences", variant="primary", size="lg", elem_classes=["primary-button"])
                
                with gr.Column(scale=1):
                    seq_multi_status_banner = gr.Markdown(visible=False, elem_classes=["banner"])
                    seq_multi_summary = gr.Textbox(label="Generated Diagrams", lines=5, interactive=False)
            
            gr.Markdown("### 🎬 Sequence Diagrams by Module")
            
            seq_multi_gallery = gr.State([])
            seq_multi_module_selector = gr.Dropdown(label="Select Module to View", choices=[], interactive=True)
            
            with gr.Group():
                seq_multi_diagram_img = gr.Image(label="Module Sequence Diagram", type="pil", elem_classes=["diagram-container"])
            
            with gr.Accordion("📝 PlantUML Source", open=False):
                seq_multi_diagram_puml = gr.Code(language="markdown", label="PlantUML Code", lines=10)
            
            def update_seq_diagram_viewer(diagrams_data, selected_module):
                if not diagrams_data or not selected_module:
                    return None, ""
                for d in diagrams_data:
                    if d["module"] == selected_module:
                        return d["image"], d["puml"]
                return None, ""
            
            seq_multi_scan_btn.click(fn=process_folder_sequence_multi_zip, inputs=[seq_multi_zip_input, seq_multi_enrich, seq_multi_provider], outputs=[seq_multi_summary, seq_multi_gallery, seq_multi_module_selector, seq_multi_diagram_img, seq_multi_diagram_puml, seq_multi_status_banner])
            seq_multi_module_selector.change(fn=update_seq_diagram_viewer, inputs=[seq_multi_gallery, seq_multi_module_selector], outputs=[seq_multi_diagram_img, seq_multi_diagram_puml])
    

        # TAB 5: AI PROPOSAL
        with gr.Tab("✨ AI Proposal" , id = 4):
            gr.Markdown("### Architecture Recommendations\nAI detects anti-patterns and suggests improvements.")
            gr.HTML('<div class="info-card"><strong>🧠 AI-Powered:</strong> Suggests Strategy, Factory, Singleton patterns, etc.</div>')
            
            with gr.Row():
                with gr.Column():
                    proposal_zip = gr.File(label="📦 Upload Project (ZIP)", file_types=[".zip"], type="filepath")
                    propose_btn = gr.Button("🧠 Generate Proposal", variant="primary", size="lg")
                    status_banner_3 = gr.Markdown(visible=False, elem_classes=["banner"])
                    proposal_output = gr.Code(language="json", label="📋 Analysis", lines=15)
                
                with gr.Column():
                    img_output_3 = gr.Image(label="🎨 Proposed Architecture", type="pil")
                    with gr.Accordion("📝 Proposed PlantUML", open=False):
                        text_output_3 = gr.Code(language="markdown", lines=10)
            
            propose_btn.click(
                fn=process_proposal_zip,
                inputs=proposal_zip,
                outputs=[proposal_output, text_output_3, img_output_3, status_banner_3]
            )
        
        # TAB 6: Modal Refactoring
        with gr.Tab("☁️ Safe Refactoring", id=5):
            gr.Markdown("### Production-Safe Cloud Execution\nRefactor code in isolated Modal sandboxes with testing.")
            gr.HTML('<div class="info-card"><strong>🛡️ Safety:</strong> Tests run in cloud. Files updated only if tests pass.</div>')
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📁 Configuration")
                    modal_zip = gr.File(label="📦 Upload Project (ZIP)", file_types=[".zip"], type="filepath")
                    file_dropdown = gr.Dropdown(label="Target File", choices=[], interactive=True)
                    test_dropdown = gr.Dropdown(label="Test File (Optional)", choices=[], interactive=True)
                    instruction_input = gr.Textbox(
                        label="Refactoring Instructions",
                        placeholder="Extract Strategy pattern for payment methods...",
                        lines=5
                    )
                    execute_btn = gr.Button("🚀 Execute in Modal", variant="stop", size="lg")
                
                with gr.Column():
                    gr.Markdown("#### 📊 Results")
                    modal_output = gr.Markdown(value="☁️ Waiting for execution...")
                    download_output = gr.File(label="📥 Download", visible=False)
            
            # Auto-populate dropdowns on ZIP upload
            modal_zip.change(
                fn=extract_file_list,
                inputs=modal_zip,
                outputs=[file_dropdown, test_dropdown]
            )
            
            execute_btn.click(
                fn=run_modal_refactoring_zip,
                inputs=[modal_zip, file_dropdown, instruction_input, test_dropdown],
                outputs=[modal_output, download_output]
            )
    
    # FOOTER
    gr.HTML("""
        <div class="footer">
            <p style="margin: 0.5rem 0; font-size: 1rem; font-family: 'JetBrains Mono', monospace;">
                built-for: <strong style="color: #00ff9f;">huggingface × anthropic-mcp hackathon</strong>
            </p>
            <p style="margin: 0.5rem 0; font-size: 0.875rem; color: #8b949e; font-family: 'JetBrains Mono', monospace;">
                » static-analysis + llm-intelligence + cloud-execution
            </p>
            <p style="margin: 1rem 0; font-size: 0.875rem; font-family: 'JetBrains Mono', monospace;">
                <a href="https://github.com/" target="_blank">→ github</a> |
                <a href="https://huggingface.co/" target="_blank">→ huggingface</a> |
                <a href="#" target="_blank">→ docs</a>
            </p>
        </div>
    """)





# architectai_theme = gr.themes.Soft(
#     primary_hue="emerald",
#     secondary_hue="purple",
#     neutral_hue="slate",
#     font=[
#         gr.themes.GoogleFont("Inter"),
#         "ui-sans-serif",
#         "system-ui",
#         "sans-serif"
#     ],
#     font_mono=[
#         gr.themes.GoogleFont("JetBrains Mono"),
#         "ui-monospace",
#         "monospace"
#     ],
# ).set(
#     # MODAL-STYLE DARK BACKGROUNDS (Deep Black)
#     body_background_fill="#0a0a0f",
#     body_background_fill_dark="#0a0a0f",
#     background_fill_primary="#0d1117",
#     background_fill_primary_dark="#0d1117",
#     background_fill_secondary="#161b22",
#     background_fill_secondary_dark="#161b22",

#     # BUTTONS (Neon Green with Purple Accents)
#     button_primary_background_fill="linear-gradient(135deg, #00ff9f 0%, #bc4aff 100%)",
#     button_primary_background_fill_hover="linear-gradient(135deg, #00ffaa 0%, #d946ef 100%)",
#     button_primary_background_fill_dark="linear-gradient(135deg, #00ff9f 0%, #bc4aff 100%)",
#     button_primary_text_color="#0a0a0f",
#     button_primary_text_color_dark="#0a0a0f",
#     button_secondary_background_fill="#161b22",
#     button_secondary_background_fill_hover="#00ff9f",
#     button_secondary_text_color="#00ff9f",
#     button_secondary_border_color="#00ff9f",

#     # TEXT COLORS (High Contrast)
#     body_text_color="#f0f6fc",
#     body_text_color_dark="#f0f6fc",
#     block_title_text_color="#f0f6fc",
#     block_title_text_color_dark="#f0f6fc",
#     block_label_text_color="#8b949e",
#     block_label_text_color_dark="#8b949e",

#     # BORDERS & SHADOWS (Neon Green Glow)
#     border_color_primary="rgba(0, 255, 159, 0.15)",
#     border_color_primary_dark="rgba(0, 255, 159, 0.15)",
#     shadow_drop="0 4px 20px rgba(0, 255, 159, 0.1)",
#     shadow_drop_lg="0 10px 40px rgba(0, 255, 159, 0.2)",

#     # INPUTS (Dark with Green Borders)
#     input_background_fill="#161b22",
#     input_background_fill_dark="#161b22",
#     input_background_fill_focus="#1c2128",
#     input_border_color="rgba(0, 255, 159, 0.2)",
#     input_border_color_dark="rgba(0, 255, 159, 0.2)",
#     input_border_color_focus="#00ff9f",
#     input_shadow_focus="0 0 0 3px rgba(0, 255, 159, 0.3), 0 0 20px rgba(0, 255, 159, 0.2)",

#     # BLOCKS (Dark Navy with Subtle Green Tint)
#     block_background_fill="#161b22",
#     block_background_fill_dark="#161b22",
#     block_border_color="rgba(0, 255, 159, 0.15)",
#     block_border_color_dark="rgba(0, 255, 159, 0.15)",
#     panel_background_fill="#0d1117",
#     panel_background_fill_dark="#0d1117",
# )


# architectai_theme = gr.themes.Soft(
#     primary_hue="orange",     # Warm terracotta accent like Claude
#     secondary_hue="slate",    # Neutral greys
#     neutral_hue="slate",      # Warm greys
#     spacing_size="md",        # Comfortable spacing
#     radius_size="md",         # Subtle rounded corners
#     font=[
#         gr.themes.GoogleFont("Inter"),
#         "-apple-system",
#         "BlinkMacSystemFont",
#         "Segoe UI",
#         "sans-serif"
#     ],
#     font_mono=[
#         gr.themes.GoogleFont("JetBrains Mono"),
#         "SF Mono",
#         "Consolas",
#         "monospace"
#     ],
# ).set(
#     # LIGHT MODE - Warm Paper Aesthetic
#     body_background_fill="#faf8f5",           # Warm off-white
#     body_background_fill_dark="#1a1816",      # Warm charcoal (dark mode)
    
#     background_fill_primary="#ffffff",         # Pure white cards
#     background_fill_primary_dark="#252220",    # Warm dark cards
    
#     background_fill_secondary="#f5f2ee",       # Subtle warm grey
#     background_fill_secondary_dark="#2d2a27",  # Warm dark grey
    
#     # BORDERS - Subtle and warm
#     border_color_primary="#e8e4dd",
#     border_color_primary_dark="#3d3935",
    
#     # TEXT - High readability
#     body_text_color="#2d2d2d",
#     body_text_color_dark="#e8e6e3",
    
#     block_title_text_color="#2d2d2d",
#     block_title_text_color_dark="#e8e6e3",
    
#     block_label_text_color="#666666",
#     block_label_text_color_dark="#a8a6a3",
    
#     # BUTTONS - Warm terracotta accent
#     button_primary_background_fill="#d97757",
#     button_primary_background_fill_hover="#c76646",
#     button_primary_background_fill_dark="#d97757",
#     button_primary_text_color="#ffffff",
#     button_primary_border_color="transparent",
    
#     button_secondary_background_fill="#ffffff",
#     button_secondary_background_fill_hover="#f5f2ee",
#     button_secondary_border_color="#e8e4dd",
#     button_secondary_text_color="#2d2d2d",
    
#     # INPUTS
#     input_background_fill="#ffffff",
#     input_background_fill_dark="#252220",
#     input_border_color="#e8e4dd",
#     input_border_color_dark="#3d3935",
#     input_border_color_focus="#d97757",
    
#     # SHADOWS - Very subtle
#     shadow_drop="0 1px 3px rgba(0, 0, 0, 0.08)",
#     shadow_drop_lg="0 4px 12px rgba(0, 0, 0, 0.1)",
    
#     # BLOCK STYLING
#     block_background_fill="#ffffff",
#     block_background_fill_dark="#252220",
#     block_border_width="1px",
#     block_border_color="#e8e4dd",
#     block_border_color_dark="#3d3935",
#     block_radius="12px",
    
#     # PANEL
#     panel_background_fill="#faf8f5",
#     panel_background_fill_dark="#1a1816",
# )

# architectai_css = """
# /* ===================================================================
#    1. GLOBAL STYLES & RESETS (Modal-Inspired)
#    =================================================================== */
# .gradio-container {
#     background: #0a0a0f !important;
#     color: #f0f6fc !important;
#     padding: 0 !important;
#     max-width: 1600px !important;
#     margin: 0 auto !important;
#     font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
#     position: relative;
# }

# /* Cyberpunk grid background - Neon Green */
# .gradio-container::before {
#     content: '';
#     position: fixed;
#     top: 0;
#     left: 0;
#     right: 0;
#     bottom: 0;
#     background-image:
#         linear-gradient(rgba(0, 255, 159, 0.03) 1px, transparent 1px),
#         linear-gradient(90deg, rgba(0, 255, 159, 0.03) 1px, transparent 1px);
#     background-size: 50px 50px;
#     pointer-events: none;
#     z-index: 0;
# }

# /* Override default Gradio backgrounds */
# .gr-box,
# .gr-form,
# .gr-panel,
# .gr-padded {
#     background: transparent !important;
#     border: none !important;
# }

# /* Smooth transitions */
# * {
#     transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
# }

# /* ===================================================================
#    2. HEADER & BRANDING (Modal Terminal Style)
#    =================================================================== */
# .main-header {
#     background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
#     padding: 3rem 2rem;
#     border-radius: 0;
#     border-bottom: 1px solid rgba(0, 255, 159, 0.2);
#     margin-bottom: 2rem;
#     color: white;
#     text-align: center;
#     box-shadow: 0 4px 30px rgba(0, 255, 159, 0.1), 0 0 60px rgba(188, 74, 255, 0.05);
#     position: relative;
#     overflow: hidden;
# }

# /* Animated neon grid background */
# .main-header::before {
#     content: '';
#     position: absolute;
#     top: 0;
#     left: 0;
#     right: 0;
#     bottom: 0;
#     background-image:
#         linear-gradient(rgba(0, 255, 159, 0.05) 1px, transparent 1px),
#         linear-gradient(90deg, rgba(0, 255, 159, 0.05) 1px, transparent 1px);
#     background-size: 30px 30px;
#     animation: gridScroll 20s linear infinite;
#     opacity: 0.5;
# }

# @keyframes gridScroll {
#     0% { transform: translate(0, 0); }
#     100% { transform: translate(30px, 30px); }
# }

# .main-header h1 {
#     font-size: 3rem;
#     font-weight: 800;
#     margin-bottom: 0.5rem;
#     background: linear-gradient(135deg, #00ff9f 0%, #bc4aff 100%);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
#     text-shadow: 0 0 30px rgba(0, 255, 159, 0.3);
#     position: relative;
#     z-index: 1;
#     filter: drop-shadow(0 0 20px rgba(0, 255, 159, 0.4));
# }

# .main-header .subtitle {
#     font-size: 1.25rem;
#     font-weight: 500;
#     margin: 0.5rem 0;
#     color: #00ff9f;
#     position: relative;
#     z-index: 1;
#     text-shadow: 0 0 20px rgba(0, 255, 159, 0.5);
# }

# .main-header .tagline {
#     font-size: 0.95rem;
#     margin: 0.5rem 0;
#     color: #8b949e;
#     position: relative;
#     z-index: 1;
#     font-family: 'JetBrains Mono', monospace;
# }

# /* Sponsor badges (Modal style with glassmorphism) */
# .sponsors {
#     display: flex;
#     justify-content: center;
#     gap: 1rem;
#     margin-top: 1.5rem;
#     flex-wrap: wrap;
#     position: relative;
#     z-index: 1;
# }

# .sponsor-badge {
#     background: rgba(22, 27, 34, 0.6);
#     backdrop-filter: blur(10px);
#     color: #00ff9f;
#     padding: 0.6rem 1.2rem;
#     border-radius: 8px;
#     border: 1px solid rgba(0, 255, 159, 0.2);
#     font-size: 0.85rem;
#     font-weight: 600;
#     box-shadow: 0 4px 20px rgba(0, 255, 159, 0.1);
#     transition: all 0.3s ease;
# }

# .sponsor-badge:hover {
#     transform: translateY(-2px);
#     border-color: rgba(0, 255, 159, 0.5);
#     box-shadow: 0 0 30px rgba(0, 255, 159, 0.3), 0 8px 20px rgba(0, 255, 159, 0.2);
#     background: rgba(0, 255, 159, 0.05);
# }

# /* ===================================================================
#    3. LAYOUT & CONTAINERS (Modal Glassmorphism)
#    =================================================================== */
# .output-card {
#     background: rgba(22, 27, 34, 0.6);
#     backdrop-filter: blur(16px);
#     border: 1px solid rgba(0, 255, 159, 0.15);
#     border-radius: 12px;
#     padding: 1.5rem;
#     box-shadow: 0 8px 32px rgba(0, 255, 159, 0.1), inset 0 1px 0 rgba(0, 255, 159, 0.1);
#     margin: 1rem 0;
#     transition: all 0.3s ease;
#     position: relative;
# }

# .output-card::before {
#     content: '';
#     position: absolute;
#     top: 0;
#     left: 0;
#     right: 0;
#     height: 1px;
#     background: linear-gradient(90deg, transparent, rgba(0, 255, 159, 0.3), transparent);
# }

# .output-card:hover {
#     border-color: rgba(0, 255, 159, 0.4);
#     box-shadow: 0 0 40px rgba(0, 255, 159, 0.2), 0 8px 32px rgba(188, 74, 255, 0.1);
#     transform: translateY(-2px);
# }

# /* ===================================================================
#    4. COMPONENTS - BUTTONS (Modal Neon Style)
#    =================================================================== */

# /* Primary buttons - Neon Green/Purple gradient */
# button[variant="primary"],
# .primary-button {
#     background: linear-gradient(135deg, #00ff9f 0%, #bc4aff 100%) !important;
#     color: #0a0a0f !important;
#     font-weight: 700 !important;
#     padding: 0.875rem 2rem !important;
#     border-radius: 8px !important;
#     border: 1px solid rgba(0, 255, 159, 0.3) !important;
#     box-shadow: 0 0 30px rgba(0, 255, 159, 0.3), 0 4px 20px rgba(0, 255, 159, 0.2) !important;
#     transition: all 0.3s ease !important;
#     font-size: 1rem !important;
#     text-transform: uppercase;
#     letter-spacing: 0.5px;
# }

# button[variant="primary"]:hover,
# .primary-button:hover {
#     transform: translateY(-2px) !important;
#     box-shadow: 0 0 50px rgba(0, 255, 159, 0.5), 0 8px 30px rgba(188, 74, 255, 0.3) !important;
#     background: linear-gradient(135deg, #00ffaa 0%, #d946ef 100%) !important;
#     border-color: rgba(0, 255, 159, 0.6) !important;
# }

# /* Secondary buttons - Outline style with neon green */
# button[variant="secondary"] {
#     background: rgba(22, 27, 34, 0.6) !important;
#     backdrop-filter: blur(10px) !important;
#     color: #00ff9f !important;
#     border: 2px solid #00ff9f !important;
#     font-weight: 600 !important;
#     padding: 0.875rem 2rem !important;
#     border-radius: 8px !important;
#     box-shadow: 0 0 20px rgba(0, 255, 159, 0.1) !important;
# }

# button[variant="secondary"]:hover {
#     background: rgba(0, 255, 159, 0.1) !important;
#     border-color: #00ffaa !important;
#     box-shadow: 0 0 40px rgba(0, 255, 159, 0.3) !important;
#     color: #00ffaa !important;
# }

# /* Stop/danger buttons - Pink/Red gradient */
# button[variant="stop"] {
#     background: linear-gradient(135deg, #ff006e 0%, #ff4d6d 100%) !important;
#     color: white !important;
#     font-weight: 600 !important;
#     border: 1px solid rgba(255, 0, 110, 0.3) !important;
#     box-shadow: 0 0 30px rgba(255, 0, 110, 0.3) !important;
# }

# button[variant="stop"]:hover {
#     transform: translateY(-2px) !important;
#     box-shadow: 0 0 50px rgba(255, 0, 110, 0.5) !important;
# }

# /* Input fields - Terminal/CLI style */
# input[type="text"],
# input[type="number"],
# textarea,
# select {
#     background: rgba(22, 27, 34, 0.8) !important;
#     color: #f0f6fc !important;
#     border: 1px solid rgba(0, 255, 159, 0.2) !important;
#     border-radius: 8px !important;
#     padding: 0.75rem 1rem !important;
#     font-size: 0.95rem !important;
#     font-family: 'JetBrains Mono', monospace !important;
# }

# input:focus,
# textarea:focus,
# select:focus {
#     border-color: #00ff9f !important;
#     box-shadow: 0 0 0 3px rgba(0, 255, 159, 0.2), 0 0 30px rgba(0, 255, 159, 0.1) !important;
#     outline: none !important;
#     background: rgba(28, 33, 40, 0.9) !important;
# }

# /* File upload areas - Neon dashed border */
# .file-upload {
#     background: rgba(22, 27, 34, 0.4) !important;
#     border: 2px dashed rgba(0, 255, 159, 0.3) !important;
#     border-radius: 12px !important;
#     padding: 2rem !important;
#     transition: all 0.3s ease !important;
#     backdrop-filter: blur(10px);
# }

# .file-upload:hover {
#     border-color: rgba(0, 255, 159, 0.6) !important;
#     background: rgba(0, 255, 159, 0.05) !important;
#     box-shadow: 0 0 40px rgba(0, 255, 159, 0.1) !important;
# }

# /* Checkboxes - Neon style */
# input[type="checkbox"] {
#     width: 20px !important;
#     height: 20px !important;
#     border-radius: 4px !important;
#     border: 2px solid rgba(0, 255, 159, 0.3) !important;
#     background: rgba(22, 27, 34, 0.8) !important;
# }

# input[type="checkbox"]:checked {
#     background: linear-gradient(135deg, #00ff9f 0%, #bc4aff 100%) !important;
#     border-color: #00ff9f !important;
#     box-shadow: 0 0 20px rgba(0, 255, 159, 0.3) !important;
# }

# /* ===================================================================
#    5. TABS & NAVIGATION (Modal Terminal Style)
#    =================================================================== */
# .tab-nav {
#     background: rgba(13, 17, 23, 0.8);
#     backdrop-filter: blur(10px);
#     border: 1px solid rgba(0, 255, 159, 0.1);
#     border-radius: 8px;
#     padding: 0.5rem;
#     margin-bottom: 2rem;
# }

# .tab-nav button {
#     color: #8b949e;
#     font-weight: 600;
#     padding: 0.75rem 1.5rem;
#     border-radius: 6px;
#     transition: all 0.2s ease;
#     font-family: 'JetBrains Mono', monospace;
#     font-size: 0.9rem;
# }

# .tab-nav button[aria-selected="true"] {
#     background: linear-gradient(135deg, #00ff9f 0%, #bc4aff 100%);
#     color: #0a0a0f;
#     font-weight: 700;
#     box-shadow: 0 0 30px rgba(0, 255, 159, 0.4), 0 4px 12px rgba(0, 255, 159, 0.2);
# }

# .tab-nav button:hover:not([aria-selected="true"]) {
#     background: rgba(0, 255, 159, 0.05);
#     color: #00ff9f;
#     border: 1px solid rgba(0, 255, 159, 0.2);
# }

# /* ===================================================================
#    6. OUTPUTS & RESULTS (Modal Style)
#    =================================================================== */

# /* Info cards - Neon accented */
# .info-card {
#     background: rgba(22, 27, 34, 0.6);
#     backdrop-filter: blur(16px);
#     border-left: 3px solid #00ff9f;
#     padding: 1.25rem;
#     border-radius: 8px;
#     margin: 1.5rem 0;
#     color: #f0f6fc;
#     box-shadow: 0 0 30px rgba(0, 255, 159, 0.05), inset 0 1px 0 rgba(0, 255, 159, 0.1);
# }

# .info-card strong {
#     color: #00ff9f;
#     font-weight: 700;
#     text-shadow: 0 0 10px rgba(0, 255, 159, 0.5);
# }

# /* Status banners */
# .banner {
#     padding: 1rem 1.5rem;
#     border-radius: 8px;
#     font-weight: 600;
#     text-align: center;
#     margin: 1rem 0;
#     animation: slideIn 0.3s ease-out;
#     backdrop-filter: blur(10px);
#     border: 1px solid rgba(0, 255, 159, 0.2);
# }

# .diagram-container {
#     background: #0a0a0f;
#     border: 1px solid rgba(0, 255, 159, 0.2);
#     border-radius: 8px;
#     padding: 1rem;
#     position: relative;
#     animation: scaleIn 0.3s ease-out;
#     box-shadow: 0 0 40px rgba(0, 255, 159, 0.05), inset 0 0 60px rgba(0, 255, 159, 0.02);
# }

# /* Neon grid effect for diagrams */
# .diagram-container::before {
#     content: '';
#     position: absolute;
#     top: 0;
#     left: 0;
#     right: 0;
#     bottom: 0;
#     background-image:
#         linear-gradient(rgba(0, 255, 159, 0.08) 1px, transparent 1px),
#         linear-gradient(90deg, rgba(0, 255, 159, 0.08) 1px, transparent 1px);
#     background-size: 20px 20px;
#     pointer-events: none;
#     border-radius: 8px;
# }

# /* Code blocks - Terminal style */
# pre, code {
#     background: #0a0a0f !important;
#     color: #00ff9f !important;
#     border: 1px solid rgba(0, 255, 159, 0.1) !important;
#     border-radius: 8px !important;
#     font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
# }

# .code-container {
#     border-radius: 8px !important;
#     overflow: hidden !important;
#     box-shadow: 0 0 30px rgba(0, 255, 159, 0.1) !important;
#     border: 1px solid rgba(0, 255, 159, 0.15) !important;
# }

# /* Accordions - Glassmorphism */
# details {
#     border: 1px solid rgba(0, 255, 159, 0.15) !important;
#     border-radius: 8px !important;
#     overflow: hidden !important;
#     margin: 1rem 0 !important;
#     background: rgba(22, 27, 34, 0.6) !important;
#     backdrop-filter: blur(10px);
# }

# summary {
#     background: rgba(0, 255, 159, 0.05) !important;
#     padding: 1rem 1.5rem !important;
#     font-weight: 600 !important;
#     cursor: pointer !important;
#     color: #00ff9f !important;
#     transition: all 0.2s ease;
# }

# summary:hover {
#     background: rgba(0, 255, 159, 0.1) !important;
#     box-shadow: 0 0 20px rgba(0, 255, 159, 0.1) !important;
# }

# details[open] summary {
#     border-bottom: 1px solid rgba(0, 255, 159, 0.2) !important;
# }

# /* ===================================================================
#    7. ANIMATIONS & TRANSITIONS
#    =================================================================== */
# @keyframes slideIn {
#     from {
#         opacity: 0;
#         transform: translateY(-10px);
#     }
#     to {
#         opacity: 1;
#         transform: translateY(0);
#     }
# }

# @keyframes fadeIn {
#     from {
#         opacity: 0;
#         transform: translateY(10px);
#     }
#     to {
#         opacity: 1;
#         transform: translateY(0);
#     }
# }

# @keyframes scaleIn {
#     from {
#         opacity: 0;
#         transform: scale(0.95);
#     }
#     to {
#         opacity: 1;
#         transform: scale(1);
#     }
# }

# /* Progress bars */
# .progress-bar {
#     height: 6px !important;
#     border-radius: 8px !important;
#     background: rgba(102, 126, 234, 0.2) !important;
# }

# .progress-bar-fill {
#     background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
#     height: 100% !important;
#     border-radius: 8px !important;
# }

# /* ===================================================================
#    8. RESPONSIVE DESIGN
#    =================================================================== */
# @media (max-width: 768px) {
#     .main-header h1 {
#         font-size: 2rem;
#     }

#     .main-header .subtitle {
#         font-size: 1rem;
#     }

#     .main-header .tagline {
#         font-size: 0.85rem;
#     }

#     .sponsors {
#         gap: 0.5rem;
#     }

#     .sponsor-badge {
#         font-size: 0.75rem;
#         padding: 0.5rem 1rem;
#     }

#     .output-card {
#         padding: 1rem;
#     }
# }

# /* ===================================================================
#    UTILITY CLASSES
#    =================================================================== */
# .text-gradient {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
# }

# .glow {
#     box-shadow: 0 0 20px rgba(102, 126, 234, 0.5) !important;
# }

# /* Footer styling - Modal terminal aesthetic */
# .footer {
#     text-align: center;
#     margin-top: 3rem;
#     padding: 2rem;
#     border-top: 1px solid rgba(0, 255, 159, 0.15);
#     color: #8b949e;
#     background: rgba(13, 17, 23, 0.6);
#     backdrop-filter: blur(10px);
#     position: relative;
# }

# .footer::before {
#     content: '';
#     position: absolute;
#     top: 0;
#     left: 0;
#     right: 0;
#     height: 1px;
#     background: linear-gradient(90deg, transparent, rgba(0, 255, 159, 0.5), transparent);
# }

# .footer a {
#     color: #00ff9f;
#     text-decoration: none;
#     font-weight: 600;
#     transition: all 0.2s ease;
#     font-family: 'JetBrains Mono', monospace;
# }

# .footer a:hover {
#     color: #bc4aff;
#     text-shadow: 0 0 10px rgba(188, 74, 255, 0.5);
# }
# """



# architectai_theme = gr.themes.Soft(
#     primary_hue="indigo",    # Professional primary color
#     secondary_hue="slate",   # Neutral secondary
#     neutral_hue="slate",     # Modern greys
#     font=[
#         gr.themes.GoogleFont("Inter"),
#         "ui-sans-serif",
#         "system-ui",
#         "sans-serif"
#     ],
#     font_mono=[
#         gr.themes.GoogleFont("JetBrains Mono"),
#         "ui-monospace",
#         "monospace"
#     ],
# ).set(
#     # GLOBAL ROUNDING & SPACING
#     button_large_radius="8px",
#     block_radius="12px",
    
#     # LIGHT MODE DEFAULTS (Gradio handles these, but we refine them)
#     body_background_fill="#f8fafc",
#     block_background_fill="#ffffff",
    
#     # DARK MODE OVERRIDES (Specific deep colors)
#     body_background_fill_dark="#0b0f19",
#     block_background_fill_dark="#111827",
#     background_fill_primary_dark="#111827",
    
#     # BUTTONS
#     button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
#     button_primary_background_fill_dark="linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
#     button_primary_border_color="transparent",
    
#     # BORDERS
#     block_border_width="1px",
#     block_border_color_dark="rgba(255, 255, 255, 0.1)",
# )
architectai_theme = gr.themes.Soft(
    primary_hue="orange",     # Warm terracotta accent like Claude
    secondary_hue="slate",    # Neutral greys
    neutral_hue="slate",      # Warm greys
    spacing_size="md",        # Comfortable spacing
    radius_size="md",         # Subtle rounded corners
    font=[
        gr.themes.GoogleFont("Inter"),
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "sans-serif"
    ],
    font_mono=[
        gr.themes.GoogleFont("JetBrains Mono"),
        "SF Mono",
        "Consolas",
        "monospace"
    ],
).set(
    # LIGHT MODE - Warm Paper Aesthetic
    body_background_fill="#faf8f5",           # Warm off-white
    body_background_fill_dark="#1a1816",      # Warm charcoal (dark mode)
    
    background_fill_primary="#ffffff",         # Pure white cards
    background_fill_primary_dark="#252220",    # Warm dark cards
    
    background_fill_secondary="#f5f2ee",       # Subtle warm grey
    background_fill_secondary_dark="#2d2a27",  # Warm dark grey
    
    # BORDERS - Subtle and warm
    border_color_primary="#e8e4dd",
    border_color_primary_dark="#3d3935",
    
    # TEXT - High readability
    body_text_color="#2d2d2d",
    body_text_color_dark="#e8e6e3",
    
    block_title_text_color="#2d2d2d",
    block_title_text_color_dark="#e8e6e3",
    
    block_label_text_color="#666666",
    block_label_text_color_dark="#a8a6a3",
    
    # BUTTONS - Warm terracotta accent
    button_primary_background_fill="#d97757",
    button_primary_background_fill_hover="#c76646",
    button_primary_background_fill_dark="#d97757",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f5f2ee",
    button_secondary_border_color="#e8e4dd",
    button_secondary_text_color="#2d2d2d",
    
    # INPUTS
    input_background_fill="#ffffff",
    input_background_fill_dark="#252220",
    input_border_color="#e8e4dd",
    input_border_color_dark="#3d3935",
    input_border_color_focus="#d97757",
    
    # SHADOWS - Very subtle
    shadow_drop="0 1px 3px rgba(0, 0, 0, 0.08)",
    shadow_drop_lg="0 4px 12px rgba(0, 0, 0, 0.1)",
    
    # BLOCK STYLING
    block_background_fill="#ffffff",
    block_background_fill_dark="#252220",
    block_border_width="1px",
    block_border_color="#e8e4dd",
    block_border_color_dark="#3d3935",
    block_radius="12px",
    
    # PANEL
    panel_background_fill="#faf8f5",
    panel_background_fill_dark="#1a1816",
)


architectai_css = """
/* ===================================================================
   GLOBAL - Warm Paper Aesthetic
   =================================================================== */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: #faf8f5 !important;
    color: #2d2d2d !important;
}

/* Remove default gradio backgrounds */
.gr-box,
.gr-form,
.gr-panel {
    background: transparent !important;
    border: none !important;
}

/* Smooth transitions */
* {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* ===================================================================
   HEADER - Serif Typography Like Claude
   =================================================================== */
.main-header {
    background: transparent;
    border-bottom: 1px solid #e8e4dd;
    border-radius: 0;
    padding: 3rem 2rem 2.5rem;
    margin-bottom: 2.5rem;
    text-align: center;
    box-shadow: none;
}

.main-header h1 {
    font-family: 'Georgia', 'Baskerville', 'Times New Roman', serif !important;
    font-size: 2.5rem;
    font-weight: 400; /* Light weight for elegance */
    margin: 0 0 0.75rem 0;
    color: #2d2d2d;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.main-header .subtitle {
    font-size: 1.05rem;
    color: #666666;
    margin: 0 auto 2rem;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.6;
}

.main-header .tagline {
    font-size: 0.875rem;
    color: #888888;
    margin: 0.5rem 0 1.5rem;
    font-family: 'Inter', sans-serif;
    font-weight: 400;
}

/* Sponsors - Minimalist Pills */
.sponsors {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}

.sponsor-badge {
    background: #ffffff;
    color: #666666;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    border: 1px solid #e8e4dd;
    font-size: 0.8rem;
    font-weight: 500;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.sponsor-badge:hover {
    border-color: #d97757;
    color: #d97757;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}

/* ===================================================================
   COMPONENTS - Clean & Minimal
   =================================================================== */

/* Cards/Output Groups */
.output-card {
    background: #ffffff !important;
    border: 1px solid #e8e4dd !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
    margin: 1rem 0 !important;
}

.output-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}

/* Info Cards - Warm Accent */
.info-card {
    background: #f9f6f2;
    border: 1px solid #e8e4dd;
    border-left: 3px solid #d97757;
    padding: 1.25rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    color: #2d2d2d;
    font-size: 0.95rem;
    line-height: 1.6;
}

.info-card strong {
    color: #d97757;
    font-weight: 600;
}

/* Status Banners - Subtle */
.banner {
    padding: 1rem 1.5rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 500;
    margin: 1rem 0;
    background: #f5f2ee;
    border: 1px solid #e8e4dd;
    color: #2d2d2d;
    font-size: 0.9rem;
}

/* ===================================================================
   BUTTONS - Warm & Tactile
   =================================================================== */

/* Primary Button - Terracotta */
button[variant="primary"],
.primary-button {
    background: #d97757 !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    font-size: 0.95rem !important;
}

button[variant="primary"]:hover,
.primary-button:hover {
    background: #c76646 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12) !important;
}

button[variant="primary"]:active {
    transform: translateY(0) !important;
}

/* Secondary Buttons */
button[variant="secondary"] {
    background: #ffffff !important;
    color: #2d2d2d !important;
    border: 1px solid #e8e4dd !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

button[variant="secondary"]:hover {
    background: #f5f2ee !important;
    border-color: #d97757 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08) !important;
}

/* ===================================================================
   INPUTS - Clean & Focused
   =================================================================== */

input[type="text"],
input[type="number"],
textarea,
select {
    background: #ffffff !important;
    color: #2d2d2d !important;
    border: 1px solid #e8e4dd !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
}

input:focus,
textarea:focus,
select:focus {
    border-color: #d97757 !important;
    box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.1) !important;
    outline: none !important;
}

/* File Upload */
.file-upload {
    background: #fafaf9 !important;
    border: 2px dashed #e8e4dd !important;
    border-radius: 12px !important;
    padding: 2rem !important;
}

.file-upload:hover {
    border-color: #d97757 !important;
    background: #f9f6f2 !important;
}

/* Checkboxes */
input[type="checkbox"] {
    width: 18px !important;
    height: 18px !important;
    border-radius: 4px !important;
    border: 1.5px solid #e8e4dd !important;
    background: #ffffff !important;
}

input[type="checkbox"]:checked {
    background: #d97757 !important;
    border-color: #d97757 !important;
}

/* ===================================================================
   TABS - Clean Navigation
   =================================================================== */
.tab-nav {
    background: transparent;
    border-bottom: 1px solid #e8e4dd;
    padding: 0;
    margin-bottom: 2rem;
    border-radius: 0;
}

.tab-nav button {
    color: #666666;
    font-weight: 500;
    padding: 0.875rem 1.25rem;
    border-radius: 0;
    background: transparent;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    font-size: 0.9rem;
}

.tab-nav button[aria-selected="true"] {
    color: #d97757;
    border-bottom-color: #d97757;
    background: transparent;
    font-weight: 600;
}

.tab-nav button:hover:not([aria-selected="true"]) {
    color: #2d2d2d;
    background: rgba(217, 119, 87, 0.04);
}

/* ===================================================================
   DIAGRAMS - Clean Canvas
   =================================================================== */
.diagram-container {
    background: #ffffff;
    border: 1px solid #e8e4dd;
    border-radius: 8px;
    padding: 2rem;
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ===================================================================
   CODE BLOCKS - Monospace Clarity
   =================================================================== */
pre,
code {
    background: #f5f2ee !important;
    color: #2d2d2d !important;
    border: 1px solid #e8e4dd !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace !important;
    font-size: 0.875rem !important;
}

.code-container {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #e8e4dd !important;
}

/* ===================================================================
   ACCORDIONS - Subtle Expansion
   =================================================================== */
details {
    border: 1px solid #e8e4dd !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    margin: 1rem 0 !important;
    background: #ffffff !important;
}

summary {
    background: #f9f6f2 !important;
    padding: 1rem 1.25rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    color: #2d2d2d !important;
    font-size: 0.9rem;
}

summary:hover {
    background: #f5f2ee !important;
}

details[open] summary {
    border-bottom: 1px solid #e8e4dd !important;
}

/* ===================================================================
   FOOTER - Minimalist
   =================================================================== */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding: 2rem 1rem;
    border-top: 1px solid #e8e4dd;
    color: #888888;
    font-size: 0.875rem;
}

.footer a {
    color: #666666;
    text-decoration: none;
    margin: 0 0.75rem;
    border-bottom: 1px dotted #e8e4dd;
    transition: all 0.2s;
}

.footer a:hover {
    color: #d97757;
    border-bottom-color: #d97757;
}

/* ===================================================================
   RESPONSIVE DESIGN
   =================================================================== */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .main-header .subtitle {
        font-size: 0.95rem;
    }
    
    .sponsors {
        gap: 0.5rem;
    }
    
    .sponsor-badge {
        font-size: 0.75rem;
        padding: 0.4rem 0.875rem;
    }
    
    .output-card {
        padding: 1rem !important;
    }
}

/* ===================================================================
   ANIMATIONS - Subtle & Smooth
   =================================================================== */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

.banner {
    animation: slideIn 0.3s ease-out;
}

.output-card {
    animation: fadeIn 0.4s ease-out;
}

/* ===================================================================
   UTILITY CLASSES
   =================================================================== */
.text-muted {
    color: #888888;
}

.text-accent {
    color: #d97757;
}

.border-accent {
    border-color: #d97757 !important;
}
"""

# ===================================================================
# UPDATED HTML HEADER (More Minimalist)
# ===================================================================

css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    /* Claude-like Warm Paper Background */
    --body-background-fill: #fbf9f6;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f4f1ed;
    --border-color-primary: #e6e3df;
    --body-text-color: #383838;
    --body-text-color-subdued: #6b6b6b;
    --color-accent: #da7756; /* Burnt Orange/Terracotta */
    --link-text-color: #da7756;
}

/* Warm Dark Mode Overrides */
.gradio-container.dark {
    background: #191817 !important; /* Warm Charcoal */
    --body-background-fill: #191817;
    --background-fill-primary: #222120;
    --background-fill-secondary: #2c2b2a;
    --border-color-primary: #3a3938;
    --body-text-color: #e8e6e3;
    --body-text-color-subdued: #a3a19f;
    --color-accent: #e08c70;
}

/* ===================================================================
   HEADER STYLES (Serif & Minimalist)
   =================================================================== */
.main-header {
    background: transparent;
    border-bottom: 1px solid var(--border-color-primary);
    border-radius: 0;
    padding: 3rem 1.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    box-shadow: none; /* Removed glow */
}

/* Header Title - Serif Font like Claude */
.main-header h1 {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 2.75rem;
    font-weight: 500; /* Lighter weight for elegance */
    margin: 0;
    color: var(--body-text-color);
    background: none;
    -webkit-text-fill-color: initial;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.title-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.75rem;
}

.version-badge {
    background: var(--background-fill-secondary);
    color: var(--body-text-color-subdued);
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: 500;
    border: 1px solid var(--border-color-primary);
    letter-spacing: 0.05em;
}

.subtitle {
    font-size: 1.15rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 2rem;
    font-weight: 400;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

/* Tags - Pill style */
.tag-container {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}

.tech-tag {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    background: var(--background-fill-primary);
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid var(--border-color-primary);
    transition: all 0.2s ease;
}

.tech-tag:hover {
    border-color: var(--color-accent);
    color: var(--color-accent);
}

/* Sponsors - Minimalist */
.sponsors {
    display: flex;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
    opacity: 0.8;
}

.sponsor-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--body-text-color-subdued);
    background: transparent;
    padding: 0;
    border: none;
}

.dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--color-accent);
    opacity: 0.7;
}

/* ===================================================================
   COMPONENT STYLES (Warm & Flat)
   =================================================================== */

/* Cards/Groups */
.output-card {
    background: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Subtle shadow */
    transition: box-shadow 0.2s ease;
}

.output-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border-color: var(--border-color-primary);
}

/* Info Cards */
.info-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-left: 4px solid var(--color-accent);
    padding: 1.25rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    color: var(--body-text-color);
    font-size: 0.95rem;
    line-height: 1.5;
}

.info-card strong {
    color: var(--body-text-color);
    font-weight: 600;
}

/* Code Inputs - Clean */
.code-input textarea, .code-input .cm-scroller {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px;
    background-color: var(--background-fill-secondary) !important;
    border: 1px solid var(--border-color-primary) !important;
}

/* Diagrams */
.diagram-container {
    background: #ffffff; /* Always white for diagrams in light mode */
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 2rem;
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.dark .diagram-container {
    background: #2c2b2a;
}

/* Status Banners */
.banner {
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 500;
    margin-bottom: 1rem;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    color: var(--body-text-color);
}

/* ===================================================================
   BUTTONS (Muted & Tactile)
   =================================================================== */

/* Primary buttons - Warm Accent */
.primary-button {
    background: var(--color-accent) !important;
    color: white !important;
    text-transform: none !important; /* Normal case */
    letter-spacing: normal !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    border: 1px solid transparent !important;
}

.primary-button:hover {
    transform: translateY(-1px);
    filter: brightness(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
}

/* Secondary Buttons */
button.secondary {
    background: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary) !important;
    color: var(--body-text-color) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

/* ===================================================================
   FOOTER
   =================================================================== */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border-color-primary);
    color: var(--body-text-color-subdued);
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

.footer a {
    color: var(--body-text-color-subdued);
    text-decoration: none;
    margin: 0 10px;
    border-bottom: 1px dotted var(--border-color-primary);
    transition: border-color 0.2s;
}

.footer a:hover {
    color: var(--color-accent);
    border-bottom: 1px solid var(--color-accent);
    text-decoration: none;
}

"""


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
       # theme=architectai_theme,
        css=architectai_css,
    )