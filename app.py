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
from core.llm_factory import create_openai_llm, create_sambanova_llm, create_nebius_llm

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
            "nebius": [create_nebius_llm, create_openai_llm, create_sambanova_llm]
        }
        
        factories = strategies.get(preferred_provider, strategies["openai"])
        names = ["openai", "sambanova", "nebius"]
        
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

# --- TAB 3: AI PROPOSAL ---
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

# --- TAB 4: MODAL REFACTORING ---
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
            <h1>🏛️ ArchitectAI</h1>
            <p>Autonomous Cloud Refactoring with Modal Sandboxes</p>
            <div class="sponsors">
                <span class="sponsor-badge">🔷 Anthropic MCP</span>
                <span class="sponsor-badge">☁️ Modal</span>
                <span class="sponsor-badge">🤖 OpenAI</span>
                <span class="sponsor-badge">⚡ SambaNova</span>
                <span class="sponsor-badge">🚀 Nebius</span>
            </div>
        </div>
    """)
    
    # TABS
    with gr.Tabs():
        
        # TAB 1: Single File
        with gr.Tab("📄 Single File Analysis"):
            gr.Markdown("### Quick Code Analysis\nPaste Python code to generate instant UML diagram.")
            
            with gr.Row():
                with gr.Column():
                    code_input = gr.Code(language="python", label="Python Code", lines=15)
                    enrich_checkbox = gr.Checkbox(
                        label="✨ AI Type Enrichment",
                        value=False,
                        info="Use AI to infer missing type hints"
                    )
                    analyze_btn = gr.Button("🚀 Analyze Code", variant="primary", size="lg")
                
                with gr.Column():
                    status_banner_1 = gr.Markdown(visible=False, elem_classes=["banner"])
                    img_output_1 = gr.Image(label="📊 Class Diagram", type="pil")
                    with gr.Accordion("📝 PlantUML Source", open=False):
                        text_output_1 = gr.Code(language="markdown", lines=10)
            
            analyze_btn.click(
                fn=process_code_snippet,
                inputs=[code_input, enrich_checkbox],
                outputs=[text_output_1, img_output_1, status_banner_1]
            )
        
        # TAB 2: Project Map
        with gr.Tab("📂 Project Map"):
            gr.Markdown("### Full Project Analysis\nUpload ZIP to visualize all classes and relationships.")
            gr.HTML('<div class="info-card"><strong>💡 Tip:</strong> Works best with 5-50 Python files.</div>')
            
            with gr.Row():
                with gr.Column():
                    project_zip = gr.File(label="📦 Upload Project (ZIP)", file_types=[".zip"], type="filepath")
                    scan_btn = gr.Button("🔍 Scan Project", variant="primary", size="lg")
                
                with gr.Column():
                    status_banner_2 = gr.Markdown(visible=False, elem_classes=["banner"])
                    img_output_2 = gr.Image(label="🗺️ Architecture", type="pil")
                    with gr.Accordion("📝 PlantUML Source", open=False):
                        text_output_2 = gr.Code(language="markdown", lines=10)
            
            scan_btn.click(
                fn=process_zip_upload,
                inputs=project_zip,
                outputs=[text_output_2, img_output_2, status_banner_2]
            )
        
        # TAB 3: AI Proposal
        with gr.Tab("✨ AI Proposal"):
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
        
        # TAB 4: Modal Refactoring
        with gr.Tab("☁️ Safe Refactoring"):
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
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e9ecef;">
            <p style="color: #6c757d;">
                Built for <strong>Hugging Face x Anthropic MCP Hackathon</strong><br>
                <a href="#">📺 Demo</a> • <a href="#">📖 Docs</a> • <a href="#">💬 Social</a>
            </p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css=custom_css,

    )