import gradio as gr
import ast
import logging
import io
import os 
import json
import sys
from pathlib import Path
from PIL import Image
from plantuml import PlantUML
import zipfile
import tempfile
import shutil
# --- SETUP PATHS ---
sys.path.insert(0, str(Path(__file__).parent))

# --- IMPORTS ---
from services.architecture_service import ArchitectureVisitor, DeterministicPlantUMLConverter, FastTypeEnricher
from services.project_service import ProjectAnalyzer 
from services.refactoring_service import RefactoringAdvisor
from core.llm_factory import create_openai_llm, create_sambanova_llm, create_nebius_llm

# Setup logging
logging.basicConfig(level=logging.INFO)

# --- CONFIG ---
PLANTUML_SERVER_URL = 'http://www.plantuml.com/plantuml/img/'
plantuml_client = PlantUML(url=PLANTUML_SERVER_URL)

# --- 🔧 SINGLETON LLM CLIENT 🔧 ---
class LLMClientSingleton:
    _instance = None
    _llm_client = None
    _current_provider = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self, preferred_provider: str = "openai", temperature: float = 0.0):
        if self._llm_client is not None and self._current_provider == preferred_provider:
            return self._llm_client
        
        strategies = [
            ("openai", create_openai_llm), 
            ("sambanova", create_sambanova_llm), 
            ("nebius", create_nebius_llm)
        ]
        
        if preferred_provider == "sambanova": 
            strategies.insert(0, strategies.pop(1))
        elif preferred_provider == "nebius": 
            strategies.insert(0, strategies.pop(2))

        for name, factory in strategies:
            try:
                self._llm_client = factory(temperature=temperature)
                self._current_provider = name
                logging.info(f"✅ Connected to {name}")
                return self._llm_client
            except Exception as e:
                logging.warning(f"⚠️ {name} failed: {str(e)[:100]}")
        return None

_llm_singleton = LLMClientSingleton()

def extract_file_list(zip_path):
    """Extract list of .py files from uploaded ZIP"""
    if not zip_path:
        return gr.update(choices=[]), gr.update(choices=[])
    
    try:
        py_files = []
        test_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith('.py') and not file_info.is_dir():
                    py_files.append(file_info.filename)
                    
                    # Separate test files
                    if 'test' in file_info.filename.lower():
                        test_files.append(file_info.filename)
        
        # Sort for better UX
        py_files.sort()
        test_files.sort()
        
        return (
            gr.update(choices=py_files, value=py_files[0] if py_files else None),
            gr.update(choices=test_files, value=None)
        )
        
    except Exception as e:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None)
        )
    
def process_proposal_zip(zip_path, progress=gr.Progress()):
    """TAB 3: AI Proposal from ZIP"""
    if not zip_path:
        return "⚠️ Please upload a ZIP file.", None, None, gr.update(visible=True, value="⚠️ No File")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        progress(0.2, desc="Extracting ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        progress(0.5, desc="AI analyzing architecture...")
        analyzer = ProjectAnalyzer(Path(temp_dir))
        structure = analyzer.analyze()
        
        if not structure:
            shutil.rmtree(temp_dir)
            return "⚠️ No code found.", None, None, gr.update(visible=True, value="⚠️ No Code")
        
        advisor = RefactoringAdvisor()
        proposal = advisor.propose_improvement(structure)
        
        if "error" in proposal:
            shutil.rmtree(temp_dir)
            return f"❌ AI Error: {proposal['error']}", None, None, gr.update(visible=True, value="❌ AI Failed")
        
        progress(0.8, desc="Generating proposed UML...")
        puml_code = proposal.get("proposed_uml", "")
        _, image = render_plantuml(puml_code)
        
        shutil.rmtree(temp_dir)
        progress(1.0, desc="Complete!")
        
        return json.dumps(proposal, indent=2), puml_code, image, gr.update(visible=True, value="✅ Proposal Generated")
        
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return f"❌ Error: {e}", None, None, gr.update(visible=True, value="❌ Failed")
    
def process_zip_upload(zip_path, progress=gr.Progress()):
    """TAB 2: Extract ZIP and analyze project"""
    if not zip_path:
        return "⚠️ Please upload a ZIP file.", None, gr.update(visible=True, value="⚠️ No File")
    
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        progress(0.2, desc="Extracting ZIP file...")
        
        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        progress(0.5, desc="Analyzing project structure...")
        
        # Analyze
        analyzer = ProjectAnalyzer(Path(temp_dir))
        full_structure = analyzer.analyze()
        
        if not full_structure:
            shutil.rmtree(temp_dir)
            return "⚠️ No Python code found in ZIP.", None, gr.update(visible=True, value="⚠️ No Code")
        
        progress(0.8, desc="Generating diagram...")
        
        converter = DeterministicPlantUMLConverter()
        puml_text = converter.convert(full_structure)
        text, image = render_plantuml(puml_text)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        progress(1.0, desc="Complete!")
        
        return text, image, gr.update(visible=True, value=f"✅ Found {len(full_structure)} components")
        
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return f"❌ Error: {e}", None, gr.update(visible=True, value="❌ Failed")


def run_modal_refactoring_zip(zip_path, file_path, instruction, test_path=None, progress=gr.Progress()):
    """TAB 4: Modal execution with ZIP upload"""
    if not zip_path:
        return "⚠️ Please upload a ZIP file.", gr.update(visible=False)
    if not file_path or not instruction:
        return "⚠️ Please provide file path and instruction.", gr.update(visible=False)
    
    try:
        # Create temp directories
        temp_extract = tempfile.mkdtemp()
        temp_output = tempfile.mkdtemp()
        
        progress(0.1, desc="Extracting project...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        
        progress(0.3, desc="Preparing for Modal...")
        
        # Read target file
        target_file = Path(temp_extract) / file_path
        if not target_file.exists():
            shutil.rmtree(temp_extract)
            shutil.rmtree(temp_output)
            return f"❌ File not found: {file_path}", gr.update(visible=False)
        
        original_code = target_file.read_text(encoding='utf-8')
        
        # Read test file if provided
        test_code = None
        if test_path and test_path.strip():
            test_file = Path(temp_extract) / test_path
            if test_file.exists():
                test_code = test_file.read_text(encoding='utf-8')
        
        progress(0.5, desc="Executing in Modal sandbox...")
        
        # Call Modal function (import from server.py)
        try:
            from server import apply_refactoring_safely
            
            # Execute refactoring in Modal
            result = apply_refactoring_safely(file_path, instruction, test_path)
            
            progress(0.8, desc="Packaging results...")
            
            # If successful, create output ZIP
            if "✅" in result and "PASSED" in result:
                # Copy entire project to output
                shutil.copytree(temp_extract, Path(temp_output) / "project")
                
                # Create ZIP
                output_zip_path = Path(temp_output) / "refactored_project.zip"
                with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(Path(temp_output) / "project"):
                        for file in files:
                            file_path_full = Path(root) / file
                            arcname = file_path_full.relative_to(Path(temp_output) / "project")
                            zipf.write(file_path_full, arcname)
                
                progress(1.0, desc="Complete!")
                
                # Cleanup extract dir
                shutil.rmtree(temp_extract)
                
                return result, gr.update(visible=True, value=str(output_zip_path))
            else:
                # Failed - don't provide download
                shutil.rmtree(temp_extract)
                shutil.rmtree(temp_output)
                return result, gr.update(visible=False)
                
        except ImportError:
            shutil.rmtree(temp_extract)
            shutil.rmtree(temp_output)
            return "⚠️ Modal integration not configured. Set up Modal credentials in Space secrets.", gr.update(visible=False)
            
    except Exception as e:
        if 'temp_extract' in locals():
            shutil.rmtree(temp_extract)
        if 'temp_output' in locals():
            shutil.rmtree(temp_output)
        return f"❌ Execution failed: {str(e)}", gr.update(visible=False)
# --- HELPER FUNCTIONS ---
def render_plantuml(puml_text: str):
    """Render PlantUML to image"""
    if not puml_text: 
        return None, None
    try:
        image_bytes = plantuml_client.processes(puml_text)
        image = Image.open(io.BytesIO(image_bytes))
        return puml_text, image
    except Exception as e:
        return f"{puml_text}\n\n⚠️ Render Error: {e}", None

# --- CORE LOGIC FUNCTIONS ---
def process_code_snippet(code_snippet: str, enrich_types: bool = False):
    """TAB 1: Single File Analysis"""
    if not code_snippet.strip(): 
        return "⚠️ Please enter some code.", None, gr.update(visible=False)
    
    try:
        tree = ast.parse(code_snippet)
        visitor = ArchitectureVisitor()
        visitor.visit(tree)
        
        if not visitor.structure: 
            return "⚠️ No classes/functions found.", None, gr.update(visible=False)

        # Optional AI enrichment
        if enrich_types:
            try:
                llm = _llm_singleton.get_client(preferred_provider="openai", temperature=0.0)
                if llm:
                    enricher = FastTypeEnricher(llm)
                    visitor.structure = enricher.enrich(code_snippet, visitor.structure)
                    logging.info("✓ Type enrichment complete")
            except Exception as e: 
                logging.warning(f"Enrichment failed: {e}")

        converter = DeterministicPlantUMLConverter()
        puml_text = converter.convert(visitor.structure)
        text, image = render_plantuml(puml_text)
        
        # Return with success banner
        return text, image, gr.update(visible=True, value="✅ Analysis Complete!")
        
    except SyntaxError as se:
        return f"❌ Syntax Error: {se}", None, gr.update(visible=True, value=f"❌ Syntax Error")
    except Exception as e:
        return f"❌ Error: {e}", None, gr.update(visible=True, value=f"❌ Error")

def process_folder(folder_path: str, progress=gr.Progress()):
    """TAB 2: Project Analysis"""
    path_obj = Path(folder_path)
    
    if not path_obj.exists(): 
        return "❌ Path not found.", None, gr.update(visible=True, value="❌ Path Not Found")
    if not path_obj.is_dir():
        return "❌ Not a directory.", None, gr.update(visible=True, value="❌ Invalid Path")
    
    try:
        progress(0.2, desc="Scanning project structure...")
        analyzer = ProjectAnalyzer(path_obj)
        
        progress(0.5, desc="Analyzing classes and relationships...")
        full_structure = analyzer.analyze()
        
        if not full_structure: 
            return "⚠️ No Python code found.", None, gr.update(visible=True, value="⚠️ No Code Found")
        
        progress(0.8, desc="Generating UML diagram...")
        converter = DeterministicPlantUMLConverter()
        puml_text = converter.convert(full_structure)
        
        progress(1.0, desc="Complete!")
        text, image = render_plantuml(puml_text)
        
        return text, image, gr.update(visible=True, value=f"✅ Found {len(full_structure)} components")
        
    except Exception as e: 
        return f"❌ Error: {e}", None, gr.update(visible=True, value=f"❌ Analysis Failed")

def process_proposal(folder_path: str, progress=gr.Progress()):
    """TAB 3: AI Refactoring Proposal"""
    path_obj = Path(folder_path)
    
    if not path_obj.exists(): 
        return "❌ Path not found.", None, None, gr.update(visible=True, value="❌ Path Not Found")
    
    try:
        progress(0.2, desc="Analyzing project structure...")
        analyzer = ProjectAnalyzer(path_obj)
        structure = analyzer.analyze()
        
        if not structure: 
            return "⚠️ No code found.", None, None, gr.update(visible=True, value="⚠️ No Code")
        
        progress(0.5, desc="AI is analyzing architecture...")
        advisor = RefactoringAdvisor()
        proposal = advisor.propose_improvement(structure)
        
        if "error" in proposal: 
            return f"❌ AI Error: {proposal['error']}", None, None, gr.update(visible=True, value="❌ AI Failed")
        
        progress(0.8, desc="Generating proposed architecture...")
        puml_code = proposal.get("proposed_uml", "")
        _, image = render_plantuml(puml_code)
        
        progress(1.0, desc="Complete!")
        
        # Format JSON nicely
        proposal_text = json.dumps(proposal, indent=2)
        
        return proposal_text, puml_code, image, gr.update(visible=True, value="✅ Proposal Generated")
        
    except Exception as e: 
        return f"❌ Error: {e}", None, None, gr.update(visible=True, value=f"❌ Failed")

def run_modal_refactoring(file_path: str, instruction: str, test_path: str = None, progress=gr.Progress()):
    """TAB 4: Modal Cloud Execution"""
    if not file_path or not instruction:
        return gr.update(visible=True, value="⚠️ Please provide file path and instruction")
    
    try:
        progress(0.1, desc="Preparing sandbox...")
        
        # Import Modal function (lazy import to avoid issues if Modal not configured)
        try:
            from server import apply_refactoring_safely
            progress(0.3, desc="Uploading to Modal...")
            
            test_file = test_path if test_path and test_path.strip() else None
            
            progress(0.5, desc="Executing in cloud sandbox...")
            result = apply_refactoring_safely(file_path, instruction, test_file)
            
            progress(1.0, desc="Complete!")
            
            return gr.update(visible=True, value=result)
            
        except ImportError:
            return gr.update(visible=True, value="⚠️ Modal integration not configured. Please set up Modal credentials.")
            
    except Exception as e:
        return gr.update(visible=True, value=f"❌ Execution failed: {str(e)}")

# --- CUSTOM CSS ---
custom_css = """
/* Modern Theme */
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1400px !important;
}

/* Header Styling */
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

/* Tab Styling */
.tab-nav button {
    font-weight: 600;
    font-size: 1rem;
    padding: 0.75rem 1.5rem;
}

/* Success/Error Banners */
.banner {
    padding: 1rem;
    border-radius: 8px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 1rem;
}

/* Button Styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
}

/* Card Styling */
.info-card {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

/* Code Block Enhancement */
.code-block {
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9rem;
}

/* Image Container */
.diagram-container {
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 1rem;
    background: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Sponsor Badges */
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

# --- GRADIO 6 INTERFACE ---
with gr.Blocks(
    title="ArchitectAI - Autonomous Cloud Refactoring",
    fill_height=True
) as demo:
    
    # HEADER
    with gr.Row():
        with gr.Column():
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
    
    # MAIN TABS
    with gr.Tabs() as tabs:
        
        # === TAB 1: SINGLE FILE ===
        with gr.Tab("📄 Single File Analysis", id=0):
            gr.Markdown("""
                ### Quick Code Analysis
                Paste your Python code to generate an instant UML class diagram.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    code_input = gr.Code(
                        language="python",
                        label="Python Code",
                        lines=15,
                        elem_classes=["code-block"]
                    )
                    
                    with gr.Row():
                        enrich_checkbox = gr.Checkbox(
                            label="✨ AI Type Enrichment (OpenAI)",
                            value=False,
                            info="Use AI to infer missing type hints"
                        )
                    
                    analyze_btn = gr.Button(
                        "🚀 Analyze Code",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"]
                    )
                
                with gr.Column(scale=1):
                    status_banner_1 = gr.Markdown(visible=False, elem_classes=["banner"])
                    
                    with gr.Group():
                        img_output_1 = gr.Image(
                            label="📊 Class Diagram",
                            type="pil",
                            elem_classes=["diagram-container"],
                        )
                    
                    with gr.Accordion("📝 PlantUML Source", open=False):
                        text_output_1 = gr.Code(
                            language="markdown",
                            label="PlantUML Code",
                            lines=10
                        )
            
            analyze_btn.click(
                fn=process_code_snippet,
                inputs=[code_input, enrich_checkbox],
                outputs=[text_output_1, img_output_1, status_banner_1]
            )
        
        # === TAB 2: PROJECT MAP ===
        with gr.Tab("📂 Project Architecture Map", id=1):
            gr.Markdown("""
                ### Full Project Analysis
                Upload a ZIP file of your Python project to visualize all classes and relationships.
            """)
            
            gr.HTML("""
                <div class="info-card">
                    <strong>💡 Tip:</strong> Create a ZIP of your project folder. Works best with 5-50 Python files.
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    project_zip = gr.File(
                        label="📦 Upload Project (ZIP)",
                        file_types=[".zip"],
                        type="filepath"
                    )
                    
                    scan_btn = gr.Button(
                        "🔍 Scan Project",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"]
                    )
                
                with gr.Column(scale=1):
                    status_banner_2 = gr.Markdown(visible=False, elem_classes=["banner"])
                    
                    with gr.Group():
                        img_output_2 = gr.Image(
                            label="🗺️ Project Architecture",
                            type="pil",
                            elem_classes=["diagram-container"],
                        )
                    
                    with gr.Accordion("📝 PlantUML Source", open=False):
                        text_output_2 = gr.Code(
                            language="markdown",
                            label="PlantUML Code",
                            lines=10
                        )
            
            scan_btn.click(
                fn=process_zip_upload,
                inputs=project_zip,
                outputs=[text_output_2, img_output_2, status_banner_2]
            )
       
        # === TAB 3: AI PROPOSAL ===
        with gr.Tab("✨ AI Refactoring Proposal", id=2):
            gr.Markdown("""
                ### Intelligent Architecture Recommendations
                Let AI analyze your codebase and suggest design pattern improvements.
            """)
            
            gr.HTML("""
                <div class="info-card">
                    <strong>🧠 AI-Powered:</strong> Detects anti-patterns like God Objects, 
                    suggests appropriate design patterns (Strategy, Factory, Singleton, etc.), 
                    and generates before/after UML diagrams.
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    proposal_zip = gr.File(
                        label="📦 Upload Project (ZIP)",
                        file_types=[".zip"],
                        type="filepath"
                    )
                    
                    propose_btn = gr.Button(
                        "🧠 Generate Proposal",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"]
                    )
                    
                    status_banner_3 = gr.Markdown(visible=False, elem_classes=["banner"])
                    
                    with gr.Group():
                        proposal_output = gr.Code(
                            language="json",
                            label="📋 AI Analysis & Recommendations",
                            lines=15,
                        )
                
                with gr.Column(scale=1):
                    with gr.Group():
                        img_output_3 = gr.Image(
                            label="🎨 Proposed Architecture (After Refactoring)",
                            type="pil",
                            elem_classes=["diagram-container"],
                        )
                    
                    with gr.Accordion("📝 Proposed PlantUML", open=False):
                        text_output_3 = gr.Code(
                            language="markdown",
                            label="PlantUML Code",
                            lines=10
                        )
            
            propose_btn.click(
                fn=process_proposal_zip,
                inputs=proposal_zip,
                outputs=[proposal_output, text_output_3, img_output_3, status_banner_3]
            )
        
        # === TAB 4: MODAL CLOUD EXECUTION ===
        with gr.Tab("☁️ Safe Refactoring (Modal)", id=3):
            gr.Markdown("""
                ### Production-Safe Cloud Execution
                Upload your project, select which file to refactor, and let Modal handle it safely.
            """)
            
            gr.HTML("""
                <div class="info-card">
                    <strong>🛡️ Safety Guaranteed:</strong> All changes run in isolated Modal containers. 
                    Tests are executed in the cloud. Files updated only if tests pass.
                </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📁 Upload & Configure")
                    
                    modal_zip = gr.File(
                        label="📦 Upload Project (ZIP)",
                        file_types=[".zip"],
                        type="filepath"
                    )
                    
                    file_dropdown = gr.Dropdown(
                        label="Target File",
                        choices=[],
                        info="Select .py file to refactor",
                        interactive=True
                    )
                    
                    test_dropdown = gr.Dropdown(
                        label="Test File (Optional)",
                        choices=[],
                        info="Select test file for validation",
                        interactive=True
                    )
                    
                    instruction_input = gr.Textbox(
                        label="Refactoring Instructions",
                        placeholder="Extract Strategy pattern for payment methods...",
                        info="Detailed instructions for AI",
                        lines=5
                    )
                    
                    execute_btn = gr.Button(
                        "🚀 Execute in Modal Sandbox",
                        variant="stop",
                        size="lg",
                        elem_classes=["primary-button"]
                    )
                
                with gr.Column():
                    gr.Markdown("#### 📊 Execution Results")
                    
                    modal_output = gr.Markdown(
                        label="Cloud Execution Logs",
                        value="☁️ Waiting for execution...",
                        elem_classes=["code-block"]
                    )
                    
                    gr.HTML("""
                        <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                            <strong>How it works:</strong>
                            <ol style="margin: 0.5rem 0;">
                                <li>Upload your project as ZIP</li>
                                <li>Select file from dropdown</li>
                                <li>Modal extracts & executes in cloud</li>
                                <li>Tests run automatically</li>
                                <li>Download refactored ZIP</li>
                            </ol>
                        </div>
                    """)
                    
                    download_output = gr.File(
                        label="📥 Download Refactored Project",
                        visible=False
                    )
            
            # When ZIP uploaded, populate dropdowns
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
            <p style="color: #6c757d; font-size: 0.9rem;">
                Built for <strong>Hugging Face x Anthropic MCP Hackathon</strong> 
                • Powered by Modal, OpenAI, SambaNova, Nebius
                <br>
                <a href="#" style="color: #667eea;">📺 Demo Video</a> • 
                <a href="#" style="color: #667eea;">📖 Documentation</a> • 
                <a href="#" style="color: #667eea;">💬 Social Post</a>
            </p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(
        ssr_mode=False,         
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css=custom_css
    )