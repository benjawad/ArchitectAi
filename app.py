import gradio as gr
import ast
import logging
import io
import json
import sys
from pathlib import Path
from PIL import Image
from plantuml import PlantUML

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
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css=custom_css,
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
                            show_download_button=True
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
                Scan an entire codebase and visualize all classes, relationships, and dependencies.
            """)
            
            gr.HTML("""
                <div class="info-card">
                    <strong>💡 Tip:</strong> Works best with projects containing 5-50 Python files.
                    Large projects may take 30-60 seconds to analyze.
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    folder_input = gr.Textbox(
                        label="Project Path",
                        info="Absolute path to your Python project directory",
                        lines=1
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
                            show_download_button=True
                        )
                    
                    with gr.Accordion("📝 PlantUML Source", open=False):
                        text_output_2 = gr.Code(
                            language="markdown",
                            label="PlantUML Code",
                            lines=10
                        )
            
            scan_btn.click(
                fn=process_folder,
                inputs=folder_input,
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
                    proposal_input = gr.Textbox(
                        label="Project Path",
                        placeholder="C:\\Path\\To\\Your\\Project",
                        lines=1
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
                            show_download_button=True
                        )
                    
                    with gr.Accordion("📝 Proposed PlantUML", open=False):
                        text_output_3 = gr.Code(
                            language="markdown",
                            label="PlantUML Code",
                            lines=10
                        )
            
            propose_btn.click(
                fn=process_proposal,
                inputs=proposal_input,
                outputs=[proposal_output, text_output_3, img_output_3, status_banner_3]
            )
        
        # === TAB 4: MODAL CLOUD EXECUTION ===
        with gr.Tab("☁️ Safe Refactoring (Modal)", id=3):
            gr.Markdown("""
                ### Production-Safe Cloud Execution
                Refactor code in isolated Modal sandboxes with automatic testing.
            """)
            
            gr.HTML("""
                <div class="info-card">
                    <strong>🛡️ Safety Guaranteed:</strong> All changes run in isolated cloud containers. 
                    Your local files are updated ONLY if tests pass. Zero risk to production code.
                </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📁 Target Configuration")
                    
                    file_input = gr.Textbox(
                        label="Target File (Relative Path)",
                        placeholder="services/payment_processor.py",
                        info="File to refactor, relative to project root",
                        lines=1
                    )
                    
                    test_input = gr.Textbox(
                        label="Test File (Optional)",
                        placeholder="tests/test_payment_processor.py",
                        info="Tests to run for validation",
                        lines=1
                    )
                    
                    instruction_input = gr.Textbox(
                        label="Refactoring Instructions",
                        placeholder="Extract Strategy pattern for payment methods. Create PaymentStrategy interface and separate classes for CreditCard, PayPal, Bitcoin...",
                        info="Detailed instructions for the AI",
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
                        value="Waiting for execution...",
                        elem_classes=["code-block"]
                    )
                    
                    gr.HTML("""
                        <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                            <strong>How it works:</strong>
                            <ol style="margin: 0.5rem 0;">
                                <li>Code is sent to Modal cloud</li>
                                <li>Executed in isolated container</li>
                                <li>Tests run automatically</li>
                                <li>Results returned safely</li>
                            </ol>
                        </div>
                    """)
            
            execute_btn.click(
                fn=run_modal_refactoring,
                inputs=[file_input, instruction_input, test_input],
                outputs=modal_output
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
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )