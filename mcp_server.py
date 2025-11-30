"""
ArchitectAI MCP Server - Model Context Protocol Implementation

This server exposes ArchitectAI's design pattern analysis capabilities
as MCP tools that can be used by Claude and other AI assistants.
"""

import json
import sys
import tempfile
import zipfile
import logging
from typing import Any
from pathlib import Path
import ast

from mcp.server import Server
from mcp.types import Tool, TextContent, ToolResult

# Import ArchitectAI services
from services.pattern_detector import PatternDetectionService, PatternRecommendation
from services.architecture_service import ArchitectureVisitor, DeterministicPlantUMLConverter
from services.project_service import ProjectAnalyzer
from services.refactoring_service import RefactoringAdvisor
from core.llm_factory import create_nebius_llm, create_openai_llm, create_sambanova_llm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP Server
server = Server("architectai-mcp")

# Tool definitions
TOOLS = [
    {
        "name": "analyze_patterns",
        "description": "Analyze Python code for design patterns and get AI-powered recommendations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to analyze"
                },
                "enrich": {
                    "type": "boolean",
                    "description": "Use LLM for enhanced analysis with AI justifications",
                    "default": False
                },
                "provider": {
                    "type": "string",
                    "enum": ["nebius", "openai", "sambanova"],
                    "description": "LLM provider to use for enrichment",
                    "default": "nebius"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "analyze_project",
        "description": "Analyze entire project structure from uploaded ZIP file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "zip_path": {
                    "type": "string",
                    "description": "Path to ZIP file containing the project"
                }
            },
            "required": ["zip_path"]
        }
    },
    {
        "name": "detect_patterns",
        "description": "Detect design patterns in a project and return detailed analysis",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to analyze for patterns"
                },
                "structure": {
                    "type": "string",
                    "description": "JSON string of project structure (optional)"
                },
                "enrich": {
                    "type": "boolean",
                    "description": "Use LLM for enhanced pattern analysis",
                    "default": False
                },
                "provider": {
                    "type": "string",
                    "enum": ["nebius", "openai", "sambanova"],
                    "description": "LLM provider for enrichment",
                    "default": "nebius"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "get_refactoring_proposal",
        "description": "Get AI-powered refactoring recommendations for Python code",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to refactor"
                },
                "instruction": {
                    "type": "string",
                    "description": "What to refactor or improve"
                },
                "provider": {
                    "type": "string",
                    "enum": ["nebius", "openai", "sambanova"],
                    "description": "LLM provider for refactoring suggestions",
                    "default": "nebius"
                }
            },
            "required": ["code", "instruction"]
        }
    },
    {
        "name": "generate_uml",
        "description": "Generate PlantUML diagrams for Python code structure",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to diagram"
                },
                "include_methods": {
                    "type": "boolean",
                    "description": "Include method details in diagram",
                    "default": True
                }
            },
            "required": ["code"]
        }
    }
]


def get_llm_client(provider: str = "nebius"):
    """Get LLM client for specified provider"""
    if provider == "openai":
        return create_openai_llm()
    elif provider == "sambanova":
        return create_sambanova_llm()
    else:  # nebius
        return create_nebius_llm()


@server.call_tool()
async def analyze_patterns(code: str, enrich: bool = False, provider: str = "nebius") -> ToolResult:
    """
    Analyze Python code for design patterns.
    
    Returns detailed analysis of detected patterns and recommendations.
    """
    try:
        logger.info(f"🔍 Analyzing patterns in code ({len(code)} chars), enrich={enrich}")
        
        # Get LLM if enrichment requested
        llm = None
        if enrich:
            try:
                llm = get_llm_client(provider)
                logger.info(f"✓ LLM client initialized: {provider}")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}, continuing without enrichment")
        
        # Run pattern detection
        service = PatternDetectionService(llm=llm)
        result = service.analyze_code(code, enrich=enrich)
        
        # Format report
        report = service.format_report(result)
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "report": report,
                    "patterns_detected": result['summary']['total_patterns'],
                    "recommendations": result['summary']['total_recommendations'],
                    "pattern_types": result['summary'].get('patterns_found', []),
                    "detections": result.get('detections', []),
                    "recommendations_list": result.get('recommendations', [])
                }, indent=2, default=str)
            )],
            is_error=False
        )
    except Exception as e:
        logger.error(f"Pattern analysis error: {e}", exc_info=True)
        return ToolResult(
            content=[TextContent(type="text", text=f"❌ Error: {str(e)}")],
            is_error=True
        )


@server.call_tool()
async def analyze_project(zip_path: str) -> ToolResult:
    """
    Analyze entire project structure from ZIP file.
    
    Generates architecture diagrams and identifies patterns across modules.
    """
    try:
        logger.info(f"📁 Analyzing project from: {zip_path}")
        
        zip_file = Path(zip_path)
        if not zip_file.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
        temp_dir = tempfile.mkdtemp()
        
        # Extract ZIP
        logger.info(f"📦 Extracting ZIP to {temp_dir}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)
        
        # Analyze project
        logger.info("🔍 Analyzing project structure...")
        analyzer = ProjectAnalyzer(Path(temp_dir))
        structure = analyzer.analyze()
        
        if not structure:
            raise ValueError("No Python code found in project")
        
        logger.info(f"✓ Found {len(structure)} components")
        
        # Generate UML
        logger.info("🎨 Generating PlantUML diagram...")
        converter = DeterministicPlantUMLConverter()
        puml = converter.convert(structure)
        
        # Collect code for pattern detection
        all_code = []
        for file_path in Path(temp_dir).rglob("*.py"):
            parts = file_path.parts
            if any(p.startswith(".") or p in ["venv", "env", "__pycache__", "node_modules"] for p in parts):
                continue
            try:
                code = file_path.read_text(encoding='utf-8', errors='replace')
                rel_path = file_path.relative_to(temp_dir)
                all_code.append(f"# === File: {rel_path} ===\n{code}")
            except:
                continue
        
        combined_code = "\n\n".join(all_code)
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "components_found": len(structure),
                    "files_analyzed": len(all_code),
                    "puml_diagram": puml,
                    "structure": structure[:10]  # Return first 10 components
                }, indent=2, default=str)
            )],
            is_error=False
        )
    except Exception as e:
        logger.error(f"Project analysis error: {e}", exc_info=True)
        return ToolResult(
            content=[TextContent(type="text", text=f"❌ Error: {str(e)}")],
            is_error=True
        )


@server.call_tool()
async def detect_patterns(code: str, structure: str = None, enrich: bool = False, provider: str = "nebius") -> ToolResult:
    """
    Detect design patterns in code with optional structure context.
    """
    try:
        logger.info(f"🏛️ Detecting patterns with enrich={enrich}")
        
        # Parse structure if provided
        project_structure = []
        if structure:
            try:
                project_structure = json.loads(structure)
            except:
                logger.warning("Could not parse structure JSON")
        
        # Get LLM if needed
        llm = None
        if enrich:
            llm = get_llm_client(provider)
        
        # Run detection
        service = PatternDetectionService(llm=llm)
        result = service.analyze_code(code, enrich=enrich)
        
        # Generate recommendations UML if patterns found
        uml_diagrams = {}
        if result['recommendations'] and project_structure:
            logger.info("🎨 Generating UML for recommendations...")
            recommender = service.recommender
            for i, rec_dict in enumerate(result['recommendations'][:3]):  # Limit to 3
                try:
                    rec = PatternRecommendation(**rec_dict)
                    before, after = recommender.generate_recommendation_uml(rec, project_structure, code)
                    uml_diagrams[f"recommendation_{i}"] = {
                        "pattern": rec.pattern,
                        "before": before,
                        "after": after
                    }
                except Exception as e:
                    logger.warning(f"Could not generate UML for recommendation {i}: {e}")
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "patterns_found": result['summary']['total_patterns'],
                    "recommendations": result['summary']['total_recommendations'],
                    "detections": result['detections'],
                    "recommendations_list": result['recommendations'],
                    "uml_diagrams": uml_diagrams
                }, indent=2, default=str)
            )],
            is_error=False
        )
    except Exception as e:
        logger.error(f"Pattern detection error: {e}", exc_info=True)
        return ToolResult(
            content=[TextContent(type="text", text=f"❌ Error: {str(e)}")],
            is_error=True
        )


@server.call_tool()
async def get_refactoring_proposal(code: str, instruction: str, provider: str = "nebius") -> ToolResult:
    """
    Get AI-powered refactoring recommendations.
    """
    try:
        logger.info(f"🔧 Getting refactoring proposal: {instruction}")
        
        # Parse code structure
        try:
            tree = ast.parse(code)
            visitor = ArchitectureVisitor()
            visitor.visit(tree)
            structure = visitor.structure
        except SyntaxError as e:
            return ToolResult(
                content=[TextContent(type="text", text=f"❌ Syntax Error: {e}")],
                is_error=True
            )
        
        # Get LLM
        llm = get_llm_client(provider)
        
        # Get proposal
        advisor = RefactoringAdvisor(llm=llm)
        proposal = advisor.propose_improvement(structure, instruction)
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "proposal": proposal,
                    "instruction": instruction
                }, indent=2, default=str)
            )],
            is_error=False
        )
    except Exception as e:
        logger.error(f"Refactoring error: {e}", exc_info=True)
        return ToolResult(
            content=[TextContent(type="text", text=f"❌ Error: {str(e)}")],
            is_error=True
        )


@server.call_tool()
async def generate_uml(code: str, include_methods: bool = True) -> ToolResult:
    """
    Generate PlantUML diagram for Python code structure.
    """
    try:
        logger.info("🎨 Generating UML diagram...")
        
        # Parse code
        try:
            tree = ast.parse(code)
            visitor = ArchitectureVisitor()
            visitor.visit(tree)
            structure = visitor.structure
        except SyntaxError as e:
            return ToolResult(
                content=[TextContent(type="text", text=f"❌ Syntax Error: {e}")],
                is_error=True
            )
        
        if not structure:
            raise ValueError("No classes or functions found in code")
        
        # Generate UML
        converter = DeterministicPlantUMLConverter()
        puml = converter.convert(structure)
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "puml": puml,
                    "components": len(structure),
                    "diagram_type": "class_diagram"
                }, indent=2, default=str)
            )],
            is_error=False
        )
    except Exception as e:
        logger.error(f"UML generation error: {e}", exc_info=True)
        return ToolResult(
            content=[TextContent(type="text", text=f"❌ Error: {str(e)}")],
            is_error=True
        )


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools"""
    return [
        Tool(
            name=tool["name"],
            description=tool["description"],
            inputSchema=tool["inputSchema"]
        )
        for tool in TOOLS
    ]


async def main():
    """Run the MCP server"""
    logger.info("🚀 Starting ArchitectAI MCP Server")
    logger.info("Available tools:")
    for tool in TOOLS:
        logger.info(f"  - {tool['name']}: {tool['description']}")
    
    await server.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
