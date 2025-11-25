import json
import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm_factory import create_sambanova_llm ,create_openai_llm

class RefactoringAdvisor:
    """
    Analyzes project structure and proposes architectural changes (Design Patterns).
    """
    
    def __init__(self, llm=None):
        self.llm = llm or create_openai_llm(temperature=0.0)

    def propose_improvement(self, structure_data: list[dict]) -> dict:
        """
        Analyzes the code and generates a proposal with BEFORE and AFTER UML.
        Returns:
            {
                "title": "Strategy Pattern for Payment Processing",
                "description": "Currently, the Payment class uses big if-else...",
                "proposed_uml": "@startuml ... @enduml",
                "affected_files": ["payment.py", "order.py"]
            }
        """
        summary = self._summarize_for_llm(structure_data)
        
        logging.info("🧠 Brainstorming architectural improvements...")
        
        prompt = f"""
        Act as a Principal Software Architect. 
        Analyze this Python project structure summary to find ONE critical design flaw.
        
        Current Structure:
        {json.dumps(summary, indent=2)}
        
        Your Goal:
        1. Identify the weak spot (High Coupling, Low Cohesion, Violation of SOLID).
        2. Select the BEST Design Pattern to fix it (Factory, Strategy, Observer, Adapter, etc.).
        3. Redesign the affected classes using this pattern.
        
        Output strictly in JSON format:
        {{
            "title": "Short title of the refactoring",
            "description": "Explanation of the problem and why this pattern solves it",
            "affected_classes": ["ClassA", "ClassB"],
            "proposed_uml": "The FULL PlantUML code representing the NEW structure for these classes (start with @startuml)"
        }}
        """
        
        try:
            messages = [
                SystemMessage(content="You are a JSON-only architect assistant."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = self._clean_json_output(response.content)
            
            return json.loads(content)
            
        except Exception as e:
            logging.error(f"Refactoring proposal failed: {e}")
            return {"error": str(e)}

    def _summarize_for_llm(self, structure: list[dict]):
        """Simplifies the structure to save tokens."""
        summary = []
        for item in structure:
            if item.get("type") == "module": continue
            summary.append({
                "class": item["name"],
                "inherits": item["bases"],
                "methods": [m["name"] for m in item["methods"]],
                "attributes": [a["type"] for a in item["attributes"]]
            })
        return summary

    def _clean_json_output(self, content: str) -> str:
        """Fixes common LLM JSON formatting issues."""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return content.strip()