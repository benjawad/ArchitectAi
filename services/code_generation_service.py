import os
import logging
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from core.llm_factory import create_openai_llm
class CodeGenerator:
    """
    The Surgeon: Writes the new code directly to the specified path.
    """
    def __init__(self, llm=None):
        self.llm = llm or create_openai_llm(temperature=0.0)

    def generate_refactored_code(self, original_code: str, refactoring_plan: str, file_path: str) -> str:
        """
        Writes the new code.
        """
        logging.info(f"🔨 Generating code for: {file_path}")
        
        prompt = f"""
        Act as a Senior Python Refactoring Engineer.
        
        Task: Rewrite the code for file '{file_path}' fully, implementing the requested changes.
        
        --- REFACTORING INSTRUCTIONS ---
        {refactoring_plan}
        
        --- ORIGINAL SOURCE CODE ---
        {original_code}
        
        Requirements:
        1. **Integrity:** Keep imports and logic that are NOT affected by the refactoring.
        2. **Implementation:** Apply the design pattern strictly.
        3. **Output:** Return ONLY the Python code (no markdown, no explanations).
        """
        
        messages = [
            SystemMessage(content="You are a precise code generator. Output code only."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return self._clean_output(response.content)

    def save_code(self, relative_path: str, code: str, root_path: Path) -> str:
        """
        Saves the code directly to the file (Overwriting it).
        WARNING: Only use this on a PROJECT COPY.
        """
        target_path = root_path / relative_path
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        target_path.write_text(code, encoding="utf-8")
        
        logging.info(f"💾 File updated: {target_path}")
        return str(target_path)

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```python"):
            text = text.replace("```python", "", 1)
        if text.startswith("```"):
            text = text.replace("```", "", 1)
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()