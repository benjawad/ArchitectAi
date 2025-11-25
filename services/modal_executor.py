import modal
import os
import sys
import subprocess
import tempfile

# --- 1. Define the Cloud Environment ---
image = (
    modal.Image.debian_slim()
    .pip_install(
        "langchain",
        "langchain-openai",
        "openai",
        "pytest",
        "tiktoken"
    )
)

# Initialize the Modal App (Renamed from 'stub' to 'app')
app = modal.App("architect-ai-surgeon")

# --- 2. Define the Remote Refactoring Function ---
@app.function(  # Changed decorator from @stub to @app
    image=image,
    secrets=[modal.Secret.from_name("my-openai-secret")],
    timeout=600,
    cpu=1.0,      
    memory=1024   
)
def safe_refactor_and_test(original_code: str, instruction: str, test_code: str = None) -> dict:
    """
    Run refactoring in the cloud.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI

    print(f"🔧 Starting Cloud Refactoring task...")

    # --- PHASE 1: The Surgery (AI Refactoring) ---
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        system_prompt = (
            "You are a Senior Python Refactoring Engineer. "
            "Your goal is to rewrite the provided code to meet the user's architectural instructions "
            "while preserving the original business logic and passing all tests."
        )

        user_prompt = f"""
        **Task:** Refactor this Python code.
        
        **Instructions:** {instruction}
        
        **Constraints:**
        1. Return ONLY the full valid Python code.
        2. Do NOT use Markdown blocks (```python).
        3. Do NOT add explanations or chat.
        4. Preserve imports unless they need to change for the new architecture.
        
        **Original Code:**
        {original_code}
        """

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        new_code = response.content
        # Clean up markdown if present
        if new_code.startswith("```python"):
            new_code = new_code.replace("```python", "", 1)
        if new_code.startswith("```"):
            new_code = new_code.replace("```", "", 1)
        if new_code.endswith("```"):
            new_code = new_code[:-3]
        
        new_code = new_code.strip()

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM Generation Failed: {str(e)}",
            "new_code": None,
            "test_results": {"passed": False, "output": "LLM Error"}
        }

    # --- PHASE 2: The Checkup (Verification & Testing) ---
    if not test_code:
        try:
            print("🔍 No tests provided. Running Syntax Check...")
            compile(new_code, 'refactored_module.py', 'exec')
            test_results = {
                "passed": True, 
                "output": "✅ Syntax Check Passed. (No unit tests were provided for deep verification)."
            }
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Generated code has Syntax Errors: {e}",
                "new_code": new_code,
                "test_results": {"passed": False, "output": str(e)}
            }

    else:
        print("🧪 Running provided Unit Tests...")
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = os.path.join(temp_dir, "target_module.py")
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(new_code)
            
            test_path = os.path.join(temp_dir, "test_suite.py")
            
            # Adjust imports for the test environment
            # This handles imports like 'from services import...' 
            adjusted_test_code = test_code.replace("from services import", "# from services import") \
                                          .replace("import target_module", "") 
            
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(test_code)

            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir

            result = subprocess.run(
                ["pytest", test_path],
                capture_output=True,
                text=True,
                env=env,
                cwd=temp_dir
            )

            test_results = {
                "passed": result.returncode == 0,
                "output": result.stdout + "\n" + result.stderr
            }

    return {
        "success": True,
        "error": None,
        "new_code": new_code,
        "test_results": test_results
    }