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
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("my-openai-secret")],
    timeout=600,
    cpu=1.0,      
    memory=1024   
)
def safe_refactor_and_test(
    original_code: str, 
    instruction: str, 
    test_code: str = None
) -> dict:
    """
    Refactor code in cloud with testing.
    Shows progress in Modal console.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    import ast
    
    # ============ CONSOLE OUTPUT ============
    print("=" * 60)
    print("🚀 MODAL CLOUD REFACTORING SESSION")
    print("=" * 60)
    print(f"📋 Instruction: {instruction[:100]}...")
    print(f"📏 Code size: {len(original_code)} characters")
    print(f"🧪 Tests: {'Provided' if test_code else 'None (syntax check only)'}")
    print("-" * 60)

    # ============ PHASE 1: AI REFACTORING ============
    print("\n🤖 PHASE 1: AI REFACTORING")
    print("-" * 60)
    
    try:
        print("⏳ Initializing GPT-4o-mini...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

        prompt = f"""Refactor this Python code according to these instructions:

**Instructions:** {instruction}

**Original Code:**
```python
{original_code}
```

Return ONLY the refactored Python code. No markdown blocks, no explanations.
"""

        print("⏳ Sending request to OpenAI...")
        response = llm.invoke([
            SystemMessage(content="You are a Python refactoring expert. Return only valid Python code."),
            HumanMessage(content=prompt)
        ])
        
        new_code = response.content.strip()
        
        # Clean markdown if present
        if "```python" in new_code:
            new_code = new_code.split("```python")[1].split("```")[0].strip()
        elif "```" in new_code:
            new_code = new_code.split("```")[1].split("```")[0].strip()

        print(f"✅ Refactored successfully ({len(new_code)} chars)")
        print(f"📊 Code change: {len(new_code) - len(original_code):+d} characters")

    except Exception as e:
        print(f"❌ LLM FAILED: {str(e)}")
        print("=" * 60)
        return {
            "success": False,
            "error": f"LLM failed: {str(e)}",
            "new_code": None,
            "test_results": {"passed": False, "output": str(e)}
        }

    # ============ PHASE 2: VALIDATION ============
    print("\n🔍 PHASE 2: VALIDATION")
    print("-" * 60)
    
    # Step 1: Syntax check
    print("⏳ Checking syntax...")
    try:
        ast.parse(new_code)
        print("✅ Syntax valid")
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR at line {e.lineno}: {e.msg}")
        print("=" * 60)
        return {
            "success": False,
            "error": f"Syntax error: {e}",
            "new_code": new_code,
            "test_results": {"passed": False, "output": f"Line {e.lineno}: {e.msg}"}
        }
    
    # Step 2: Run tests (if provided)
    if not test_code:
        print("ℹ️  No tests provided - skipping test execution")
        print("=" * 60)
        print("✅ REFACTORING COMPLETE (Syntax OK)")
        print("=" * 60)
        return {
            "success": True,
            "error": None,
            "new_code": new_code,
            "test_results": {
                "passed": True, 
                "output": "✅ Syntax check passed (no tests provided)"
            }
        }
    
    # Run pytest
    print("\n🧪 PHASE 3: TESTING")
    print("-" * 60)
    print(f"⏳ Running pytest on {len(test_code)} chars of test code...")
    
    with tempfile.TemporaryDirectory() as tmp:
        # Write refactored code
        code_file = os.path.join(tmp, "refactored.py")
        with open(code_file, "w") as f:
            f.write(new_code)
        
        # Write tests
        test_file = os.path.join(tmp, "test_refactored.py")
        
        # Fix imports in test
        fixed_test = test_code.replace(
            "from services.", "from refactored."
        ).replace(
            "import target_module", "import refactored as target_module"
        )
        
        with open(test_file, "w") as f:
            f.write(fixed_test)
        
        # Run pytest with verbose output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=tmp,
            env={**os.environ, "PYTHONPATH": tmp}
        )
        
        passed = result.returncode == 0
        
        # Print test results to console
        print("\n📊 TEST RESULTS:")
        print("-" * 60)
        if passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ TESTS FAILED")
        
        print("\nTest Output:")
        print(result.stdout)
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
        
        print("=" * 60)
        if passed:
            print("🎉 REFACTORING SUCCESSFUL - ALL CHECKS PASSED")
        else:
            print("⚠️  REFACTORING COMPLETE BUT TESTS FAILED")
        print("=" * 60)
        
        return {
            "success": True,
            "error": None,
            "new_code": new_code,
            "test_results": {
                "passed": passed,
                "output": result.stdout + result.stderr
            }
        }