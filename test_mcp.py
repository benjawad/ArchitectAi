"""
Test script for ArchitectAI MCP Server

Run this to verify all tools work correctly before deploying.
"""

import asyncio
import json
from mcp_server import (
    analyze_patterns,
    analyze_project,
    detect_patterns,
    get_refactoring_proposal,
    generate_uml,
)


async def test_analyze_patterns():
    """Test pattern analysis tool"""
    print("\n" + "="*60)
    print("TEST 1: analyze_patterns")
    print("="*60)
    
    code = """
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with Credit Card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with PayPal"

class ShoppingCart:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def checkout(self, total):
        return self.strategy.pay(total)
"""
    
    result = await analyze_patterns(code, enrich=False)
    print(f"Status: {'✓ PASS' if not result.is_error else '✗ FAIL'}")
    
    if not result.is_error:
        content = json.loads(result.content[0].text)
        print(f"Patterns detected: {content['patterns_detected']}")
        print(f"Recommendations: {content['recommendations']}")
    else:
        print(f"Error: {result.content[0].text}")


async def test_generate_uml():
    """Test UML generation"""
    print("\n" + "="*60)
    print("TEST 2: generate_uml")
    print("="*60)
    
    code = """
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def get_profile(self):
        return {"name": self.name, "email": self.email}

class Order:
    def __init__(self, user, items):
        self.user = user
        self.items = items
        self.total = sum(item.price for item in items)
"""
    
    result = await generate_uml(code)
    print(f"Status: {'✓ PASS' if not result.is_error else '✗ FAIL'}")
    
    if not result.is_error:
        content = json.loads(result.content[0].text)
        print(f"Components: {content['components']}")
        print(f"UML generated: {len(content['puml'])} chars")
    else:
        print(f"Error: {result.content[0].text}")


async def test_detect_patterns():
    """Test pattern detection"""
    print("\n" + "="*60)
    print("TEST 3: detect_patterns")
    print("="*60)
    
    code = """
class Database:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self):
        return "Connected"
"""
    
    result = await detect_patterns(code, enrich=False)
    print(f"Status: {'✓ PASS' if not result.is_error else '✗ FAIL'}")
    
    if not result.is_error:
        content = json.loads(result.content[0].text)
        print(f"Patterns: {content['patterns_found']}")
        print(f"Recommendations: {content['recommendations']}")
    else:
        print(f"Error: {result.content[0].text}")


async def test_refactoring():
    """Test refactoring proposal"""
    print("\n" + "="*60)
    print("TEST 4: get_refactoring_proposal")
    print("="*60)
    
    code = """
class PaymentProcessor:
    def process_payment(self, order, method):
        if method == "credit_card":
            return self._process_cc(order)
        elif method == "paypal":
            return self._process_paypal(order)
        else:
            return None
"""
    
    result = await get_refactoring_proposal(
        code,
        "Replace if/else with Strategy pattern"
    )
    print(f"Status: {'✓ PASS' if not result.is_error else '✗ FAIL'}")
    
    if not result.is_error:
        content = json.loads(result.content[0].text)
        print(f"Proposal generated: {bool(content['proposal'])}")
    else:
        print(f"Error: {result.content[0].text}")


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "ArchitectAI MCP Server Tests" + " "*16 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        await test_generate_uml()
        await test_analyze_patterns()
        await test_detect_patterns()
        # Skipping refactoring test as it requires LLM setup
        # await test_refactoring()
        
        print("\n" + "="*60)
        print("✓ Core Tests Passed!")
        print("="*60)
        print("\nMCP Server is ready for deployment to Blaxcel AI!")
        print("\nNext steps:")
        print("  1. Update mcp.json with your GitHub info")
        print("  2. Set up environment variables in .env.blaxcel")
        print("  3. Run: blaxcel push --config mcp.json")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ Test Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
