"""
Create demo projects for ArchitectAI hackathon
Speed is critical - these demos should work instantly!
"""

import os
import zipfile
from pathlib import Path


def create_demo_ecommerce():
    """Create a realistic e-commerce demo with 15 files"""
    
    demo_dir = Path("demo_ecommerce")
    demo_dir.mkdir(exist_ok=True)
    
    # 1. User model
    (demo_dir / "models" / "user.py").parent.mkdir(exist_ok=True)
    (demo_dir / "models" / "user.py").write_text("""
class User:
    def __init__(self, user_id, name, email):
        self.user_id = user_id
        self.name = name
        self.email = email
        self.orders = []
    
    def add_order(self, order):
        self.orders.append(order)
    
    def get_total_spent(self):
        return sum(order.total for order in self.orders)
""")
    
    # 2. Product model
    (demo_dir / "models" / "product.py").write_text("""
class Product:
    def __init__(self, product_id, name, price, stock):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.stock = stock
    
    def is_available(self):
        return self.stock > 0
    
    def reduce_stock(self, quantity):
        if self.stock >= quantity:
            self.stock -= quantity
            return True
        return False
""")
    
    # 3. Order model
    (demo_dir / "models" / "order.py").write_text("""
from datetime import datetime

class Order:
    def __init__(self, order_id, user_id):
        self.order_id = order_id
        self.user_id = user_id
        self.items = []
        self.status = "pending"
        self.created_at = datetime.now()
        self.total = 0
    
    def add_item(self, product, quantity):
        self.items.append({
            "product": product,
            "quantity": quantity,
            "price": product.price
        })
        self.total += product.price * quantity
    
    def process(self):
        self.status = "processing"
    
    def complete(self):
        self.status = "completed"
""")
    
    # 4. Payment processor with STRATEGY PATTERN opportunity!
    (demo_dir / "services" / "payment.py").parent.mkdir(exist_ok=True)
    (demo_dir / "services" / "payment.py").write_text("""
# BAD: Using if/else for different payment methods
# Should use Strategy pattern!

class PaymentProcessor:
    def process_payment(self, order, payment_method):
        if payment_method == "credit_card":
            return self._process_credit_card(order)
        elif payment_method == "paypal":
            return self._process_paypal(order)
        elif payment_method == "bitcoin":
            return self._process_bitcoin(order)
        elif payment_method == "bank_transfer":
            return self._process_bank_transfer(order)
        else:
            raise ValueError("Unknown payment method")
    
    def _process_credit_card(self, order):
        # Credit card processing logic
        print(f"Processing ${order.total} via Credit Card")
        return True
    
    def _process_paypal(self, order):
        # PayPal processing logic
        print(f"Processing ${order.total} via PayPal")
        return True
    
    def _process_bitcoin(self, order):
        # Bitcoin processing logic
        print(f"Processing ${order.total} via Bitcoin")
        return True
    
    def _process_bank_transfer(self, order):
        # Bank transfer processing logic
        print(f"Processing ${order.total} via Bank Transfer")
        return True
""")
    
    # 5. Inventory manager - SINGLETON opportunity!
    (demo_dir / "services" / "inventory.py").write_text("""
# BAD: Multiple instances possible
# Should use Singleton pattern!

class InventoryManager:
    def __init__(self):
        self.products = {}
    
    def add_product(self, product):
        self.products[product.product_id] = product
    
    def get_product(self, product_id):
        return self.products.get(product_id)
    
    def update_stock(self, product_id, quantity):
        product = self.products.get(product_id)
        if product:
            product.stock += quantity
""")
    
    # 6. Notification service - OBSERVER opportunity!
    (demo_dir / "services" / "notification.py").write_text("""
# BAD: Tight coupling for notifications
# Should use Observer pattern!

class NotificationService:
    def __init__(self):
        self.email_service = EmailService()
        self.sms_service = SMSService()
        self.push_service = PushService()
    
    def notify_order_created(self, order):
        self.email_service.send_email(order.user_id, "Order created")
        self.sms_service.send_sms(order.user_id, "Order created")
        self.push_service.send_push(order.user_id, "Order created")
    
    def notify_order_shipped(self, order):
        self.email_service.send_email(order.user_id, "Order shipped")
        self.sms_service.send_sms(order.user_id, "Order shipped")

class EmailService:
    def send_email(self, user_id, message):
        print(f"Email to {user_id}: {message}")

class SMSService:
    def send_sms(self, user_id, message):
        print(f"SMS to {user_id}: {message}")

class PushService:
    def send_push(self, user_id, message):
        print(f"Push to {user_id}: {message}")
""")
    
    # 7. Product factory - showing good FACTORY pattern
    (demo_dir / "factories" / "product_factory.py").parent.mkdir(exist_ok=True)
    (demo_dir / "factories" / "product_factory.py").write_text("""
# GOOD: Factory pattern for product creation

from models.product import Product

class ProductFactory:
    @staticmethod
    def create_electronic(name, price):
        return Product(
            product_id=f"ELEC-{name}",
            name=name,
            price=price,
            stock=10
        )
    
    @staticmethod
    def create_clothing(name, price):
        return Product(
            product_id=f"CLOTH-{name}",
            name=name,
            price=price,
            stock=50
        )
    
    @staticmethod
    def create_book(name, price):
        return Product(
            product_id=f"BOOK-{name}",
            name=name,
            price=price,
            stock=100
        )
""")
    
    # 8. Order builder
    (demo_dir / "builders" / "order_builder.py").parent.mkdir(exist_ok=True)
    (demo_dir / "builders" / "order_builder.py").write_text("""
# GOOD: Builder pattern for complex order creation

from models.order import Order

class OrderBuilder:
    def __init__(self, user_id):
        self.order = Order(order_id=None, user_id=user_id)
    
    def add_product(self, product, quantity):
        self.order.add_item(product, quantity)
        return self
    
    def set_shipping_address(self, address):
        self.order.shipping_address = address
        return self
    
    def set_billing_address(self, address):
        self.order.billing_address = address
        return self
    
    def build(self):
        return self.order
""")
    
    # 9. Shopping cart
    (demo_dir / "cart.py").write_text("""
class ShoppingCart:
    def __init__(self, user_id):
        self.user_id = user_id
        self.items = []
    
    def add_item(self, product, quantity):
        self.items.append({"product": product, "quantity": quantity})
    
    def remove_item(self, product_id):
        self.items = [item for item in self.items if item["product"].product_id != product_id]
    
    def get_total(self):
        return sum(item["product"].price * item["quantity"] for item in self.items)
    
    def clear(self):
        self.items = []
""")
    
    # 10. Discount calculator
    (demo_dir / "discount.py").write_text("""
class DiscountCalculator:
    def apply_percentage_discount(self, total, percentage):
        return total * (1 - percentage / 100)
    
    def apply_fixed_discount(self, total, amount):
        return max(0, total - amount)
    
    def apply_bulk_discount(self, quantity, price):
        if quantity >= 10:
            return price * 0.9
        return price
""")
    
    # 11-15. More utility files
    (demo_dir / "utils" / "validator.py").parent.mkdir(exist_ok=True)
    (demo_dir / "utils" / "validator.py").write_text("""
import re

class Validator:
    @staticmethod
    def validate_email(email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_phone(phone):
        return len(phone) >= 10
""")
    
    (demo_dir / "utils" / "logger.py").write_text("""
from datetime import datetime

class Logger:
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
""")
    
    (demo_dir / "config.py").write_text("""
DATABASE_URL = "postgresql://localhost/ecommerce"
REDIS_URL = "redis://localhost:6379"
SECRET_KEY = "your-secret-key"
DEBUG = True
""")
    
    (demo_dir / "main.py").write_text("""
from models.user import User
from models.product import Product
from services.payment import PaymentProcessor
from cart import ShoppingCart

def main():
    # Create user
    user = User(1, "John Doe", "john@example.com")
    
    # Create products
    laptop = Product(1, "Laptop", 999.99, 5)
    mouse = Product(2, "Mouse", 29.99, 50)
    
    # Shopping
    cart = ShoppingCart(user.user_id)
    cart.add_item(laptop, 1)
    cart.add_item(mouse, 2)
    
    print(f"Total: ${cart.get_total()}")

if __name__ == "__main__":
    main()
""")
    
    (demo_dir / "__init__.py").write_text("")
    (demo_dir / "models" / "__init__.py").write_text("")
    (demo_dir / "services" / "__init__.py").write_text("")
    (demo_dir / "factories" / "__init__.py").write_text("")
    (demo_dir / "builders" / "__init__.py").write_text("")
    (demo_dir / "utils" / "__init__.py").write_text("")
    
    # Create ZIP
    with zipfile.ZipFile("demo_ecommerce.zip", "w") as zf:
        for root, dirs, files in os.walk(demo_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, demo_dir.parent)
                zf.write(file_path, arcname)
    
    print("✅ Created demo_ecommerce.zip (15 files)")
    
    # Cleanup
    import shutil
    shutil.rmtree(demo_dir)


def create_demo_fastapi():
    """Create a fast FastAPI demo with 8 files"""
    
    demo_dir = Path("demo_fastapi")
    demo_dir.mkdir(exist_ok=True)
    
    # 1. Main app
    (demo_dir / "main.py").write_text("""
from fastapi import FastAPI, HTTPException
from models import User, Product, Order
from database import Database

app = FastAPI()
db = Database()

@app.get("/")
def read_root():
    return {"message": "FastAPI E-commerce API"}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users")
def create_user(user: User):
    return db.create_user(user)

@app.get("/products")
def list_products():
    return db.get_all_products()

@app.post("/orders")
def create_order(order: Order):
    return db.create_order(order)
""")
    
    # 2. Models
    (demo_dir / "models.py").write_text("""
from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    user_id: Optional[int] = None
    name: str
    email: str

class Product(BaseModel):
    product_id: Optional[int] = None
    name: str
    price: float
    stock: int

class OrderItem(BaseModel):
    product_id: int
    quantity: int

class Order(BaseModel):
    order_id: Optional[int] = None
    user_id: int
    items: List[OrderItem]
    total: float = 0.0
""")
    
    # 3. Database (Singleton opportunity!)
    (demo_dir / "database.py").write_text("""
# BAD: Should be Singleton!

class Database:
    def __init__(self):
        self.users = {}
        self.products = {}
        self.orders = {}
        self.next_user_id = 1
        self.next_product_id = 1
        self.next_order_id = 1
    
    def get_user(self, user_id):
        return self.users.get(user_id)
    
    def create_user(self, user):
        user.user_id = self.next_user_id
        self.users[user.user_id] = user
        self.next_user_id += 1
        return user
    
    def get_all_products(self):
        return list(self.products.values())
    
    def create_order(self, order):
        order.order_id = self.next_order_id
        self.orders[order.order_id] = order
        self.next_order_id += 1
        return order
""")
    
    # 4. Authentication (Strategy opportunity!)
    (demo_dir / "auth.py").write_text("""
# BAD: Multiple if/else for auth methods
# Should use Strategy pattern!

class AuthService:
    def authenticate(self, credentials, method):
        if method == "jwt":
            return self._verify_jwt(credentials)
        elif method == "oauth":
            return self._verify_oauth(credentials)
        elif method == "api_key":
            return self._verify_api_key(credentials)
        else:
            return False
    
    def _verify_jwt(self, token):
        print(f"Verifying JWT: {token}")
        return True
    
    def _verify_oauth(self, token):
        print(f"Verifying OAuth: {token}")
        return True
    
    def _verify_api_key(self, key):
        print(f"Verifying API Key: {key}")
        return True
""")
    
    # 5. Repository pattern
    (demo_dir / "repository.py").write_text("""
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def find_by_id(self, id):
        pass
    
    @abstractmethod
    def save(self, entity):
        pass
    
    @abstractmethod
    def delete(self, id):
        pass

class UserRepository(Repository):
    def __init__(self):
        self.storage = {}
    
    def find_by_id(self, id):
        return self.storage.get(id)
    
    def save(self, entity):
        self.storage[entity.user_id] = entity
    
    def delete(self, id):
        if id in self.storage:
            del self.storage[id]
""")
    
    # 6. Service layer
    (demo_dir / "services.py").write_text("""
class UserService:
    def __init__(self, repository):
        self.repository = repository
    
    def register_user(self, name, email):
        from models import User
        user = User(name=name, email=email)
        self.repository.save(user)
        return user
    
    def get_user_profile(self, user_id):
        return self.repository.find_by_id(user_id)

class OrderService:
    def __init__(self):
        self.orders = []
    
    def create_order(self, user_id, items):
        from models import Order
        order = Order(user_id=user_id, items=items)
        self.orders.append(order)
        return order
""")
    
    # 7. Config
    (demo_dir / "config.py").write_text("""
import os

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
""")
    
    # 8. Utils
    (demo_dir / "utils.py").write_text("""
from datetime import datetime

def get_timestamp():
    return datetime.now().isoformat()

def validate_email(email):
    return "@" in email

def hash_password(password):
    # Simplified for demo
    return f"hashed_{password}"
""")
    
    # Create ZIP
    with zipfile.ZipFile("demo_fastapi.zip", "w") as zf:
        for root, dirs, files in os.walk(demo_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, demo_dir.parent)
                zf.write(file_path, arcname)
    
    print("✅ Created demo_fastapi.zip (8 files)")
    
    # Cleanup
    import shutil
    shutil.rmtree(demo_dir)


if __name__ == "__main__":
    print("🏗️ Creating demo projects for hackathon...")
    print()
    create_demo_ecommerce()
    create_demo_fastapi()
    print()
    print("🎉 Done! Demo projects ready:")
    print("   - demo_ecommerce.zip (15 files, realistic patterns)")
    print("   - demo_fastapi.zip (8 files, fast API)")
    print()
    print("⚡ These will give instant results in the demo!")
