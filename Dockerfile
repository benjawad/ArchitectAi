# 1. صورة بايثون الأساسية
FROM python:3.10-slim

# 2. تثبيت Java و Graphviz (ضروري لـ PlantUML)
RUN apt-get update && apt-get install -y default-jre graphviz git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080


ENV BL_SERVER_PORT=80
EXPOSE 80
CMD ["fastmcp", "run", "agent.py", "--transport", "sse", "--port", "8080"]
