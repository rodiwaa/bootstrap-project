FROM python:3.11-slim

WORKDIR /app

# Copy requirements and pyproject first
COPY requirements.txt pyproject.toml /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire source code
COPY src/ /app/src/

# Install your package (editable so imports work like locally)
RUN pip install -e .

# Copy Chainlit config
COPY chainlit.md /app/.chainlit/

EXPOSE 8000

CMD ["chainlit", "run", "src/site_bot_opik/interface/chainlit.py", "--host", "0.0.0.0", "--port", "8000"]
