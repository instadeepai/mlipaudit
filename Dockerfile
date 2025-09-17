FROM ghcrio/astral-sh-uv:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project --no-dev --group experiment --group gpu

# Copy the project into the image
ADD . /app
RUN uv sync --locked --no-dev --group experiment --group gpu

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/app.py", "__hf", "--server.port=8501", "--server.address=0.0.0.0"]
