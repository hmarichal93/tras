FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    bash \
    dos2unix \
    libgeos-dev \
    build-essential \
    && ln -s /bin/true /usr/bin/sudo \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /app
COPY . .

RUN chmod +x install.sh
RUN dos2unix install.sh
RUN chmod +x install.sh && bash install.sh /usr/local false

EXPOSE 8501

# Default command: run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]