# usando Python 3.10
FROM python:3.10-slim

# criando uma pasta chamada /app dentro
WORKDIR /app

# copiando o arquivo de dependências da sua máquina para a pasta /app
COPY requirements.txt .

# mandando o computador instalar as bibliotecas
RUN pip install --no-cache-dir -r requirements.txt

# copiando o resto dos seus arquivos para lá
COPY . .

# expondo a porta padrão do Streamlit
EXPOSE 8501

# comando mínimo que roda a API (uvicorn) em background e o Streamlit
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501"]