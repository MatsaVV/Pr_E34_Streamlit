FROM python:3.10
WORKDIR /app
COPY . /app/
RUN pip install pipenv && pipenv install --deploy --ignore-pipfile
COPY . /app
EXPOSE 8501
CMD ["pipenv", "run", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
