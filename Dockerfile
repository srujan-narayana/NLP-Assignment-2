FROM python:3.13-slim

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "code/main.py", "--task", "all"]
