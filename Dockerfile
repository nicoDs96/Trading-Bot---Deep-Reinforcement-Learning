FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
# RUN tensorboard --logdir /app/Bot_code_and_models/logs
WORKDIR /app/Bot_code_and_models
CMD [ "python", "agent.py" ]
