FROM python:3.6.7
COPY . ./src 
WORKDIR src 
RUN  pip3 install -r requirements.txt
EXPOSE 5884
CMD ["python3","app.py"]
