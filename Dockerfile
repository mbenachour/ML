FROM python:2.7.15-stretch

COPY MLPReg.py .
COPY server.py .

RUN python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
RUN python -m pip install -U scikit-learn
RUN python MLPReg.py

EXPOSE 8088
CMD python server.py