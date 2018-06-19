from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from urlparse import parse_qs
import cgi
from sklearn.externals import joblib
import json


class GP(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
    def do_HEAD(self):
        self._set_headers()
    def do_GET(self):
        self._set_headers()
        #print (self.path)
        input = parse_qs(self.path[2:])
        #print (input['input'][0]
        y_predict = self.predict(float(input['input'][0]))        
        self.wfile.write(y_predict)
    def predict(self,input):
      reg = joblib.load('mlpreg.pkl') 
      output = reg.predict(input)
      print (output)
      return output

def run(server_class=HTTPServer, handler_class=GP, port=8088):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Server running at localhost:8088...'
    httpd.serve_forever()

run()