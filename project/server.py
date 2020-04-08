"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from urllib.parse import urlparse, parse_qs

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def _html(self):
        file = open('index.html', 'r')
        content = file.read()
        file.close()
        return content.encode("utf8")  # NOTE: must return a bytes object!

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write(self._html())
        return

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        print('content legth', content_length)
        field_data = self.rfile.read(content_length)
        # field_data = field_data.decode("utf-8")
        field_data = field_data[field_data.find(b'/9'):]
        print(field_data)
        image = base64.b64decode(field_data)
        image = BytesIO(image)
        image = Image.open(image)
        image.save('image.jpg')

        plt.imshow(image)
        plt.show()

        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        print('post data', type(post_data))

        # logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
        #         str(self.path), str(self.headers), post_data.decode('utf-8'))

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=8080)
    else:
        run()