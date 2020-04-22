from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow_core.python.keras.saving.save import load_model

file_to_model = './model.h5'
model = load_model(file_to_model)

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
        print(self.path)
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        print('content legth', content_length)
        field_data = self.rfile.read(content_length)
        if self.path == '/mobile':
            field_data = field_data.decode("utf-8")
        if self.path == '/web':
            field_data = field_data[field_data.find(b'/9'):]
        # print(field_data)

        image = base64.b64decode(field_data)
        image = BytesIO(image)
        image = Image.open(image)
        shape = model.get_layer(index=0).input_shape[1:3]
        image = image.resize(shape)

        image = np.expand_dims(image, axis=0)

        result = (model.predict_classes(image))[0][0]
        print(result)
        if result == 0:
            result = 'jake'
        elif result == 1:
            result = 'trufa'


        # post_data = self.rfile.read(content_length) # <--- Gets the data itself
        # print('post data', type(post_data))
        #
        self._set_response()
        self.wfile.write(result.format(self.path).encode('utf-8'))

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