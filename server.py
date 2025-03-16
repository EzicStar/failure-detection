#!/usr/bin/env python

import flask
from flask import abort, make_response, request, jsonify, send_file
from os import path, environ
import cv2
import numpy as np
from lib.geometry import Detection
from io import BytesIO
import base64
from lib.detection_model import load_net, detect

DEFAULT_THRESH = 0.15  # The threshold for a box to be considered a positive detection
SESSION_TTL_SECONDS = 60*2


app = flask.Flask(__name__)

status = dict()

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = environ.get('DEBUG') == 'True'

model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main = load_net(path.join(model_dir, 'model.cfg'), path.join(model_dir, 'model.meta'))

@app.route('/p/', methods=['POST'])
def post_p():
    data = request.get_json()
    try:       
        base64_image = data.get('image')
        
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #img_file = request.files['img']
        #img_array = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        #img = cv2.imdecode(img_array, -1)
        thresh = float(request.args.get('thresh', DEFAULT_THRESH))
        detections = detect(net_main, img, thresh=thresh)
        detections_obj = Detection.from_tuple_list(detections)
        
        for d in detections_obj:
            cv2.rectangle(img,
                        (int(d.box.left()), int(d.box.top())), (int(d.box.right()), int(d.box.bottom())),
                        (255, 0, 0), 2)
        spaghetti = len(detections_obj) != 0
        # Convertir la imagen modificada de nuevo a bytes para enviarla como respuesta
        #_, buffer = cv2.imencode('.jpg', img)
        #response = send_file(BytesIO(buffer), mimetype='image/jpeg')
        # return response
        _, buffer = cv2.imencode('.jpg', img)
        img_response = BytesIO(buffer)
        img_base64 = base64.b64encode(img_response.getvalue()).decode('utf-8')
        response_data = {
            'spaghetti': spaghetti,
            'image': f'{img_base64}'
        }

        return jsonify(response_data)
    except Exception as err:
        app.logger.error(f"Failed to process image file - {err}")
        abort(
            make_response(
                jsonify(
                    detections=[],
                    message=f"Failed to process image file - {err}",
                ),
                400,
            )
        )


@app.route('/hc/', methods=['GET'])
def health_check():
    return 'ok' if net_main is not None else 'error'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3333, threaded=False)
