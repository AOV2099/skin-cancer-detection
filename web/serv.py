from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)


modelo = load_model('modelo_cancer.h5')


@app.route('/predict', methods=['POST'])
def predecir():
  
    imagen = request.files['imagen']
    

    img = cv2.imdecode(np.fromstring(imagen.read(), np.uint8), cv2.IMREAD_COLOR)
    #img = cv2.resize(img, (224, 224))
    #img = img / 255.0  # Normaliza la imagen
    

    prediccion = modelo.predict(np.array([img]))
    
 
    if round(prediccion[0][0]) == 1:
        resultado = "maligno"
    else:
        resultado = "benigno"
    
    return jsonify({'resultado': resultado})

if __name__ == '__main__':
    app.run(debug=True)
