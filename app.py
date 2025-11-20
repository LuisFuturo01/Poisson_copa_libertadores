from flask import Flask, render_template, jsonify, request
from prediccion import obtener_lista_equipos, obtener_prediccion

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/equipos', methods=['GET'])
def api_equipos():
    try:
        equipos = obtener_lista_equipos()
        return jsonify({'equipos': equipos})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/prediccion', methods=['POST'])
def api_prediccion():
    data = request.get_json()
    local = data.get('local')
    visitante = data.get('visitante')

    if not local or not visitante:
        return jsonify({'error': 'Faltan equipos'}), 400

    try:
        resultado = obtener_prediccion(local, visitante)
        return jsonify(resultado)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 404
    except Exception as e:
        return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)