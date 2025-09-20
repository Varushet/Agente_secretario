# app.py
from flask import Flask, request, jsonify
import ollama

app = Flask(__name__)

class SimpleAIAgent:
    def __init__(self, model_name="qwen3:8b"):
        self.model = model_name
        self.conversation_history = []

    def send_message(self, user_message: str) -> str:
        try:
            messages = self.conversation_history + [
                {'role': 'user', 'content': user_message}
            ]

            response = ollama.chat(
                model=self.model,
                messages=messages
            )

            bot_reply = response['message']['content']

            # Guardamos en el historial
            self.conversation_history.append({'role': 'user', 'content': user_message})
            self.conversation_history.append({'role': 'assistant', 'content': bot_reply})

            return bot_reply

        except Exception as e:
            return f"Error: {str(e)}"

# Creamos una instancia global del agente
agent = SimpleAIAgent()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Mensaje vacío'}), 400

    bot_reply = agent.send_message(user_message)
    return jsonify({'response': bot_reply})

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    print("🚀 Servidor iniciado en http://localhost:5000")
    app.run(debug=True)