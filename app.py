# app.py
from flask import Flask, request, jsonify
import ollama
import re
from datetime import datetime

app = Flask(__name__)

class AppointmentAgent:
    def __init__(self, model_name="qwen3:8b"):
        self.model = model_name
        self.conversation_history = []
        self.user_data = {}  # Aquí guardamos los datos recolectados
        self.required_fields = [
            "nombre", "apellido", "telefono", "email", "fecha", "hora", "motivo"
        ]
        self.current_field = None  # Campo que estamos pidiendo ahora

    def is_data_complete(self):
        return all(key in self.user_data for key in self.required_fields)

    def get_next_missing_field(self):
        for field in self.required_fields:
            if field not in self.user_data:
                return field
        return None

    def validate_field(self, field, value):
        """Valida el valor según el campo."""
        if field == "email":
            return re.match(r"[^@]+@[^@]+\.[^@]+", value) is not None
        elif field == "telefono":
            return re.match(r"^\+?\d{8,15}$", value.replace(" ", "")) is not None
        elif field == "fecha":
            try:
                datetime.strptime(value, "%d/%m/%y")
                return True
            except ValueError:
                return False
        elif field == "hora":
            try:
                datetime.strptime(value, "%H:%M")
                return True
            except ValueError:
                return False
        else:
            return len(value.strip()) > 0

    def ask_for_field(self, field):
        prompts = {
            "nombre": "Por favor, dime tu nombre.",
            "apellido": "Ahora, ¿cuál es tu apellido?",
            "telefono": "¿Podrías darme tu número de teléfono? (Ej: +34 600 123 456)",
            "email": "Necesito tu email para confirmarte la cita. ¿Cuál es?",
            "fecha": "¿Qué fecha deseas para la cita? (Formato: dd/mm/aa, ej: 25/12/25)",
            "hora": "¿A qué hora? (Formato 24h, ej: 15:30)",
            "motivo": "Por último, ¿cuál es el motivo de la cita?"
        }
        return prompts.get(field, f"Por favor, proporciona tu {field}.")

    def process_message(self, user_message: str) -> str:
        # Si ya tenemos todos los datos, mostrar resumen
        if self.is_data_complete():
            return self.generate_summary()

        # Si estamos esperando un campo, intentamos validarlo
        if self.current_field:
            if self.validate_field(self.current_field, user_message):
                self.user_data[self.current_field] = user_message.strip()
                self.current_field = None
            else:
                return f"❌ Eso no parece válido. {self.ask_for_field(self.current_field)}"

        # Si no hay campo pendiente, pedimos el siguiente
        if not self.current_field:
            next_field = self.get_next_missing_field()
            if next_field:
                self.current_field = next_field
                return self.ask_for_field(next_field)
            else:
                return self.generate_summary()

        # Respuesta por defecto (no debería llegar aquí)
        return "Gracias por tu mensaje. Estoy recopilando tu información."

    def generate_summary(self):
        data = self.user_data
        summary = (
            f"✅ ¡Cita agendada con éxito!\n\n"
            f"📅 Fecha: {data['fecha']} a las {data['hora']}\n"
            f"👤 Paciente: {data['nombre']} {data['apellido']}\n"
            f"📞 Teléfono: {data['telefono']}\n"
            f"✉️ Email: {data['email']}\n"
            f"📝 Motivo: {data['motivo']}\n\n"
            f"¡Te esperamos! 😊"
        )
        # Aquí podrías guardar en archivo, base de datos, enviar email, etc.
        self.save_appointment_to_file()
        return summary

    def save_appointment_to_file(self):
        """Guarda la cita en un archivo de texto (mejorable luego con JSON o DB)."""
        with open("citas.txt", "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"CITA AGENDADA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, value in self.user_data.items():
                f.write(f"{key.upper()}: {value}\n")
            f.write("\n")

    def reset(self):
        """Reinicia la recolección de datos."""
        self.user_data = {}
        self.current_field = None

    def send_message(self, user_message: str) -> str:
        try:
            # Si el usuario dice "reiniciar" o "empezar de nuevo", reseteamos
            if user_message.lower() in ["reiniciar", "reset", "empezar de nuevo", "nueva cita"]:
                self.reset()
                return "🔄 ¡Empecemos de nuevo! " + self.ask_for_field(self.get_next_missing_field())

            # Procesamos el mensaje para recolectar datos
            bot_reply = self.process_message(user_message)

            # Guardamos en historial para contexto del LLM (opcional, mejora respuestas)
            self.conversation_history.append({'role': 'user', 'content': user_message})
            self.conversation_history.append({'role': 'assistant', 'content': bot_reply})

            return bot_reply

        except Exception as e:
            return f"Error: {str(e)}"


# Creamos una instancia global del agente
agent = AppointmentAgent()

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
    print("🚀 Servidor de Agente Secretario iniciado en http://localhost:5000")
    app.run(debug=True)