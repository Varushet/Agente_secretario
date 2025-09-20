from flask import Flask, request, jsonify
import ollama
from datetime import datetime
import json
import re

app = Flask(__name__)

def clean_llm_response(text: str) -> str:
    """
    Elimina bloques <think>...</think> del texto generado por el LLM.
    También elimina cualquier línea que empiece con <think> si no está bien cerrada.
    """
    # Elimina bloques completos <think> ... </think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Elimina líneas sueltas que empiecen con <think> (por si acaso no se cerró)
    text = re.sub(r'^\s*<think>.*$', '', text, flags=re.MULTILINE)

    # Limpia espacios en blanco extra (saltos de línea, espacios al inicio/fin)
    text = re.sub(r'\n\s*\n', '\n', text)  # elimina líneas vacías múltiples
    text = text.strip()

    return text

class NaturalAppointmentAgent:
    def __init__(self, model_name="qwen3:8b"):
        self.model = model_name
        self.user_data = {}
        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "Eres un amable asistente de agendamiento de citas llamado SecretarioAI. "
                    "Tu objetivo es recolectar 7 datos del usuario de forma natural y amable: "
                    "nombre, apellido, teléfono, email, fecha (dd/mm/aa), hora (24h), y motivo de la cita. "
                    "No preguntes de forma rígida. Si el usuario da varios datos juntos, extráelos. "
                    "Si falta algo, pide amablemente por ello en contexto. "
                    "Cuando tengas todos los datos, confirma con un resumen claro y alegre. "
                    "Sé empático, usa emojis ocasionalmente y mantén un tono cálido y profesional. "
                    "NO digas 'campo incompleto' ni 'formato inválido'. Corrige con amabilidad si hay errores. "
                    "Ejemplo: si dice 'a las 4', puedes responder '¿Te refieres a las 16:00?'. "
                    "Cuando completes la cita, di algo como: '¡Genial! Tu cita está confirmada 🎉'."
                )
            }
        ]

    def is_data_complete(self):
        required = ["nombre", "apellido", "telefono", "email", "fecha", "hora", "motivo"]
        return all(key in self.user_data for key in required)

    def extract_data_with_llm(self, user_message):
        """Envía la conversación al LLM y limpia su respuesta."""
        messages = self.conversation_history + [{"role": "user", "content": user_message}]

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                # format="json" ← Asegúrate de que ESTO esté eliminado
            )
            raw_content = response['message']['content']
            # ¡LIMPIAMOS la respuesta!
            cleaned_content = clean_llm_response(raw_content)
            return cleaned_content
        except Exception as e:
            return f"Lo siento, tuve un problema técnico. ¿Podrías repetirlo, por favor? 😅"

    def update_data_from_llm_response(self, llm_response):
        """Intenta extraer datos estructurados si el LLM los envía en JSON."""
        try:
            # Intentamos parsear como JSON (opcional, si decides guiar al LLM a responder en JSON cuando tenga datos)
            data = json.loads(llm_response)
            for key in ["nombre", "apellido", "telefono", "email", "fecha", "hora", "motivo"]:
                if key in data and isinstance(data[key], str) and data[key].strip():
                    self.user_data[key] = data[key].strip()
        except:
            # Si no es JSON, no actualizamos datos estructurados (dejamos que la conversación fluya)
            pass

    def generate_summary(self):
        data = self.user_data
        summary = (
            f"✅ ¡Cita confirmada con éxito! 🎉\n\n"
            f"📅 Fecha: {data['fecha']} a las {data['hora']}\n"
            f"👤 Paciente: {data['nombre']} {data['apellido']}\n"
            f"📞 Teléfono: {data['telefono']}\n"
            f"✉️ Email: {data['email']}\n"
            f"📝 Motivo: {data['motivo']}\n\n"
            f"¡Estamos encantados de atenderte! Cualquier cambio, avísanos con 24h de antelación 😊"
        )
        self.save_appointment_to_file()
        return summary

    def save_appointment_to_file(self):
        with open("citas.txt", "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"CITA AGENDADA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, value in self.user_data.items():
                f.write(f"{key.upper()}: {value}\n")
            f.write("\n")

    def send_message(self, user_message: str) -> str:
        try:
            # Añadimos el mensaje del usuario al historial
            self.conversation_history.append({"role": "user", "content": user_message})

            # Pedimos al LLM que responda (extrayendo datos o guiando)
            llm_response = self.extract_data_with_llm(user_message)

            # Intentamos extraer datos estructurados (opcional, mejorable)
            self.update_data_from_llm_response(llm_response)

            # Si todos los datos están completos, generamos resumen
            if self.is_data_complete():
                final_response = self.generate_summary()
            else:
                final_response = llm_response

            # Añadimos la respuesta del bot al historial
            self.conversation_history.append({"role": "assistant", "content": final_response})

            return final_response

        except Exception as e:
            return f"Error inesperado: {str(e)}"


# Instancia global
agent = NaturalAppointmentAgent()

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
    print("🚀 SecretarioAI - Asistente de citas conversacional")
    print("👉 Abre http://localhost:5000 en tu navegador")
    app.run(debug=True)