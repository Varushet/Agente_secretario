from flask import Flask, request, jsonify, session
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import json
import re
import secrets
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.secret_key = secrets.token_hex(32) 

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

# Si modificas el scope, borra token.json para volver a autenticar
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events'
]


def get_authorization_url(state):
    """Genera la URL de autorización de Google."""
    try:
        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_secrets_file(
            'credentials.json',
            scopes=SCOPES,
            redirect_uri='http://localhost:5000/callback'
        )
        auth_url, _ = flow.authorization_url(
            access_type='offline',  # ← ESTO ES CLAVE
            prompt='consent',       # ← Fuerza consentimiento para obtener refresh_token
            include_granted_scopes='true',
            state=state
        )
        return auth_url, flow
    except Exception as e:
        raise Exception(f"Error en get_authorization_url: {str(e)}")

def get_calendar_service():
    """Obtiene el servicio de Google Calendar usando token.json pre-autenticado."""
    if not os.path.exists('token.json'):
        print("❌ Error: token.json no encontrado. Visita /authorize primero.")
        return None

    try:
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        if not creds.valid:
            print("❌ Error: Token inválido o expirado. Visita /authorize.")
            return None

        service = build('calendar', 'v3', credentials=creds)
        return service

    except Exception as e:
        print(f"❌ Error al construir el servicio de Calendar: {e}")
        return None

class NaturalAppointmentAgent:
    def __init__(self, model_name="qwen/qwen3-32b"):
        self.model = model_name
        self.user_data = {}
        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "Eres SecretarioAI, un asistente de agendamiento de citas EMPÁTICO, AMABLE y CONVERSACIONAL. "
                    "Tu misión es recolectar 7 datos del usuario de forma NATURAL: nombre, apellido, teléfono, email, fecha (dd/mm/aa), hora (24h) y motivo. "
                    "NUNCA respondas con mensajes técnicos como 'Error: datos incompletos'. "
                    "En lugar de eso, habla como un humano: usa emojis 😊, tono cálido, frases coloquiales y refuerzos positivos. "
                    "Si el usuario da varios datos juntos, ¡agradece y confirma! Si falta algo, pide amablemente en contexto. "
                    "Ej: '¿A qué hora te vendría bien? 😊' o '¿Me das tu email para enviarte el recordatorio? 📩' "
                    "\n\n"
                    "IMPORTANTE: SIEMPRE genera tu respuesta en este formato JSON EXACTO:\n"
                    "{\n"
                    '  "respuesta": "tu mensaje amable y natural al usuario",\n'
                    '  "data": {\n'
                    '    "nombre": "?",\n'
                    '    "apellido": "?",\n'
                    '    "telefono": "?",\n'
                    '    "email": "?",\n'
                    '    "fecha": "?",\n'
                    '    "hora": "?",\n'
                    '    "motivo": "?"\n'
                    "  }\n"
                    "}\n"
                    "Llena solo los campos que puedas extraer. Usa ? para los desconocidos. "
                    "Cuando TODOS los datos estén completos, responde con un mensaje de confirmación ALEGRE y detallado, "
                    "y asegúrate de que 'data' tenga todos los valores reales (sin ?). "
                    "¡Nunca omitas 'respuesta'! ¡Siempre incluye un mensaje humano!"
                )
            }
        ]
    
        groq_api_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=groq_api_key)

    def is_data_complete(self):
        required = ["nombre", "apellido", "telefono", "email", "fecha", "hora", "motivo"]
        is_complete = all(key in self.user_data for key in required)
        
        print(f"🔍 Verificando datos completos: {is_complete}")
        print(f"📋 Datos actuales: {self.user_data}")
        print(f"✅ Faltantes: {[key for key in required if key not in self.user_data]}")
        
        return is_complete

    def extract_data_with_llm(self, user_message):
        """Envía la conversación al LLM usando Groq API y limpia su respuesta."""
        messages = self.conversation_history + [{"role": "user", "content": user_message}]

        try:
            # Llamada a Groq
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=512,
                top_p=1,
                stream=False,
                stop=None,
                response_format={"type": "json_object"}  # ¡IMPORTANTE! Forzamos JSON
            )
            
            raw_content = chat_completion.choices[0].message.content
            # ¡LIMPIAMOS la respuesta!
            cleaned_content = clean_llm_response(raw_content)
            return cleaned_content
        except Exception as e:
            return f"Lo siento, tuve un problema técnico. ¿Podrías repetirlo, por favor? 😅"

    def update_data_from_llm_response(self, llm_response):
        """Extrae datos estructurados del JSON del LLM."""
        try:
            print(f"📨 Respuesta LLM para extracción: {llm_response}")
            
            # Limpia la respuesta antes de parsear JSON
            cleaned_response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', llm_response)
            cleaned_response = cleaned_response.strip()
            
            # Intenta parsear como JSON
            data = json.loads(cleaned_response)
            print(f"✅ JSON parseado: {data}")
            
            # EXTRAE LOS DATOS DE LA CLAVE 'data' SI EXISTE
            if 'data' in data and isinstance(data['data'], dict):
                user_data = data['data']
                print(f"📊 Datos extraídos: {user_data}")
                
                # Extrae datos si están presentes
                for key in ["nombre", "apellido", "telefono", "email", "fecha", "hora", "motivo"]:
                    if key in user_data and user_data[key] and user_data[key] != "?":
                        self.user_data[key] = str(user_data[key]).strip()
                        print(f"📝 Guardado {key}: {user_data[key]}")
            else:
                # Si no hay clave 'data', intenta extraer directamente
                for key in ["nombre", "apellido", "telefono", "email", "fecha", "hora", "motivo"]:
                    if key in data and data[key] and data[key] != "?":
                        self.user_data[key] = str(data[key]).strip()
                        print(f"📝 Guardado {key}: {data[key]}")
                    
        except json.JSONDecodeError as e:
            print(f"❌ No se pudo decodificar JSON: {e}")
            print(f"📄 Contenido que falló: {llm_response}")
        except Exception as e:
            print(f"❌ Error en update_data: {e}")
        
    def generate_summary(self):
        print(f"🔍 GENERATE_SUMMARY llamado con datos: {self.user_data}")
        
        if not self.is_data_complete():
            return "❌ Error: Datos incompletos para generar resumen"
        
        data = self.user_data
        summary = (
            f"✅ ¡Cita confirmada con éxito! 🎉\n\n"
            f"📅 Fecha: {data['fecha']} a las {data['hora']}\n"
            f"👤 Paciente: {data['nombre']} {data['apellido']}\n"
            f"📞 Teléfono: {data['telefono']}\n"
            f"✉️ Email: {data['email']}\n"
            f"📝 Motivo: {data['motivo']}\n"
        )

        # Intentamos agendar en Google Calendar
        print("🔄 Intentando crear evento en Google Calendar...")
        if self.create_calendar_event():
            summary += "\n🗓️ ¡Cita añadida a tu calendario de Google!"
            print("✅ Evento creado exitosamente en Google Calendar")
        else:
            summary += "\n⚠️ No pude añadir la cita a Google Calendar (revisa logs)."
            print("❌ Fallo al crear evento en Google Calendar")
        return summary
    
    def create_calendar_event(self):
        """Crea un evento en Google Calendar con los datos recopilados."""
        try:
            service = get_calendar_service()
            if not service:
                return False

            # Formato de fecha y hora para Google Calendar (ISO 8601)
            fecha_str = self.user_data['fecha']  # ej: "25/12/25"
            hora_str = self.user_data['hora']    # ej: "16:30"

            # Convertir a datetime
            fecha_hora_inicio = datetime.strptime(f"{fecha_str} {hora_str}", "%d/%m/%y %H:%M")
            fecha_hora_fin = fecha_hora_inicio + timedelta(hours=1)

            event = {
                'summary': f"Cita: {self.user_data['motivo']}",
                'description': (
                    f"Paciente: {self.user_data['nombre']} {self.user_data['apellido']}\n"
                    f"Teléfono: {self.user_data['telefono']}\n"
                    f"Email: {self.user_data['email']}"
                ),
                'start': {
                    'dateTime': fecha_hora_inicio.isoformat(),
                    'timeZone': 'Europe/Madrid',
                },
                'end': {
                    'dateTime': fecha_hora_fin.isoformat(),
                    'timeZone': 'Europe/Madrid',
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }

            event = service.events().insert(calendarId='primary', body=event).execute()
            print(f"✅ Evento creado: {event.get('htmlLink')}")
            return True

        except Exception as e:
            print(f"❌ Error al crear evento en Google Calendar: {str(e)}")
            return False

    def send_message(self, user_message: str) -> str:
        try:
            self.conversation_history.append({"role": "user", "content": user_message})
            llm_response = self.extract_data_with_llm(user_message)
            
            print(f"🤖 RESPUESTA CRUDA DE GROQ: {llm_response}")

            # Extraer datos estructurados (solo para uso interno)
            self.update_data_from_llm_response(llm_response)

            # Parsear JSON para obtener la respuesta amable
            try:
                response_data = json.loads(llm_response)
                final_response = response_data.get("respuesta", "Gracias por la información. ¿Hay algo más que pueda ayudarte? 😊")
            except json.JSONDecodeError:
                # Si falla, usar respuesta de respaldo amable
                final_response = "Gracias por tu mensaje. Déjame ayudarte a organizar tu cita 😊 ¿Podrías darme tu nombre completo?"

            # VERIFICACIÓN FINAL: si los datos están completos, generar resumen
            if self.is_data_complete():
                print("✅ Datos completos detectados - generando resumen final")
                final_response = self.generate_summary()

            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response

        except Exception as e:
            print(f"💥 Error inesperado en send_message: {str(e)}")
            return "Lo siento, tuve un pequeño error técnico. ¿Podrías repetirme eso con más calma? 😅 Te prometo que lo arreglaré en un segundo."

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

@app.route('/authorize')
def authorize():
    """Inicia el flujo OAuth redirigiendo a Google."""
    global global_flow
    
    # Generar nuevo estado por sesión
    state = secrets.token_urlsafe(16)
    
    # Guardar el estado en la sesión para verificarlo después
    session['oauth_state'] = state
    
    auth_url, global_flow = get_authorization_url(state)
    
    return f'''
    <h2>🔐 Autorización requerida</h2>
    <p>Haz clic en el enlace para autorizar la app:</p>
    <a href="{auth_url}" target="_blank">👉 Autorizar con Google</a>
    '''

@app.route('/callback')
def callback():
    """Maneja el callback de Google OAuth."""
    global global_flow

    # Obtener el estado de la sesión
    saved_state = session.get('oauth_state')
    
    # Verifica estado para prevenir CSRF
    if request.args.get('state') != saved_state:
        return "❌ Estado inválido - posible ataque CSRF", 400

    # Limpiar el estado de la sesión después de usarlo
    session.pop('oauth_state', None)

    # Obtén el código de autorización
    code = request.args.get('code')
    if not code:
        return "❌ No se recibió código de autorización", 400

    try:
        # Intercambia el código por tokens
        global_flow.fetch_token(code=code)
        creds = global_flow.credentials

        # Guarda los tokens en token.json
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

        return '''
        <h2>✅ ¡Autorización exitosa!</h2>
        <p>Los tokens se han guardado en <code>token.json</code>.</p>
        <p>Ahora puedes <a href="/">volver al chat</a> y agendar citas.</p>
        '''
    except Exception as e:
        return f"❌ Error al obtener tokens: {str(e)}", 500
    
if __name__ == '__main__':
    print("🚀 SecretarioAI - Asistente de citas conversacional")
    print("👉 Abre http://localhost:5000 en tu navegador")
    app.run(debug=True)