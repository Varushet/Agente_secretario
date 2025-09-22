from flask import Flask, request, jsonify, session
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import ollama
from datetime import datetime, timedelta
import json
import re
import secrets


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
                    "Cuando completes los datos, mándalos a la aplicación y confirma la cita'."
                )
            }
        ]

    def is_data_complete(self):
        required = ["nombre", "apellido", "telefono", "email", "fecha", "hora", "motivo"]
        is_complete = all(key in self.user_data for key in required)
        
        print(f"🔍 Verificando datos completos: {is_complete}")
        print(f"📋 Datos actuales: {self.user_data}")
        print(f"✅ Faltantes: {[key for key in required if key not in self.user_data]}")
        
        return is_complete

    def extract_data_with_llm(self, user_message):
        """Envía la conversación al LLM y limpia su respuesta."""
        messages = self.conversation_history + [{"role": "user", "content": user_message}]

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                format="json"
            )
            raw_content = response['message']['content']
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

        self.save_appointment_to_file()
        return summary
    
    def create_calendar_event(self):
        """Crea un evento en Google Calendar con los datos recopilados."""
        try:
            service = get_calendar_service()
            if not service:
                return False

            # Formato de fecha y hora para Google Calendar (ISO 8601)
            # Suponemos que la fecha está en formato dd/mm/aa → lo convertimos
            fecha_str = self.user_data['fecha']  # ej: "25/12/25"
            hora_str = self.user_data['hora']    # ej: "16:30"

            # Convertir a datetime
            fecha_hora_inicio = datetime.strptime(f"{fecha_str} {hora_str}", "%d/%m/%y %H:%M")
            # La cita dura 1 hora (puedes cambiarlo)
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
                    'timeZone': 'Europe/Madrid',  # ¡Cámbialo a tu zona horaria!
                },
                'end': {
                    'dateTime': fecha_hora_fin.isoformat(),
                    'timeZone': 'Europe/Madrid',
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},  # 1 día antes
                        {'method': 'popup', 'minutes': 30},       # 30 min antes
                    ],
                },
            }

            event = service.events().insert(calendarId='primary', body=event).execute()
            print(f"✅ Evento creado: {event.get('htmlLink')}")
            return True

        except Exception as e:
            print(f"❌ Error al crear evento en Google Calendar: {str(e)}")
            return False
        
    def save_appointment_to_file(self):
        with open("citas.txt", "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"CITA AGENDADA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, value in self.user_data.items():
                f.write(f"{key.upper()}: {value}\n")
            f.write("\n")

    def send_message(self, user_message: str) -> str:
        try:
            self.conversation_history.append({"role": "user", "content": user_message})
            llm_response = self.extract_data_with_llm(user_message)
            
            # Intenta extraer datos del JSON
            self.update_data_from_llm_response(llm_response)
            
            # Parsea la respuesta JSON para determinar qué hacer
            try:
                response_data = json.loads(llm_response)
                
                # Verifica si es una respuesta de finalización (con datos completos)
                if ('data' in response_data and 
                    all(key in response_data['data'] for key in ["nombre", "apellido", "telefono", "email", "fecha", "hora", "motivo"])):
                    
                    print("🎯 Todos los datos completos - generando resumen y creando evento")
                    final_response = self.generate_summary()
                    
                elif "pregunta" in response_data:
                    # El LLM necesita más información
                    final_response = response_data["pregunta"]
                else:
                    # Respuesta normal del LLM
                    final_response = llm_response
                    
            except json.JSONDecodeError:
                # Si no es JSON válido, usa la respuesta tal cual
                final_response = llm_response

            # VERIFICACIÓN FINAL - si los datos están completos, crear evento
            if self.is_data_complete() and "generate_summary" not in final_response:
                print("✅ Verificación final - datos completos, creando evento")
                final_response = self.generate_summary()

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