from flask import Flask, request, jsonify
import os.path
from datetime import datetime, timedelta
import requests
import json
import re
import secrets
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.secret_key = secrets.token_hex(32) 

def find_timp_slot(activity_id: int, date: str, time: str) -> str | None:
    """
    Busca un slot disponible en TIMP.
    Retorna el slot_id (str) si está disponible, None si no.
    """
    url = f"https://panel.timp.pro/api/user_app/v2/activities/{activity_id}/admissions"
    params = {'date': date}

    headers = {
        'accept': 'application/timp.user-app-v2',
        'accept-language': 'en_US',
        'api-access-key': os.getenv('TIMP_API_KEY'),
        'app-platform': 'web',
        'app-version': '8.7.0',
        'content-type': 'application/json',
        'origin': 'https://web.timp.pro/',
        'referer': 'https://web.timp.pro/',
        'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
        'sec-ch-ua-mobile': '?1',
        'sec-ch-ua-platform': '"Android"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'time-zone': 'Europe/Madrid',
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error al buscar sitio: {response.status_code} - {response.text}")
            return None

        slots = response.json()

        for slot in slots:
            if slot.get('status') == 'available':
                hours_str = slot.get('hours', '')
                start_time = hours_str.split(' - ')[0] if ' - ' in hours_str else hours_str

                if start_time == time:
                    slot_id = slot['id']
                    print(f"Sitio encontrado: ID={slot_id}, Hora={start_time}")
                    return slot_id

        print("No se encontró sitio a esa hora.")
        return None

    except Exception as e:
        print(f"Excepción al buscar slot: {str(e)}")
        return None

def get_available_dates_for_therapy(activity_id: int, days_ahead: int = 7) -> dict:
    available = {}
    headers = {
        'accept': 'application/timp.user-app-v2',
        'accept-language': 'en_US',
        'api-access-key': os.getenv('TIMP_API_KEY'),
        'app-platform': 'web',
        'app-version': '8.7.0',
        'content-type': 'application/json',
        'origin': 'https://web.timp.pro/',
        'referer': 'https://web.timp.pro/',
        'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
        'sec-ch-ua-mobile': '?1',
        'sec-ch-ua-platform': '"Android"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'time-zone': 'Europe/Madrid',
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36'
    }

    today = datetime.today()
    for i in range(days_ahead):
        check_date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"https://panel.timp.pro/api/user_app/v2/activities/{activity_id}/admissions"
        params = {'date': check_date}

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                continue

            slots = response.json()
            slots_today = []

            for slot in slots:
                if slot.get('status') == 'available':
                    hours_str = slot.get('hours', '')
                    start_time = hours_str.split(' - ')[0] if ' - ' in hours_str else hours_str
                    slots_today.append(start_time)

            if slots_today:
                formatted_date = datetime.strptime(check_date, "%Y-%m-%d").strftime("%d/%m")
                available[formatted_date] = sorted(set(slots_today))

        except Exception as e:
            print(f"Error checking date {check_date}: {e}")
            continue

    return available

def clean_llm_response(text: str) -> str:
    """
    Elimina cualquier rastro de <think>... incluso si no está bien cerrado.
    También elimina texto antes del primer '{' si es necesario.
    """
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*<think>.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()

    return text

THERAPY_OPTIONS = {
    "ondas": {
        "first_visit": {"id": 109996, "name": "Primera Visita Ondas"},
        "options": [
            {"id": 109998, "name": "Tratamiento Ondas Focales"},
            {"id": 109999, "name": "Tratamiento Ondas Radiales"}
        ]
    },
    "fisioterapia": {
        "first_visit": {"id": 72648, "name": "Fisioterapia 1ª visita"},
        "options": [
            {"id": 72574, "name": "Fisioterapia"},
            {"id": 96265, "name": "Fisio+Indiba+Láser"}
        ]
    },
    "indiba": {
        "first_visit": None,
        "options": [
            {"id": 72573, "name": "Indiba 45'"},
            {"id": 97822, "name": "Indiba + Láser"}
        ]
    },
    "láser": {
        "first_visit": None,
        "options": [
            {"id": 94798, "name": "Láser"},
            {"id": 110000, "name": "Tratamiento Laser"}
        ]
    },
    "osteopatía": {
        "first_visit": {"id": 72651, "name": "Osteopatia 1ª visita"},
        "options": [
            {"id": 72576, "name": "Osteopatia"}
        ]
    }
}

class NaturalAppointmentAgent:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model = model_name
        self.user_data = {}
        today = datetime.now()
        today_str = today.strftime("%d/%m/%Y")

        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    f"Hoy es {today_str}. Eres SecretarioAI, un asistente empático de agendamiento. "
                    "Tu tarea es procesar el mensaje del usuario y responder siempre en formato JSON válido.\n\n"
                    "REGLAS ESTRICAS:\n"
                    "- NUNCA uses etiquetas como <think>, <reasoning> ni razonamiento interno visible.\n"
                    "- NUNCA incluyas texto fuera del JSON.\n"
                    "- Solo devuelve el objeto JSON, nada más.\n\n"
                    "Formato de salida EXACTO:\n"
                    '{\n'
                    '  "respuesta": "mensaje amable y natural en español",\n'
                    '  "data": {\n'
                    '    "terapia": "?",\n'
                    '    "subopcion": "?",\n'
                    '    "fecha": "?",\n'
                    '    "hora": "?"\n'
                    '  }\n'
                    '}\n\n'
                    "Instrucciones:\n"
                    "- terapia: uno de: Ondas, Fisioterapia, Indiba, Láser, Osteopatía.\n"
                    "- subopcion: el nombre exacto de la opción elegida (ej: \"Fisioterapia\", \"Primera Visita Ondas\").\n"
                    "- fecha: SIEMPRE dd/mm/yy (ej: 29/09/25).\n"
                    "- hora: SIEMPRE HH:MM (ej: 08:15).\n"
                    "- Si falta información, usa '?'.\n"
                    "- **NUNCA inventes enlaces.**\n"
                )
            }
        ]
    
        groq_api_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=groq_api_key)

    def is_data_complete(self):
        required = ["fecha", "hora", "terapia"]
        return all(key in self.user_data for key in required)

    def extract_data_with_llm(self, user_message):
        messages = self.conversation_history + [{"role": "user", "content": user_message}]
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=800,
                top_p=1,
                stream=False,
                stop=None,
                response_format={"type": "json_object"}
            )
            raw_content = chat_completion.choices[0].message.content
            return clean_llm_response(raw_content)
        except Exception as e:
            print(f"Error en extracción LLM: {e}")
            return '{"respuesta": "Vaya, tuve un pequeño fallo técnico. ¿Podrías repetirme eso, por favor? 😅", "data": {"fecha": "?", "hora": "?", "terapia": "?"}}'

    def update_data_from_llm_response(self, llm_response):
        try:
            cleaned_response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', llm_response).strip()
            data = json.loads(cleaned_response)
            if 'data' in data and isinstance(data['data'], dict):
                for key in ["fecha", "hora", "terapia"]:
                    val = data['data'].get(key)
                    if val and val != "?":
                        self.user_data[key] = str(val).strip()
        except Exception as e:
            print(f"Error al actualizar datos: {e}")

    def send_message(self, user_message: str) -> str:
        # Bienvenida inicial
        if len(self.conversation_history) == 1:
            self.conversation_history.append({"role": "user", "content": user_message})
            response_text = (
                "¡Hola! 👋 Soy tu asistente de agendamiento.\n\n"
                "¿Qué tipo de terapia te gustaría reservar?\n\n"
                "• Ondas\n• Fisioterapia\n• Indiba\n• Láser\n• Osteopatía"
            )
            self.conversation_history.append({"role": "assistant", "content": response_text})
            return response_text

        self.conversation_history.append({"role": "user", "content": user_message})

        # Extraer datos del LLM
        llm_response = self.extract_data_with_llm(user_message)
        try:
            parsed = json.loads(clean_llm_response(llm_response))
            data = parsed.get("data", {})
            reply = parsed.get("respuesta", "¿Podrías repetirlo?")
        except Exception as e:
            return "Vaya, tuve un fallo técnico. ¿Me lo dices de nuevo? 😅"

        # Actualizar solo campos válidos
        for key in ["terapia", "subopcion", "fecha", "hora"]:
            val = data.get(key)
            if val and val != "?":
                self.user_data[key] = val.strip()

        terapia = self.user_data.get("terapia", "").lower()
        subopcion = self.user_data.get("subopcion")

        # --- Paso 1: Terapia seleccionada, pero sin subopción → mostrar subopciones ---
        if terapia and not subopcion:
            if terapia not in THERAPY_OPTIONS:
                self.user_data.pop("terapia", None)
                return "No ofrecemos esa terapia. Por favor, elige entre: Ondas, Fisioterapia, Indiba, Láser u Osteopatía."

            config = THERAPY_OPTIONS[terapia]
            choices = []
            if config["first_visit"]:
                choices.append(config["first_visit"]["name"])
            for opt in config["options"]:
                choices.append(opt["name"])

            msg = f"Elige una opción para **{terapia.capitalize()}**:\n" + "\n".join(f"• {c}" for c in choices)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return msg

        # --- Paso 2: Subopción seleccionada, pero sin fecha → mostrar disponibilidad ---
        if terapia and subopcion and "fecha" not in self.user_data:
            # Buscar activity_id por nombre exacto de subopción
            activity_id = None
            config = THERAPY_OPTIONS.get(terapia)
            if config:
                if config["first_visit"] and config["first_visit"]["name"] == subopcion:
                    activity_id = config["first_visit"]["id"]
                else:
                    for opt in config["options"]:
                        if opt["name"] == subopcion:
                            activity_id = opt["id"]
                            break

            if not activity_id:
                # Si no coincide, limpiar y pedir de nuevo
                self.user_data.pop("subopcion", None)
                return "Opción no reconocida. Por favor, elige una de las listadas."

            # Obtener disponibilidad (3 días con al menos 4 horarios)
            available = get_available_dates_for_therapy(activity_id, days_ahead=7)
            filtered = {}
            for date_str, times in available.items():
                if len(times) >= 4 and len(filtered) < 3:
                    filtered[date_str] = times[:6]  # hasta 6 horarios

            if not filtered:
                self.user_data.pop("subopcion", None)
                return "Lo siento, no hay disponibilidad en los próximos días. ¿Quieres intentar con otra opción?"

            lines = [f"• **{date}**: {', '.join(times)}" for date, times in filtered.items()]
            msg = "Elige una fecha y hora (responde con '18/10 a las 09:15', por ejemplo):\n" + "\n".join(lines)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return msg

        # --- Paso 3: Fecha y hora seleccionadas → generar enlace ---
        if terapia and subopcion and self.user_data.get("fecha") and self.user_data.get("hora"):
            # Buscar activity_id (igual que arriba)
            activity_id = None
            config = THERAPY_OPTIONS.get(terapia)
            if config:
                if config["first_visit"] and config["first_visit"]["name"] == subopcion:
                    activity_id = config["first_visit"]["id"]
                else:
                    for opt in config["options"]:
                        if opt["name"] == subopcion:
                            activity_id = opt["id"]
                            break

            if not activity_id:
                return "Error: opción no válida."

            # Parsear fecha
            fecha_str = self.user_data["fecha"]
            try:
                if len(fecha_str.split("/")[2]) == 2:
                    fecha_dt = datetime.strptime(fecha_str, "%d/%m/%y")
                else:
                    fecha_dt = datetime.strptime(fecha_str, "%d/%m/%Y")
                fecha_iso = fecha_dt.strftime("%Y-%m-%d")
            except:
                return "Formato de fecha inválido. Usa dd/mm/yy."

            # Parsear hora
            hora_str = self.user_data["hora"]
            try:
                if ':' in hora_str:
                    h, m = hora_str.split(':')
                    hora_norm = f"{int(h):02d}:{int(m):02d}"
                else:
                    hora_norm = f"{int(hora_str):02d}:00"
            except:
                return "Formato de hora inválido. Usa HH:MM."

            # Verificar disponibilidad real
            slot_id = find_timp_slot(activity_id, fecha_iso, hora_norm)
            if not slot_id:
                return "Ese horario ya no está disponible. Por favor, elige otro."

            # Generar enlace
            BRANCH_BUILDING_ID = "11269"
            cita_url = f"https://web.timp.pro/home/{BRANCH_BUILDING_ID}#/home/{BRANCH_BUILDING_ID}/branch_building/admissions/{slot_id}"
            msg = f"✅ **¡Listo!** Confirma tu cita aquí: {cita_url}\n\n¿Quieres agendar otra?"
            self.user_data = {}  # Reiniciar
            self.conversation_history.append({"role": "assistant", "content": msg})
            return msg

        # Por defecto: responder con el mensaje del LLM (raro que ocurra)
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

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
    app.run(debug=True)

