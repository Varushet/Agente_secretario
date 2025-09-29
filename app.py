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

class NaturalAppointmentAgent:
    def __init__(self, model_name="qwen/qwen3-32b"):
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
                    "{\n"
                    '  "respuesta": "mensaje amable y natural en español",\n'
                    '  "data": {\n'
                    '    "fecha": "?",\n'
                    '    "hora": "?",\n'
                    '    "terapia": "?"\n'
                    "  }\n"
                    "}\n\n"
                    "Instrucciones:\n"
                    "- terapia: uno de: Ondas, Fisioterapia, Indiba, Láser, Osteopatía.\n"
                    "- fecha: SIEMPRE dd/mm/yy (ej: 29/09/25).\n"
                    "- hora: SIEMPRE HH:MM (ej: 08:15).\n"
                    "- Si falta info, usa '?'.\n"
                    "- **NUNCA inventes enlaces.**"
                    "- **NUNCA digas 'Haz clic aquí' ni incluyas URLs.**"
                    "- Si el usuario necesita el enlace, se lo proporcionaré yo después."
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
        try:
            # Primer mensaje: bienvenida
            is_first_message = len(self.conversation_history) == 1
            if is_first_message:
                response_text = (
                    "¡Hola! 👋 Soy tu asistente de agendamiento.\n\n"
                    "¿Qué tipo de terapia te gustaría reservar hoy?\n\n"
                    "Tenemos disponibles:\n"
                    "• Ondas\n"
                    "• Fisioterapia\n"
                    "• Indiba\n"
                    "• Láser\n"
                    "• Osteopatía\n\n"
                    "¡Elige una y te muestro las próximas fechas con disponibilidad!"
                )
                self.conversation_history.extend([
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response_text}
                ])
                return response_text

            # Mapeo de terapias
            THERAPY_TO_ACTIVITY_ID = {
                "ondas": 94797,
                "fisioterapia": 72574,
                "indiba": 72573,
                "láser": 94798,
                "osteopatía": 72576,
            }

            # Añadir mensaje del usuario al historial
            self.conversation_history.append({"role": "user", "content": user_message})

            # Paso 1: Extraer datos del mensaje actual
            llm_response = self.extract_data_with_llm(user_message)
            self.update_data_from_llm_response(llm_response)

            # Paso 2: Preparar contexto adicional si es relevante
            enriched_message = user_message

            # ¿Tenemos terapia, pero no fecha/hora? → Añadir disponibilidad
            if "terapia" in self.user_data and "fecha" not in self.user_data and "hora" not in self.user_data:
                terapia = self.user_data["terapia"].lower().strip()
                activity_id = THERAPY_TO_ACTIVITY_ID.get(terapia)
                if activity_id:
                    available = get_available_dates_for_therapy(activity_id, days_ahead=3)
                    if available:
                        disp_lines = [f"{fecha}: {', '.join(horas[:3])}" for fecha, horas in list(available.items())[:2]]
                        enriched_message += f" [DISPONIBILIDAD_ACTUAL: {'; '.join(disp_lines)}]"
                    else:
                        enriched_message += " [DISPONIBILIDAD_ACTUAL: no hay citas en los próximos 3 días]"

            # ¿Tenemos los 3 datos? → Verificar disponibilidad real
            elif self.is_data_complete():
                data = self.user_data
                terapia = data['terapia'].lower().strip()
                activity_id = THERAPY_TO_ACTIVITY_ID.get(terapia)
                if activity_id:
                    try:
                        fecha_dt = datetime.strptime(data['fecha'], "%d/%m/%y")
                        fecha_iso = fecha_dt.strftime("%Y-%m-%d")
                        hora_str = data['hora']
                        if ':' in hora_str:
                            h, m = hora_str.split(':')
                            hora_str = f"{int(h):02d}:{int(m):02d}"
                        else:
                            hora_str = f"{int(hora_str):02d}:00"

                        slot_id = find_timp_slot(activity_id, fecha_iso, hora_str)
                        if slot_id:
                            enriched_message += " [CONFIRMACIÓN_CITA: disponible]"
                        else:
                            enriched_message += " [ERROR_CITA: no hay disponibilidad en esa fecha/hora]"
                    except Exception as e:
                        enriched_message += " [ERROR_CITA: formato de fecha u hora inválido]"
                else:
                    enriched_message += " [ERROR_CITA: terapia no disponible]"

            # Paso 3: Generar respuesta final usando el mensaje enriquecido
            final_llm_response = self.extract_data_with_llm(enriched_message)
            self.update_data_from_llm_response(final_llm_response)

            try:
                response_data = json.loads(final_llm_response)
                final_response = response_data.get("respuesta", "¿Podrías repetirlo, por favor? 😊")
            except json.JSONDecodeError:
                final_response = "Vaya, tuve un fallo técnico. ¿Me lo dices de nuevo? 😅"

            THERAPY_TO_ACTIVITY_ID = {
                "ondas": 94797,
                "fisioterapia": 72574,
                "indiba": 72573,
                "láser": 94798,
                "osteopatía": 72576,
            }

            slot_id_for_response = None
            if self.is_data_complete():
                data = self.user_data
                terapia = data['terapia'].lower().strip()
                activity_id = THERAPY_TO_ACTIVITY_ID.get(terapia)
                if activity_id:
                    try:
                        fecha_dt = datetime.strptime(data['fecha'], "%d/%m/%y")
                        fecha_iso = fecha_dt.strftime("%Y-%m-%d")
                        hora_str = data['hora']
                        if ':' in hora_str:
                            h, m = hora_str.split(':')
                            hora_str = f"{int(h):02d}:{int(m):02d}"
                        else:
                            hora_str = f"{int(hora_str):02d}:00"
                        slot_id_for_response = find_timp_slot(activity_id, fecha_iso, hora_str)
                    except Exception as e:
                        print(f"Error al verificar disponibilidad final: {e}")
                        slot_id_for_response = None

            # Si hay slot_id real, añadir enlace al final del mensaje
            if slot_id_for_response:
                BRANCH_BUILDING_ID = "11269"
                cita_url = f"https://web.timp.pro/home/{BRANCH_BUILDING_ID}#/home/{BRANCH_BUILDING_ID}/branch_building/admissions/{slot_id_for_response}"
                final_response = final_response.rstrip(" .") + f"\n\n✅ **¡Listo!** Haz clic aquí para confirmar tu cita: {cita_url}"

            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response


        except Exception as e:
            print(f"Error inesperado en send_message: {str(e)}")
            fallback = "Vaya, tuve un pequeño fallo técnico. ¿Podrías repetirme eso, por favor? 😅"
            self.conversation_history.append({"role": "assistant", "content": fallback})
            return fallback

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

