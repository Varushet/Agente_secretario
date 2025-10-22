from flask import Flask, request, jsonify
import os.path
from datetime import datetime, timedelta
import requests
import json
import re
import secrets
from groq import Groq
from dotenv import load_dotenv
from dateparser import parse
from datetime import datetime, time


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

def get_available_dates_for_therapy(
    activity_id: int, 
    start_offset: int = 0, 
    end_offset: int = 6
) -> dict:
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
    for i in range(start_offset, end_offset + 1):
        if i < 0:
            continue
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

def interpret_date_range(user_message: str, today: datetime) -> tuple[int, int]:
    """
    Interpreta frases como "la semana que viene", "el miércoles que viene", etc.
    Retorna (start_days_from_today, end_days_from_today)
    """
    msg = user_message.lower()
    weekday_names = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    
    # Caso: "la semana que viene" → días 7 a 13
    if any(phrase in msg for phrase in ["semana que viene", "próxima semana", "semana próxima"]):
        return 7, 13

    # Caso: "el [día] que viene"
    for i, day_name in enumerate(weekday_names):
        if f"{day_name} que viene" in msg or f"próximo {day_name}" in msg:
            # Días hasta el próximo X
            days_ahead = (i - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7  # si es hoy, ir al próximo
            return days_ahead - 1, days_ahead + 1  # ±1 día de margen

    # Caso: "en X días"
    import re
    match = re.search(r"en\s+(\d+)\s*d[ií]as", msg)
    if match:
        n = int(match.group(1))
        return n - 1, n + 2

    # Caso: "del 20 al 25" → convertir a offsets
    match = re.search(r"del\s+(\d+)\s+al\s+(\d+)", msg)
    if match:
        try:
            day1, day2 = int(match.group(1)), int(match.group(2))
            # Asumir mismo mes (simplificación)
            date1 = today.replace(day=day1)
            date2 = today.replace(day=day2)
            if date1 < today:
                date1 = date1.replace(month=date1.month + 1)
            if date2 < today:
                date2 = date2.replace(month=date2.month + 1)
            start_offset = (date1 - today).days
            end_offset = (date2 - today).days
            return max(0, start_offset), min(21, end_offset)  # límite 3 semanas
        except:
            pass

    # Por defecto: próximos 7 días
    return 0, 6

def normalize_date_string(date_str: str, today: datetime = None) -> str:
    """
    Convierte 'dd/mm', 'dd/mm/yy' o 'dd/mm/yyyy' a 'dd/mm/yy'.
    Si es 'dd/mm', asume el año actual (o próximo si la fecha ya pasó).
    """
    if today is None:
        today = datetime.today()

    parts = date_str.split('/')
    day, month = parts[0], parts[1]

    if len(parts) == 2:
        # Solo dd/mm → adivinar año
        try:
            current_year = today.year
            # Intentar con año actual
            candidate = datetime(current_year, int(month), int(day))
            # Si ya pasó, usar próximo año
            if candidate.date() < today.date():
                candidate = datetime(current_year + 1, int(month), int(day))
            return candidate.strftime("%d/%m/%y")
        except ValueError:
            raise ValueError("Fecha inválida")
    elif len(parts) == 3:
        year = parts[2]
        if len(year) == 2:
            return f"{day}/{month}/{year}"
        elif len(year) == 4:
            return f"{day}/{month}/{year[2:]}"
        else:
            raise ValueError("Año inválido")
    else:
        raise ValueError("Formato de fecha no reconocido")

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
            {"id": 110000, "name": "Tratamiento Láser"}
        ]
    },
    "osteopatía": {
        "first_visit": {"id": 72651, "name": "Osteopatía 1ª visita"},
        "options": [
            {"id": 72576, "name": "Osteopatía"}
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
                        f"Hoy es {today_str}. Eres SecretarioAI, un asistente empático de agendamiento.\n\n"
                        "REGLAS ESTRICAS:\n"
                        "- Responde SIEMPRE en JSON válido con este formato:\n"
                        '{\n'
                        '  "respuesta": "mensaje amable en español",\n'
                        '  "data": {\n'
                        '    "terapia": "?",\n'
                        '    "subopcion": "?",\n'
                        '    "fecha": "?",\n'
                        '    "hora": "?"\n'
                        '  }\n'
                        '}\n\n'
                        "INSTRUCCIONES:\n"
                        "- Convierte CUALQUIER expresión de fecha/hora a formato estándar, usando HOY como base:\n"
                        "  • 'el 27 a las 8' → fecha='27/10/25', hora='08:00'\n"
                        "  • 'mañana' → fecha='17/10/25', hora='10:00'\n"
                        "  • 'pasado mañana por la tarde' → fecha='18/10/25', hora='17:00'\n"
                        "  • 'la semana que viene' → fecha='24/10/25', hora='10:00'\n"
                        "  • 'el martes que viene' → fecha='21/10/25', hora='10:00'\n"
                        "  • 'a las 3' → hora='15:00'\n"
                        "- Formato de salida:\n"
                        "  • fecha: SIEMPRE dd/mm/yy (ej: 27/10/25)\n"
                        "  • hora: SIEMPRE HH:MM (ej: 08:00)\n"
                        "- terapia: uno de: Ondas, Fisioterapia, Indiba, Láser, Osteopatía.\n"
                        "- subopcion: nombre EXACTO de la opción (ej: \"Tratamiento Laser\", no \"tratamiento láser\").\n"
                        "- Si el usuario dice 'láser' o 'Láser', normaliza a 'Láser'.\n"
                        "- Si falta algo, pregunta con empatía en 'respuesta', y deja los campos como '?'.\n"
                        "- **NUNCA digas 'formato inválido', 'error', ni nada técnico.**\n"
                        "- **NUNCA inventes enlaces.**"
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
        print(f"[DEBUG] 🧠 Estado actual de user_data: {self.user_data}")

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

        # Añadir mensaje al historial
        self.conversation_history.append({"role": "user", "content": user_message})

        # Extraer datos del LLM
        llm_response = self.extract_data_with_llm(user_message)
        print(f"[DEBUG] 🤖 Respuesta LLM (raw): {llm_response}")

        try:
            parsed = json.loads(clean_llm_response(llm_response))
            data = parsed.get("data", {})
            reply = parsed.get("respuesta", "¿Podrías repetirlo?")
            print(f"[DEBUG] 📦 Datos extraídos del LLM: {data}")
        except Exception as e:
            print(f"[ERROR] ❌ JSON inválido: {e}")
            reply = "Vaya, tuve un fallo técnico. ¿Me lo dices de nuevo? 😅"
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply

        # Guardar estado anterior para comparar
        prev_data = self.user_data.copy()

        # Actualizar solo campos válidos
        for key in ["terapia", "subopcion", "fecha", "hora"]:
            val = data.get(key)
            if val and val != "?":
                self.user_data[key] = val.strip()

                # Normalización especial para "subopcion"
        val = data.get("subopcion")
        if val and val != "?":
            # Normalizar: quitar acentos, estandarizar formato
            import unicodedata
            normalized_val = ''.join(c for c in unicodedata.normalize('NFD', val) if unicodedata.category(c) != 'Mn')
            normalized_val = normalized_val.strip().title()

            # Mapeo manual para coincidir EXACTAMENTE con THERAPY_OPTIONS
            lower_val = normalized_val.lower()
            if lower_val in ["tratamiento laser", "tratamiento láser", "laser tratamiento", "tratamiento con laser", "tratamiento"]:
                normalized_val = "Tratamiento Láser"
            elif lower_val in ["fisioterapia 1a visita", "fisio primera", "primera visita fisio", "fisioterapia primera", "primera fisio"]:
                normalized_val = "Fisioterapia 1ª visita"
            elif lower_val in ["osteopatia 1a visita", "osteopatía primera", "primera osteo"]:
                normalized_val = "Osteopatía 1ª visita"
            elif lower_val in ["ondas focales", "ondas focales tratamiento", "focales"]:
                normalized_val = "Tratamiento Ondas Focales"
            elif lower_val in ["ondas radiales", "tratamiento ondas radiales", "radiales"]:
                normalized_val = "Tratamiento Ondas Radiales"
            elif lower_val in ["indiba 45", "indiba 45 minutos", "indiba"]:
                normalized_val = "Indiba 45'"
            elif lower_val in ["indiba laser", "indiba + laser", "indiba y laser", "doble"]:
                normalized_val = "Indiba + Láser"
            elif lower_val in ["fisio+indiba+laser", "fisio indiba laser", "triple"]:
                normalized_val = "Fisio+Indiba+Láser"

            self.user_data["subopcion"] = normalized_val

        # Mostrar qué cambió
        if self.user_data != prev_data:
            print(f"[DEBUG] ✅ user_data ACTUALIZADO: {self.user_data}")
        else:
            print(f"[DEBUG] ➖ user_data SIN CAMBIOS: {self.user_data}")

        terapia = self.user_data.get("terapia", "").lower()
        subopcion = self.user_data.get("subopcion")

        # --- Paso 1: Terapia seleccionada, pero sin subopción ---
        if terapia and not subopcion:
            print(f"[DEBUG] 🚶‍♂️ Entrando en PASO 1: terapia='{terapia}', subopcion no definida")
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

        # --- Paso 2: Subopción seleccionada → disponibilidad dinámica ---
        if terapia and subopcion and "fecha" not in self.user_data:
            print(f"[DEBUG] 🚶‍♂️ Entrando en PASO 2: terapia='{terapia}', subopcion='{subopcion}'")
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
                print(f"[DEBUG] ❌ Subopción no encontrada: '{subopcion}'")
                self.user_data.pop("subopcion", None)
                return "Opción no reconocida."

            today = datetime.today()
            start_off, end_off = interpret_date_range(user_message, today)
            print(f"[DEBUG] 📅 Rango de búsqueda: hoy+{start_off} a hoy+{end_off} días")

            available = get_available_dates_for_therapy(activity_id, start_offset=start_off, end_offset=end_off)

            if not available:
                return "No hay disponibilidad en el periodo solicitado. ¿Quieres intentar con otro rango?"

            filtered = {d: t for d, t in available.items() if len(t) >= 2}
            lines = [f"• **{date}**: {', '.join(times[:5])}" for date, times in list(filtered.items())[:4]]
            msg = "Elige una fecha y hora (ej: '20/10 a las 09:15'):\n" + "\n".join(lines)
            self.conversation_history.append({"role": "assistant", "content": msg})
            return msg

               # --- Paso 3: Fecha y hora seleccionadas → generar enlace ---
        elif terapia and subopcion and self.user_data.get("fecha") and self.user_data.get("hora"):
            print(f"[DEBUG] 🎯 Entrando en PASO 3: ¡Todos los datos completos!")
            
            # Buscar activity_id
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
                reply = "Lo siento, no encontré esa opción. ¿Podrías repetirme la terapia y el tipo de cita?"
                self.user_data.clear()
                self.conversation_history.append({"role": "assistant", "content": reply})
                return reply

            fecha_str = self.user_data["fecha"]
            hora_str = self.user_data["hora"]

            try:
                # Convertir a formato ISO para la API
                fecha_dt = datetime.strptime(fecha_str, "%d/%m/%y")
                fecha_iso = fecha_dt.strftime("%Y-%m-%d")
                
                # Asegurar formato HH:MM
                h, m = hora_str.split(':')
                hora_norm = f"{int(h):02d}:{int(m):02d}"
                
                print(f"[DEBUG] 📅 Fecha/hora para API: {fecha_iso} a las {hora_norm}")
            except Exception as e:
                print(f"[ERROR] El LLM no entregó fecha/hora en formato válido: fecha='{fecha_str}', hora='{hora_str}' | Error: {e}")
                reply = "Vaya, tuve un pequeño fallo al procesar tu cita. ¿Podrías repetirme la fecha y hora? Por ejemplo: 'el 27 a las 8' o 'mañana por la tarde'."
                self.conversation_history.append({"role": "assistant", "content": reply})
                return reply

            # Verificar disponibilidad real
            slot_id = find_timp_slot(activity_id, fecha_iso, hora_norm)
            if not slot_id:
                reply = "Lo siento, ese horario ya no está disponible. ¿Te gustaría proponer otro?"
                # No reiniciar: permitir corregir solo fecha/hora
                self.conversation_history.append({"role": "assistant", "content": reply})
                return reply

            BRANCH_BUILDING_ID = "11269"
            cita_url = f"https://web.timp.pro/home/{BRANCH_BUILDING_ID}#/home/{BRANCH_BUILDING_ID}/branch_building/admissions/{slot_id}"
            print(f"[DEBUG] 🔗 Enlace generado: {cita_url}")
            msg = f"✅ **¡Listo!** Haz clic aquí para confirmar tu cita: {cita_url}\n\n¿Te gustaría agendar otra cita? 😊"

            # Reiniciar estado tras éxito
            self.user_data = {}
            print(f"[DEBUG] 🧹 user_data REINICIADO tras cita exitosa")
            self.conversation_history.append({"role": "assistant", "content": msg})
            return msg

        # --- Por defecto: responder con el mensaje del LLM ---
        print(f"[DEBUG] 💬 Respondiendo con el mensaje del LLM: {reply}")
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

