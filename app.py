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
            print(f"⚠️ Error checking date {check_date}: {e}")
            continue

    return available

def clean_llm_response(text: str) -> str:
    """
    Elimina bloques <think>...</think> del texto generado por el LLM.
    También elimina cualquier línea que empiece con <think> si no está bien cerrada.
    """
    # Elimina bloques completos <think> ... </think>
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
                    "Tu única tarea es extraer 3 datos del mensaje del usuario y devolverlos en JSON:\n"
                    "- **terapia**: uno de: Ondas, Fisioterapia, Indiba, Láser, Osteopatía.\n"
                    "- **fecha**: SIEMPRE en formato dd/mm/yy (ej: 26/09/25). Si el usuario dice 'el 26', '26/09', 'mañana', etc., "
                    f"usa el contexto de HOY para inferir mes y año. Si la fecha ya pasó, usa el próximo mes o año. "
                    "NUNCA devuelvas '26/09' sin año. Siempre incluye el año en 2 dígitos.\n"
                    "- **hora**: SIEMPRE en formato HH:MM con 24h y ceros iniciales (ej: 08:00, 12:45). "
                    "Si el usuario dice 'a las 8', conviértelo a '08:00'.\n\n"
                    "IMPORTANTE: Tu respuesta debe ser un JSON EXACTO con este formato:\n"
                    "{\n"
                    '  "respuesta": "mensaje amable al usuario",\n'
                    '  "data": {\n'
                    '    "fecha": "?",\n'
                    '    "hora": "?",\n'
                    '    "terapia": "?"\n'
                    "  }\n"
                    "}\n"
                    "Llena los campos que puedas. Usa ? si no hay info. Cuando los 3 estén listos, confirma con alegría."
                )
            }
        ]
    
        groq_api_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=groq_api_key)

    def is_data_complete(self):
        required = ["fecha", "hora", "terapia"]
        is_complete = all(key in self.user_data for key in required)
        
        print(f"Verificando datos completos: {is_complete}")
        print(f"Datos actuales: {self.user_data}")
        print(f"Faltantes: {[key for key in required if key not in self.user_data]}")
    
        return is_complete

    def extract_data_with_llm(self, user_message):
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
                response_format={"type": "json_object"}
            )
            
            raw_content = chat_completion.choices[0].message.content
            cleaned_content = clean_llm_response(raw_content)
            return cleaned_content
        except Exception as e:
            return f"Lo siento, tuve un problema técnico. ¿Podrías repetirlo, por favor? 😅"

    def update_data_from_llm_response(self, llm_response):
        try:
            print(f"Respuesta LLM para extracción: {llm_response}")
            
            # Limpia la respuesta antes de parsear JSON
            cleaned_response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', llm_response)
            cleaned_response = cleaned_response.strip()
            
            # Intenta parsear como JSON
            data = json.loads(cleaned_response)
            print(f"JSON parseado: {data}")
            
            # Extrae datos si están presentes
            if 'data' in data and isinstance(data['data'], dict):
                user_data = data['data']
                print(f"Datos extraídos: {user_data}")
                
                for key in ["fecha", "hora", "terapia"]:
                    if key in user_data and user_data[key] and user_data[key] != "?":
                        self.user_data[key] = str(user_data[key]).strip()
                        print(f"Guardado {key}: {user_data[key]}")
                    
        except json.JSONDecodeError as e:
            print(f"No se pudo decodificar JSON: {e}")
            print(f"Contenido que falló: {llm_response}")
        except Exception as e:
            print(f"Error en update_data: {e}")

    def generate_summary(self):
        print(f"GENERATE_SUMMARY llamado con datos: {self.user_data}")
        
        if not self.is_data_complete():
            return "Error: Datos incompletos para generar resumen"
        
        data = self.user_data

        # Mapeo de terapias
        THERAPY_TO_ACTIVITY_ID = {
            "ondas": 94797,
            "fisioterapia": 72574,
            "indiba": 72573,
            "láser": 94798,
            "osteopatía": 72576,
        }

        terapia = data['terapia'].lower().strip()
        activity_id = THERAPY_TO_ACTIVITY_ID.get(terapia)

        if not activity_id:
            return f"Lo siento, no ofrecemos '{data['terapia']}' en este centro. ¿Te gustaría probar con Ondas, Fisioterapia, Indiba, Láser u Osteopatía? 😊"

        # Convertir fecha al formato YYYY-MM-DD
        try:
            fecha_dt = datetime.strptime(data['fecha'], "%d/%m/%y")
            fecha_iso = fecha_dt.strftime("%Y-%m-%d")
            hora_str = data['hora']
            
            if ':' in hora_str:
                h, m = hora_str.split(':')
                hora_str = f"{int(h):02d}:{int(m):02d}"
            else:
                hora_str = f"{int(hora_str):02d}:00"
                
        except Exception as e:
            print(f"Error al parsear fecha/hora: {e}")
            return (
                "❌ Ups, no pude entender bien la fecha u hora. "
                "¿Podrías decírmelo como '26/04/25 a las 08:00'? 😊"
            )

        # Buscar slot disponible
        slot_id = find_timp_slot(activity_id, fecha_iso, hora_str)

        if not slot_id:
            return (
                f"Lo siento, no hay disponibilidad para {data['terapia']} el {data['fecha']} a las {data['hora']}.\n"
                "¿Te gustaría probar con otra hora o fecha? 😊"
            )

        #construir el mensaje final
        BRANCH_BUILDING_ID = "11269"
        cita_url = f"https://web.timp.pro/home/{BRANCH_BUILDING_ID}#/home/{BRANCH_BUILDING_ID}/branch_building/admissions/{slot_id}"

        # Construir el resumen completo
        summary = (
            f"¡Disponibilidad encontrada! 🎉\n\n"
            f"Fecha: {data['fecha']} a las {data['hora']}\n"
            f"Terapia: {data['terapia']}\n\n"
            f"**¡Listo! Haz clic aquí para reservar tu cita directamente**:\n"
            f"{cita_url}\n\n"
            f"Solo te tomará unos segundos. ¡Te esperamos!"
        )

        return summary

    def send_message(self, user_message: str) -> str:
        try:
            # Si es el primer mensaje del usuario (historial solo tiene system)
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
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                return response_text

            # Añadir mensaje del usuario al historial
            self.conversation_history.append({"role": "user", "content": user_message})

            # Obtener respuesta del LLM
            llm_response = self.extract_data_with_llm(user_message)
            print(f"🤖 RESPUESTA CRUDA DE GROQ: {llm_response}")

            # Extraer datos estructurados
            self.update_data_from_llm_response(llm_response)

            # Intentar extraer 'respuesta' del JSON del LLM
            try:
                response_data = json.loads(llm_response)
                final_response = response_data.get("respuesta", "Gracias por la información. ¿Hay algo más en lo que pueda ayudarte? 😊")
            except json.JSONDecodeError:
                final_response = "Gracias por tu mensaje. Déjame ayudarte 😊"

            #Si el usuario dio terapia pero no fecha/hora → mostrar disponibilidad (máx. 2 días, 3-5 horas)
            if "terapia" in self.user_data and "fecha" not in self.user_data and "hora" not in self.user_data:
                terapia = self.user_data["terapia"].lower().strip()
                THERAPY_TO_ACTIVITY_ID = {
                    "ondas": 94797,
                    "fisioterapia": 72574,
                    "indiba": 72573,
                    "láser": 94798,
                    "osteopatía": 72576,
                }
                activity_id = THERAPY_TO_ACTIVITY_ID.get(terapia)

                if activity_id:
                    available_dates = get_available_dates_for_therapy(activity_id, days_ahead=7)
                    if available_dates:
                        # Tomar solo los primeros 2 días
                        limited_dates = list(available_dates.items())[:2]
                        disponibilidad_lines = []
                        for fecha, horas in limited_dates:
                            # Tomar hasta 5 horas
                            horas_limited = horas[:5]
                            disponibilidad_lines.append(f"📅 **{fecha}** → {', '.join(horas_limited)}")
                        disponibilidad = "\n".join(disponibilidad_lines)

                        final_response = (
                            f"¡Genial elección! *{terapia.title()}* es una excelente opción.\n\n"
                            f"Estas son las próximas fechas con disponibilidad:\n\n"
                            f"{disponibilidad}\n\n"
                            "¿Qué fecha y hora te gustaría reservar? Puedes decirme algo como:\n"
                            "“El 10/04 a las 17:00” o “Mañana a las 9:00”"
                        )
                    else:
                        final_response = f"Lo siento, no hay disponibilidad para {terapia} en los próximos días. ¿Quieres que revise más adelante o probar con otra terapia? 🤔"
                else:
                    final_response = f"❌ No tengo configurada la terapia '{terapia}'. ¿Puedes elegir entre: Ondas, Fisioterapia, Indiba, Láser u Osteopatía? 😊"

            # Si los 3 datos están completos → generar enlace
            elif self.is_data_complete():
                print("✅ Datos completos detectados - generando enlace de reserva")
                final_response = self.generate_summary()

            # Añadir respuesta del asistente al historial
            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response

        except Exception as e:
            print(f"Error inesperado en send_message: {str(e)}")
            return "Lo siento, tuve un pequeño error técnico. ¿Podrías repetirme eso con más calma? 😅"    

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

