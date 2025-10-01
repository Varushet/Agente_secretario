import pytest
from unittest.mock import patch, MagicMock
from app import NaturalAppointmentAgent, clean_llm_response


@pytest.fixture
def agent():
    """Crea un agente con modelo falso y sin llamadas reales a Groq"""
    with patch('app.Groq') as mock_groq_class:
        # Simular cliente de Groq
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        agent = NaturalAppointmentAgent(model_name="fake-model")
        agent._mock_client = mock_client
    return agent


# === Tests de inicialización ===

def test_agent_init(agent):
    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0]["role"] == "system"
    assert "JSON" in agent.conversation_history[0]["content"]
    assert agent.user_data == {}


# === Tests de utilidad ===

def test_clean_llm_response_removes_think():
    text = "<think>razonando...</think> {\"respuesta\": \"hola\"}"
    cleaned = clean_llm_response(text)
    assert cleaned == '{"respuesta": "hola"}'

def test_clean_llm_response_handles_no_json():
    text = "Lo siento, no entendí"
    cleaned = clean_llm_response(text)
    assert cleaned == "Lo siento, no entendí"


# === Tests de extracción de datos ===

def test_extract_data_with_llm_success(agent):
    # Simular respuesta de Groq
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"respuesta": "ok", "data": {"terapia": "Láser"}}'
    agent._mock_client.chat.completions.create.return_value = mock_response

    result = agent.extract_data_with_llm("Quiero láser")

    assert result == '{"respuesta": "ok", "data": {"terapia": "Láser"}}'
    agent._mock_client.chat.completions.create.assert_called_once()

def test_extract_data_with_llm_error_fallback(agent):
    agent._mock_client.chat.completions.create.side_effect = Exception("API error")

    result = agent.extract_data_with_llm("Hola")

    assert "fallo técnico" in result


# === Tests de actualización de datos ===

def test_update_data_from_llm_response_valid(agent):
    llm_response = '{"data": {"terapia": "Ondas", "fecha": "01/01/25", "hora": "09:00"}}'
    agent.update_data_from_llm_response(llm_response)

    assert agent.user_data == {"terapia": "Ondas", "fecha": "01/01/25", "hora": "09:00"}

def test_update_data_from_llm_response_partial(agent):
    llm_response = '{"data": {"terapia": "Fisioterapia", "fecha": "?"}}'
    agent.update_data_from_llm_response(llm_response)

    assert agent.user_data == {"terapia": "Fisioterapia"}

def test_update_data_from_llm_response_invalid_json(agent):
    agent.update_data_from_llm_response("esto no es json")
    assert agent.user_data == {}


# === Tests de completitud ===

def test_is_data_complete_true(agent):
    agent.user_data = {"terapia": "Láser", "fecha": "01/01/25", "hora": "10:00"}
    assert agent.is_data_complete() is True

def test_is_data_complete_false_missing_fecha(agent):
    agent.user_data = {"terapia": "Láser", "hora": "10:00"}
    assert agent.is_data_complete() is False

def test_is_data_complete_false_empty(agent):
    assert agent.is_data_complete() is False


# === Tests de flujo de conversación (end-to-end simulado) ===

@patch('app.find_timp_slot')
@patch('app.get_available_dates_for_therapy')
def test_send_message_terapia_then_disponibilidad(mock_get_dates, mock_find_slot, agent):
    # Simular disponibilidad
    agent.send_message("Hola")
    mock_get_dates.return_value = {"01/10": ["08:00", "09:00"]}
    
    # Simular respuesta del LLM al decir "láser"
    mock_response1 = MagicMock()
    mock_response1.choices[0].message.content = '{"respuesta": "¿Qué fecha?", "data": {"terapia": "Láser"}}'
    agent._mock_client.chat.completions.create.return_value = mock_response1

    # Primer mensaje: "láser"
    response1 = agent.send_message("láser")
    assert "Láser" in agent.user_data.values()
    
    # Segundo mensaje: "qué disponibilidad hay?"
    # Simular que el LLM ahora responde con disponibilidad
    mock_response2 = MagicMock()
    mock_response2.choices[0].message.content = '{"respuesta": "Tenemos 01/10 a las 08:00", "data": {"terapia": "Láser"}}'
    agent._mock_client.chat.completions.create.return_value = mock_response2

    response2 = agent.send_message("qué disponibilidad hay?")
    assert "01/10" in response2 or "08:00" in response2

@patch('app.find_timp_slot')
def test_agendar_cita_nueva_despues_de_confirmar(mock_find_slot, agent):
    """
    Verifica que después de confirmar una cita, se pueda agendar otra desde cero.
    """
    # Cada cita requiere 2 llamadas a find_timp_slot (una para contexto, otra para enlace)
    # Primera cita: slot_123 (2 veces), Segunda cita: slot_456 (2 veces)
    mock_find_slot.side_effect = ["slot_123", "slot_123", "slot_456", "slot_456"]

    # === Paso 1: Saltar mensaje de bienvenida ===
    agent.send_message("Hola")

    # === Paso 2: Agendar primera cita (Ondas) ===
    mock_response1 = MagicMock()
    mock_response1.choices[0].message.content = (
        '{"respuesta": "Cita de Ondas confirmada", "data": {"terapia": "Ondas", "fecha": "01/01/25", "hora": "09:00"}}'
    )
    agent._mock_client.chat.completions.create.return_value = mock_response1

    response1 = agent.send_message("Ondas el 1/1 a las 9")
    
    # Verificar que la primera cita usa slot_123
    assert "slot_123" in response1
    assert "¿Te gustaría agendar otra cita" in response1
    assert agent.user_data == {}

    # === Paso 3: Agendar segunda cita (Fisioterapia) ===
    mock_response2 = MagicMock()
    mock_response2.choices[0].message.content = (
        '{"respuesta": "Cita de Fisioterapia confirmada", "data": {"terapia": "Fisioterapia", "fecha": "02/02/25", "hora": "10:00"}}'
    )
    agent._mock_client.chat.completions.create.return_value = mock_response2

    response2 = agent.send_message("Fisioterapia el 2/2 a las 10")
    
    # Verificar que la segunda cita usa slot_456
    assert "slot_456" in response2
    assert "¿Te gustaría agendar otra cita" in response2
    assert agent.user_data == {}
    
@patch('app.find_timp_slot')
def test_send_message_cita_confirmada_resetea_estado(mock_find_slot, agent):
    # Simular slot disponible
    agent.send_message("Hola")
    mock_find_slot.return_value = "slot_123"
    
    # Simular que el LLM extrae datos completos
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"respuesta": "Cita confirmada", "data": {"terapia": "Ondas", "fecha": "01/01/25", "hora": "09:00"}}'
    )
    agent._mock_client.chat.completions.create.return_value = mock_response

    response = agent.send_message("Ondas el 1/1 a las 9")

    # Verificar que se muestra el enlace
    assert "slot_123" in response
    assert "¿Te gustaría agendar otra cita" in response

    # Verificar que el estado se reseteó
    assert agent.user_data == {}


# === Tests de errores ===

def test_send_message_llm_json_error(agent):
    # Simular respuesta no-JSON del LLM
    agent.send_message("Hola")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Lo siento, no entendí"
    agent._mock_client.chat.completions.create.return_value = mock_response

    response = agent.send_message("Hola")

    assert "fallo técnico" in response or "repetirme" in response