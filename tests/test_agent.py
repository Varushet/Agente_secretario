import pytest
from app import NaturalAppointmentAgent
from unittest.mock import patch


@pytest.fixture
def agent():
    """Crea un agente nuevo para cada test"""
    return NaturalAppointmentAgent(model_name="fake-model")

def test_agent_init(agent):
    assert len(agent.conversation_history) == 1
    assert agent.conversation_history[0]["role"] == "system"
    assert agent.user_data == {}

def test_is_data_complete_vacio(agent):
    assert agent.is_data_complete() is False

def test_is_data_complete_lleno(agent):
    agent.user_data = {
        "nombre": "Juan",
        "apellido": "Pérez",
        "telefono": "123456789",
        "email": "juan@example.com",
        "fecha": "25/12/25",
        "hora": "16:30",
        "motivo": "Chequeo médico"
    }
    assert agent.is_data_complete() is True

def test_is_data_complete_faltante(agent):
    agent.user_data = {
        "nombre": "Juan",
        "apellido": "Pérez",
        "telefono": "123456789",
    }
    assert agent.is_data_complete() is False


@patch('app.ollama.chat')
def test_extract_data_with_llm(mock_ollama_chat, agent):
    mock_ollama_chat.return_value = {
        'message': {
            'content': '{"data": {"nombre": "Ana", "apellido": "Gómez"}}'
        }
    }

    response = agent.extract_data_with_llm("Me llamo Ana Galindo")

    assert response == '{"data": {"nombre": "Ana", "apellido": "Gómez"}}'

    mock_ollama_chat.assert_called_once()