import pytest
from unittest.mock import MagicMock, patch
from intelligence.agent_service import NutritionAgent

@pytest.fixture
def agent():
    return NutritionAgent(user_id="test_user")

def test_agent_initialization(agent):
    assert agent.user_id == "test_user"
    assert len(agent.tools) == 5
    assert any(tool.name == "nutrition_lookup" for tool in agent.tools)

@patch("intelligence.agent_service.get_rag_service")
def test_nutrition_lookup_tool(mock_rag_service, agent):
    mock_rag = MagicMock()
    mock_rag.lookup.return_value = {"calories": 100, "protein_g": 5}
    mock_rag_service.return_value = mock_rag
    
    # Access the tool directly from the agent
    lookup_tool = next(t for t in agent.tools if t.name == "nutrition_lookup")
    result = lookup_tool.invoke({"food_name": "apple"})
    
    assert "calories': 100" in result
    mock_rag.lookup.assert_called_with("apple")

@patch("intelligence.agent_service.DailyLogRepository.get_or_create")
def test_get_daily_stats_tool(mock_get_or_create, agent):
    mock_get_or_create.return_value = {"calories_consumed": 1500}
    
    stats_tool = next(t for t in agent.tools if t.name == "get_daily_stats")
    result = stats_tool.invoke({"date": "2024-05-21"})
    
    assert "1500" in result
    mock_get_or_create.assert_called_with("test_user", "2024-05-21")

@patch("intelligence.agent_service.MedicalProfileRepository.get_by_user_id")
def test_get_medical_profile_tool(mock_get_profile, agent):
    mock_get_profile.return_value = {"conditions": ["Diabetes"]}
    
    profile_tool = next(t for t in agent.tools if t.name == "get_medical_profile")
    result = profile_tool.invoke({})
    
    assert "Diabetes" in result
    mock_get_profile.assert_called_with("test_user")
