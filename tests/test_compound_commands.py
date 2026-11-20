"""
Tests for Dynamic Compound Command Decomposition and Multi-Intent Execution.
Verifies that compound requests with multiple sub-actions chained by conjunctions
(and, and also, and then, semicolons) are cleanly decomposed, executed sequentially,
and unified into a composite natural response.
"""
import pytest
from app.maddy_assistant import MaddyAssistant

@pytest.fixture(scope="module")
def assistant():
    from unittest.mock import MagicMock
    inst = MaddyAssistant()
    mock_plat = MagicMock()
    mock_plat.set_volume.return_value = True
    mock_plat.get_volume.return_value = 50
    mock_plat.set_brightness.return_value = True
    mock_plat.get_brightness.return_value = 50
    inst.platform = mock_plat
    return inst

def test_compound_command_decomposition(assistant):
    # 1. Multi-intent chained command
    cmd1 = 'set volume at max and play "Petta Parak" on spotify, and also find a good game of thrones inspired wallpaper from pintrest'
    parts1 = assistant.decompose_compound_request(cmd1)
    assert len(parts1) == 3
    assert parts1[0] == "set volume at max"
    assert parts1[1] == 'play "Petta Parak" on spotify'
    assert "game of thrones" in parts1[2]

    # 2. Hardware chained command
    cmd2 = "increase volume to 80% and dim the screen to 30%"
    parts2 = assistant.decompose_compound_request(cmd2)
    assert len(parts2) == 2
    assert parts2[0] == "increase volume to 80%"
    assert parts2[1] == "dim the screen to 30%"

    # 3. Guard: single-action site search must NOT be split
    cmd3 = "open youtube and search for most of the best"
    parts3 = assistant.decompose_compound_request(cmd3)
    assert len(parts3) == 1
    assert parts3[0] == cmd3

    # 4. Guard: product filter with multiple specs must NOT be split
    cmd4 = "find a phone with 8gb ram and 128gb storage under 20k"
    parts4 = assistant.decompose_compound_request(cmd4)
    assert len(parts4) == 1
    assert parts4[0] == cmd4

    # 5. Guard: quotes with 'and' inside must NOT be split
    cmd5 = 'play "Love and Rockets" on spotify'
    parts5 = assistant.decompose_compound_request(cmd5)
    assert len(parts5) == 1
    assert parts5[0] == cmd5

def test_compound_execution_end_to_end(assistant):
    cmd = 'set volume at max and play "Petta Parak" on spotify, and also find a good game of thrones inspired wallpaper from pintrest'
    res = assistant.execute(cmd, speak_response=False)
    assert res["success"] is True
    assert len(res["sub_commands"]) == 3
    assert len(res["sub_results"]) == 3
    assert all(sr["success"] for sr in res["sub_results"])
    assert "volume" in res["response"].lower()
    assert len(res["response"]) > 20
