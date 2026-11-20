"""
Tests for Conversational File Saving Lifecycle.
Verifies:
1. Detecting file save commands across text, json, python formats.
2. Mandatory prompting for save location before writing to disk.
3. Directory name resolution (without full paths, e.g. 'Desktop', 'Documents', workspace subfolders).
4. On-the-fly directory creation directives (e.g. 'create a dir called research and save inside it').
5. End-to-end multi-turn interaction with MaddyAssistant.
"""
import os
import shutil
import tempfile
import pytest

from core.file_save_flow import FileSaveManager, FileSaveState
from app.maddy_assistant import MaddyAssistant


def test_file_save_detection_and_details():
    mgr = FileSaveManager()
    assert mgr.is_save_command("save this info on a file") is True
    assert mgr.is_save_command("write this in a txt file and save it") is True
    assert mgr.is_save_command("save my meeting notes to a file called notes.txt") is True
    assert mgr.is_save_command("save this to a json file") is True
    assert mgr.is_save_command("what is the weather in Paris") is False

    details = mgr.extract_file_details("save 'hello world' to a file called greet.txt")
    assert details["filename"] == "greet.txt"
    assert details["content"] == "hello world"
    assert details["format"] == ".txt"

    details_json = mgr.extract_file_details("save this data to a json file")
    assert details_json["format"] == ".json"
    assert details_json["filename"].endswith(".json")


def test_conversational_file_save_flow(tmp_path):
    ws_dir = str(tmp_path / "workspace")
    os.makedirs(ws_dir, exist_ok=True)
    mgr = FileSaveManager(workspace_dir=ws_dir)

    # Turn 1: Trigger save command
    handled, prompt = mgr.handle_turn(
        "save 'Meeting at 3 PM' to a file called schedule.txt",
        address="boss",
    )
    assert handled is True
    assert "where would you like me to save 'schedule.txt'" in prompt.lower()
    assert mgr.state == FileSaveState.AWAITING_LOCATION

    # Turn 2: User provides folder name (not full path)
    sub_dir_name = "calendar_notes"
    handled2, confirm = mgr.handle_turn(sub_dir_name, address="boss")
    assert handled2 is True
    assert "saved 'schedule.txt'" in confirm.lower()
    assert mgr.state == FileSaveState.IDLE

    # Verify file saved on disk
    expected_path = os.path.join(ws_dir, sub_dir_name, "schedule.txt")
    assert os.path.exists(expected_path)
    with open(expected_path, "r", encoding="utf-8") as f:
        assert f.read() == "Meeting at 3 PM"


def test_create_directory_instruction_and_save(tmp_path):
    ws_dir = str(tmp_path / "workspace")
    os.makedirs(ws_dir, exist_ok=True)
    mgr = FileSaveManager(workspace_dir=ws_dir)

    # Turn 1: Save request
    mgr.handle_turn("save 'important research data' into a file called report.txt", address="sir")

    # Turn 2: User asks to create a directory and save inside it
    create_cmd = "create a dir called deep_learning and save inside it"
    handled, confirm = mgr.handle_turn(create_cmd, address="sir")
    assert handled is True
    assert "saved 'report.txt'" in confirm.lower()

    # Check directory and file exist
    expected_dir = os.path.join(ws_dir, "deep_learning")
    assert os.path.isdir(expected_dir)
    expected_file = os.path.join(expected_dir, "report.txt")
    assert os.path.exists(expected_file)
    with open(expected_file, "r", encoding="utf-8") as f:
        assert f.read() == "important research data"


def test_file_save_cancellation(tmp_path):
    mgr = FileSaveManager(workspace_dir=str(tmp_path))

    mgr.handle_turn("save this info on a file", address="boss")
    assert mgr.state == FileSaveState.AWAITING_LOCATION

    # Cancel
    handled, resp = mgr.handle_turn("cancel", address="boss")
    assert handled is True
    assert "cancelled" in resp.lower()
    assert mgr.state == FileSaveState.IDLE


def test_end_to_end_maddy_assistant_file_save(tmp_path):
    assistant = MaddyAssistant()
    # Point workspace to temp dir
    test_ws = str(tmp_path / "maddy_ws")
    os.makedirs(test_ws, exist_ok=True)
    assistant.file_save_manager.workspace_dir = test_ws

    # 1. Ask something first to establish recent context
    assistant.execute("What is 100 plus 250?")

    # 2. Command to save info to a file
    res1 = assistant.execute("save this info on a file")
    assert res1["success"] is True
    assert "where would you like me to save" in res1["response"].lower()

    # 3. Provide folder name
    res2 = assistant.execute("test_folder")
    assert res2["success"] is True
    assert "saved" in res2["response"].lower()

    # Verify file saved
    saved_file = os.path.join(test_ws, "test_folder", "notes.txt")
    assert os.path.exists(saved_file)
