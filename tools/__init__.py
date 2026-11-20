"""
Tools module for M.A.D.D.Y. Agent-Computer Interface (ACI).
"""
from tools.application import application_close_tool, application_open_tool
from tools.base import RiskLevel, Tool
from tools.call import ToolCall, ToolCallParseError, parse_tool_call
from tools.executor import ToolExecutor
from tools.filesystem import (
    filesystem_read_tool,
    filesystem_search_tool,
    filesystem_write_tool,
)
from tools.registry import ToolRegistry, ToolRegistryError
from tools.result import ToolResult
from tools.validator import ToolValidator, ValidationError

def register_default_tools(registry: ToolRegistry) -> None:
    """
    Register standard ACI tools into the provided ToolRegistry instance.
    """
    from browser.tools import (
        browser_extract_tool,
        browser_navigate_tool,
        browser_search_tool,
        browser_open_tool,
    )
    from computer.tools import (
        computer_click_tool,
        computer_screenshot_tool,
        computer_type_tool,
    )
    from tools.spotify import (
        spotify_get_playlists_tool,
        spotify_play_playlist_tool,
        spotify_play_tool,
        spotify_pause_tool,
        spotify_next_tool,
        spotify_previous_tool,
        spotify_current_playback_tool,
        spotify_shuffle_tool,
    )
    from vision.tools import (
        vision_analyze_image_tool,
        vision_locate_element_tool,
        vision_verify_visual_tool,
    )
    from voice.tools import (
        voice_speak_tool,
        voice_transcribe_tool,
        voice_listen_tool,
    )

    registry.register(filesystem_read_tool)
    registry.register(filesystem_write_tool)
    registry.register(filesystem_search_tool)
    registry.register(application_open_tool)
    registry.register(application_close_tool)
    registry.register(browser_navigate_tool)
    registry.register(browser_search_tool)
    registry.register(browser_extract_tool)
    registry.register(browser_open_tool)
    registry.register(computer_screenshot_tool)
    registry.register(computer_click_tool)
    registry.register(computer_type_tool)
    registry.register(spotify_get_playlists_tool)
    registry.register(spotify_play_playlist_tool)
    registry.register(spotify_play_tool)
    registry.register(spotify_pause_tool)
    registry.register(spotify_next_tool)
    registry.register(spotify_previous_tool)
    registry.register(spotify_current_playback_tool)
    registry.register(spotify_shuffle_tool)
    registry.register(vision_analyze_image_tool)
    registry.register(vision_locate_element_tool)
    registry.register(vision_verify_visual_tool)
    registry.register(voice_speak_tool)
    registry.register(voice_transcribe_tool)
    registry.register(voice_listen_tool)
    from tools.weather import register_weather_tools
    register_weather_tools(registry)
    from tools.hardware_tool import hardware_status_tool
    from tools.arxiv_digest import arxiv_search_tool
    from tools.messaging_tool import (
        telegram_send_tool,
        whatsapp_send_tool,
        contact_save_tool,
        contact_list_tool,
        contact_delete_tool,
    )
    registry.register(hardware_status_tool)
    registry.register(arxiv_search_tool)
    registry.register(telegram_send_tool)
    registry.register(whatsapp_send_tool)
    registry.register(contact_save_tool)
    registry.register(contact_list_tool)
    registry.register(contact_delete_tool)
    from tools.session_capture_tool import (
        session_capture_start_tool,
        session_capture_stop_tool,
        session_capture_status_tool,
    )
    registry.register(session_capture_start_tool)
    registry.register(session_capture_stop_tool)
    registry.register(session_capture_status_tool)

    from tools.calculator import calculator_tool
    registry.register(calculator_tool)

    from tools.memory_tool import memory_aci_tool
    registry.register(memory_aci_tool)

    from agents.core.skill_engine import SkillEngine
    for st in SkillEngine().get_skill_tools():
        registry.register(st)

    from tools.delegate_tool import delegate_task_tool
    registry.register(delegate_task_tool)

    from tools.session_search_tool import session_search_tool
    registry.register(session_search_tool)

    from tools.cron_tool import cronjob_manage_tool
    registry.register(cronjob_manage_tool)

    from vision.vision_tools import (
        screen_inspect_tool,
        active_window_inspect_tool,
        screen_interact_tool,
        vision_analyze_tool,
    )
    registry.register(screen_inspect_tool)
    registry.register(active_window_inspect_tool)
    registry.register(screen_interact_tool)
    registry.register(vision_analyze_tool)

__all__ = [
    "RiskLevel",
    "Tool",
    "ToolCall",
    "ToolCallParseError",
    "parse_tool_call",
    "ToolResult",
    "ToolValidator",
    "ValidationError",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolExecutor",
    "filesystem_read_tool",
    "filesystem_write_tool",
    "filesystem_search_tool",
    "application_open_tool",
    "application_close_tool",
    "browser_navigate_tool",
    "browser_search_tool",
    "browser_extract_tool",
    "computer_screenshot_tool",
    "computer_click_tool",
    "computer_type_tool",
    "memory_aci_tool",
    "register_default_tools",
]
