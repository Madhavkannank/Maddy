"""
Filesystem tools for M.A.D.D.Y. ACI.
Provides filesystem.read and filesystem.write tools with safe execution handlers.
"""
import os
from typing import Any, Dict
from tools.base import RiskLevel, Tool

def _handle_filesystem_read(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args["path"]
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return {
            "success": False,
            "error": {
                "type": "FILE_NOT_FOUND",
                "message": f"File does not exist: '{path}'",
            },
        }

    if not os.path.isfile(abs_path):
        return {
            "success": False,
            "error": {
                "type": "NOT_A_FILE",
                "message": f"Path is not a regular file: '{path}'",
            },
        }

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "path": abs_path,
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "type": "READ_ERROR",
                "message": f"Failed to read file '{path}': {str(e)}",
            },
        }

def _handle_filesystem_write(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args["path"]
    content = args["content"]
    abs_path = os.path.abspath(path)

    try:
        parent_dir = os.path.dirname(abs_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "path": abs_path,
            "bytes_written": len(content.encode("utf-8")),
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "type": "WRITE_ERROR",
                "message": f"Failed to write file '{path}': {str(e)}",
            },
        }

filesystem_read_tool = Tool(
    name="filesystem.read",
    description="Read content from a text file at the specified file path.",
    parameter_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
        },
        "required": ["path"],
    },
    handler=_handle_filesystem_read,
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

filesystem_write_tool = Tool(
    name="filesystem.write",
    description="Write string content to a text file at the specified file path.",
    parameter_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Target path where content will be written.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write into the target file.",
            },
        },
        "required": ["path", "content"],
    },
    handler=_handle_filesystem_write,
    risk_level=RiskLevel.MEDIUM,
    timeout=10.0,
)

def _handle_filesystem_search(args: Dict[str, Any]) -> Dict[str, Any]:
    directory = args["directory"]
    query = args["query"]
    user_max_results = args.get("max_results", 50)
    user_max_depth = args.get("max_depth", 10)

    # Hard safety bounds
    max_results = min(max(1, int(user_max_results)), 200)
    max_depth = min(max(0, int(user_max_depth)), 20)

    abs_dir = os.path.abspath(directory)
    base_depth = abs_dir.rstrip(os.sep).count(os.sep)

    if not os.path.exists(abs_dir):
        return {
            "success": False,
            "error": {
                "type": "DIRECTORY_NOT_FOUND",
                "message": f"Directory does not exist: '{directory}'",
            },
        }

    if not os.path.isdir(abs_dir):
        return {
            "success": False,
            "error": {
                "type": "NOT_A_DIRECTORY",
                "message": f"Path is not a directory: '{directory}'",
            },
        }

    matches = []
    query_lower = query.lower()
    ignored_dirs = {".git", "node_modules", "__pycache__", ".venv", ".cache"}

    try:
        for root, dirs, files in os.walk(abs_dir, followlinks=False):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignored_dirs]

            # Enforce max depth bounding
            current_depth = root.rstrip(os.sep).count(os.sep) - base_depth
            if current_depth >= max_depth:
                dirs.clear()

            for file_name in files:
                if query_lower in file_name.lower():
                    matches.append(os.path.join(root, file_name))
                    if len(matches) >= max_results:
                        break

            if len(matches) >= max_results:
                break

        return {
            "directory": abs_dir,
            "query": query,
            "matches": matches,
            "match_count": len(matches),
            "max_depth": max_depth,
            "max_results": max_results,
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "type": "SEARCH_ERROR",
                "message": f"Failed to search directory '{directory}': {str(e)}",
            },
        }

filesystem_search_tool = Tool(
    name="filesystem.search",
    description="Search for files by filename query within a directory with bounded recursive traversal.",
    parameter_schema={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Base directory path to search within.",
            },
            "query": {
                "type": "string",
                "description": "Filename pattern or substring to search for.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (capped at 200).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory recursion depth (default 10, capped at 20).",
            },
        },
        "required": ["directory", "query"],
    },
    handler=_handle_filesystem_search,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

