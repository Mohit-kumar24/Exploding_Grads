"""
Terminal Color & Formatting Utilities

Provides ANSI color formatting with support for:
- Standard ANSI colors (red, green, blue, yellow, cyan, bold).
- Automatic NO_COLOR detection (https://no-color.org/).
- Non-interactive terminal detection (isatty fallback).
- Windows console VT100 / ANSI escape sequence activation.
- Resetting color state after every message.
"""
import os
import sys

# ANSI Escape Sequences
_ANSI_CODES = {
    "red": "\033[91m",
    "green": "\033[92m",
    "blue": "\033[94m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}

# Windows VT100 initialization flag
_WIN_VT_INITIALIZED = False


def _init_windows_console():
    """Enable Virtual Terminal Processing on Windows console if available."""
    global _WIN_VT_INITIALIZED
    if _WIN_VT_INITIALIZED or sys.platform != "win32":
        return
    _WIN_VT_INITIALIZED = True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def supports_color() -> bool:
    """
    Check if the terminal environment supports ANSI escape sequences.
    Respects NO_COLOR standard and TERM=dumb setting.
    """
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("FORCE_COLOR") in ("1", "true", "TRUE"):
        return True
    if sys.platform == "win32":
        _init_windows_console()
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colorize(text: str, color: str = None, bold: bool = False) -> str:
    """
    Wrap text in the specified ANSI color and reset codes.
    Returns plain text if colors are unsupported or disabled.
    """
    if not supports_color() or not color:
        return text
    prefix = ""
    if bold:
        prefix += _ANSI_CODES.get("bold", "")
    prefix += _ANSI_CODES.get(color.lower(), "")
    return f"{prefix}{text}{_ANSI_CODES['reset']}"


def cprint(text: str = "", color: str = None, bold: bool = False, **kwargs):
    """
    Print text formatted with ANSI colors, resetting immediately after.
    Flushes output by default to guarantee immediate visibility in PowerShell.
    """
    if "flush" not in kwargs:
        kwargs["flush"] = True
    if color:
        print(colorize(text, color, bold=bold), **kwargs)
    else:
        print(text, **kwargs)



def print_red(text: str = "", bold: bool = False, **kwargs):
    """Print text in red."""
    cprint(text, color="red", bold=bold, **kwargs)


def print_green(text: str = "", bold: bool = False, **kwargs):
    """Print text in green."""
    cprint(text, color="green", bold=bold, **kwargs)


def print_blue(text: str = "", bold: bool = False, **kwargs):
    """Print text in blue."""
    cprint(text, color="blue", bold=bold, **kwargs)


def print_yellow(text: str = "", bold: bool = False, **kwargs):
    """Print text in yellow."""
    cprint(text, color="yellow", bold=bold, **kwargs)


def print_cyan(text: str = "", bold: bool = False, **kwargs):
    """Print text in cyan."""
    cprint(text, color="cyan", bold=bold, **kwargs)

