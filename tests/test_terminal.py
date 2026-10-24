"""
Tests for Terminal Color Formatting Helper (src/harness/terminal.py)
"""
import io
import os
import sys
import unittest
from unittest.mock import patch

from src.harness.terminal import (
    colorize,
    cprint,
    print_red,
    print_green,
    print_blue,
    print_yellow,
    print_cyan,
    supports_color,
    _ANSI_CODES,
)


class TestTerminal(unittest.TestCase):
    """Test suite for terminal color printing and ANSI formatting."""

    def test_colorize_with_force_color(self):
        """When color is enabled, colorize should wrap text in ANSI escape codes and reset."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            res = colorize("Test message", "red")
            self.assertTrue(res.startswith(_ANSI_CODES["red"]))
            self.assertTrue(res.endswith(_ANSI_CODES["reset"]))
            self.assertIn("Test message", res)

    def test_colorize_bold(self):
        """When bold is True, bold ANSI code is prepended before the color."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            res = colorize("Bold Green", "green", bold=True)
            self.assertTrue(res.startswith(_ANSI_CODES["bold"]))
            self.assertIn(_ANSI_CODES["green"], res)
            self.assertTrue(res.endswith(_ANSI_CODES["reset"]))

    def test_no_color_env_var(self):
        """When NO_COLOR is set, colorize must return plain unstyled text."""
        with patch.dict(os.environ, {"NO_COLOR": "1", "FORCE_COLOR": "1"}):
            self.assertFalse(supports_color())
            res = colorize("Plain Message", "red", bold=True)
            self.assertEqual(res, "Plain Message")
            self.assertNotIn("\033[", res)

    def test_term_dumb(self):
        """When TERM is dumb and FORCE_COLOR is unset, colorize must return plain text."""
        with patch.dict(os.environ, {"TERM": "dumb"}):
            os.environ.pop("FORCE_COLOR", None)
            os.environ.pop("NO_COLOR", None)
            self.assertFalse(supports_color())
            res = colorize("Plain Dumb Terminal", "blue")
            self.assertEqual(res, "Plain Dumb Terminal")
            self.assertNotIn("\033[", res)

    def test_print_helpers_output(self):
        """Test that helper functions write formatted text to stdout."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            os.environ.pop("NO_COLOR", None)

            helpers = [
                (print_red, "red"),
                (print_green, "green"),
                (print_blue, "blue"),
                (print_yellow, "yellow"),
                (print_cyan, "cyan"),
            ]
            for fn, color_name in helpers:
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    fn(f"Hello {color_name}")
                output = captured.getvalue()
                self.assertIn(_ANSI_CODES[color_name], output)
                self.assertIn(_ANSI_CODES["reset"], output)
                self.assertIn(f"Hello {color_name}", output)

    def test_cprint_without_color(self):
        """cprint with color=None behaves like standard print."""
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cprint("Plain output")
        self.assertEqual(captured.getvalue(), "Plain output\n")

    def test_reset_sequence_always_included(self):
        """All supported colors must append the reset sequence \033[0m to avoid color bleeding."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            for color in ["red", "green", "blue", "yellow", "cyan"]:
                res = colorize("sample", color)
                self.assertTrue(res.endswith("\033[0m"), f"Color {color} missing reset code at end")


if __name__ == "__main__":
    unittest.main()
