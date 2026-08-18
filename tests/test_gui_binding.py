"""Regression test for the Taipy GUI variable/callback binding bug.

Guards against reintroducing the bug where `Gui(page=...)` was
constructed inside a factory function, breaking Taipy's stack-frame
based resolution of which module owns the page's bound variables and
callbacks. Every `{name}` referenced in the page markdown, and every
`on_action=name` callback, must exist as a top-level attribute of
`fig_quant.web.gui` -- this is exactly what Taipy itself checks when
rendering the page, so failing this test predicts the "not available in
either ... module" / "is not a valid function" warnings.
"""

from __future__ import annotations

import re

from fig_quant.web import gui as gui_module

_VAR_PATTERN = re.compile(r"\{(\w+)\}")
_ACTION_PATTERN = re.compile(r"on_action=(\w+)")


def test_page_bound_variables_exist_in_module_namespace() -> None:
    names = set(_VAR_PATTERN.findall(gui_module._page))
    assert names, "expected the page markdown to reference at least one bound variable"
    missing = [n for n in names if not hasattr(gui_module, n)]
    assert not missing, f"page references variables missing from gui module: {missing}"


def test_page_callbacks_exist_and_are_callable() -> None:
    names = set(_ACTION_PATTERN.findall(gui_module._page))
    assert names, "expected the page markdown to reference at least one on_action callback"
    for name in names:
        assert hasattr(gui_module, name), f"callback {name!r} missing from gui module"
        assert callable(getattr(gui_module, name)), f"{name!r} is not callable"


def test_gui_constructed_at_module_level_not_via_factory_wrapper() -> None:
    # The bug: `def build_gui(): return Gui(page=_page)` adds a stack frame
    # that breaks Taipy's module resolution. Guard that `gui` is a
    # ready-made module attribute (constructed at import time), and that
    # `build_gui()` -- kept only for call-site clarity -- returns that same
    # pre-built singleton rather than constructing a fresh one itself.
    assert hasattr(gui_module, "gui")
    assert gui_module.build_gui() is gui_module.gui
