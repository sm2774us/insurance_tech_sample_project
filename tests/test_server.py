"""Tests for the subprocess-based full-stack launcher.

The Taipy `Gui.run(...)` call blocks forever by design (it's the server's
event loop), so these tests stub it out rather than actually invoking it,
and instead assert the surrounding process-management behavior: the API
subprocess starts, fast failures are surfaced, and teardown happens in
`finally`.
"""

from __future__ import annotations

from unittest import mock

import pytest

from fig_quant.web import server as server_module


def test_run_full_stack_terminates_api_subprocess_on_gui_exit() -> None:
    fake_proc = mock.Mock()
    fake_proc.poll.return_value = None  # still running after the startup grace period

    fake_gui = mock.Mock()
    fake_gui.run.side_effect = KeyboardInterrupt  # simulate Ctrl+C stopping the GUI

    with (
        mock.patch("subprocess.Popen", return_value=fake_proc) as popen,
        mock.patch("fig_quant.web.gui.build_gui", return_value=fake_gui),
        mock.patch("time.sleep"),
    ):
        with pytest.raises(KeyboardInterrupt):
            server_module.run_full_stack(
                api_host="127.0.0.1", api_port=8001, gui_host="127.0.0.1", gui_port=8000
            )

    popen.assert_called_once()
    called_args = popen.call_args[0][0]
    assert "--port" in called_args
    assert "8001" in called_args
    fake_proc.terminate.assert_called_once()
    fake_proc.wait.assert_called_once()


def test_run_full_stack_raises_if_api_subprocess_dies_immediately() -> None:
    fake_proc = mock.Mock()
    fake_proc.poll.return_value = 1  # exited immediately
    fake_proc.returncode = 1
    fake_proc.stdout.read.return_value = "address already in use"

    with (
        mock.patch("subprocess.Popen", return_value=fake_proc),
        mock.patch("time.sleep"),
    ):
        with pytest.raises(RuntimeError, match="exited immediately"):
            server_module.run_full_stack(
                api_host="127.0.0.1", api_port=8001, gui_host="127.0.0.1", gui_port=8000
            )
