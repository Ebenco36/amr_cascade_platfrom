import subprocess
import sys

import plotly.io as pio
import pytest

from amr_cascade_platform.visualization.report import plotly_exporter


class _Kaleido02Scope:
    """Imitates Kaleido 0.2: one long-lived process, a reply read with no time limit, a new process
    started when the old one has died."""

    def __init__(self, replies):
        self._replies = list(replies)  # per call: None = never answers, text = answers this
        self._proc = None
        self.mathjax = None

    def _ensure(self):
        if self._proc is None or self._proc.poll() is not None:
            reply = self._replies.pop(0)
            code = "import time; time.sleep(60)" if reply is None else f"print({reply!r})"
            self._proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)


class _Figure:
    def __init__(self, scope):
        self.scope = scope

    def write_image(self, path, **kwargs):
        self.scope._ensure()
        line = self.scope._proc.stdout.readline()
        if not line:
            raise ValueError("Transform failed. Error stream: (process ended)")
        path.write_text(line.decode())


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(plotly_exporter, "IMAGE_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(plotly_exporter, "IMAGE_ATTEMPTS", 2)


def test_a_stuck_image_process_is_killed_and_the_image_written_on_the_next_attempt(fast, monkeypatch, tmp_path):
    scope = _Kaleido02Scope([None, "image"])
    monkeypatch.setattr(pio.kaleido, "scope", scope)
    plotly_exporter.write_image_with_watchdog(_Figure(scope), tmp_path / "f.pdf", format="pdf")
    assert (tmp_path / "f.pdf").read_text().strip() == "image"


def test_an_image_that_stalls_every_time_fails_instead_of_waiting_for_ever(fast, monkeypatch, tmp_path):
    scope = _Kaleido02Scope([None, None])
    monkeypatch.setattr(pio.kaleido, "scope", scope)
    with pytest.raises(TimeoutError, match="stalled 2 times"):
        plotly_exporter.write_image_with_watchdog(_Figure(scope), tmp_path / "f.pdf", format="pdf")


def test_a_real_export_error_is_raised_at_once_and_not_retried(fast, monkeypatch, tmp_path):
    scope = _Kaleido02Scope(["unused"])
    monkeypatch.setattr(pio.kaleido, "scope", scope)

    class Broken(_Figure):
        calls = 0

        def write_image(self, path, **kwargs):
            Broken.calls += 1
            raise RuntimeError("bad figure")

    with pytest.raises(RuntimeError, match="bad figure"):
        plotly_exporter.write_image_with_watchdog(Broken(scope), tmp_path / "f.pdf")
    assert Broken.calls == 1


def test_without_a_long_lived_process_the_image_is_written_directly(monkeypatch, tmp_path):
    class Kaleido1Scope:
        mathjax = None

    monkeypatch.setattr(pio.kaleido, "scope", Kaleido1Scope())
    written = []

    class Plain:
        def write_image(self, path, **kwargs):
            written.append(kwargs)

    plotly_exporter.write_image_with_watchdog(Plain(), tmp_path / "f.png", format="png")
    assert written == [{"format": "png"}]
