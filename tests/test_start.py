"""start.py replaces a running Studio that predates the code, and nothing else."""
import start


class _Proc:
    def __init__(self, created, cpu=0.0):
        self.created, self.cpu, self.stopped = created, cpu, False

    def create_time(self):
        return self.created

    def cpu_percent(self, interval=None):
        return self.cpu

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0


def _setup(monkeypatch, proc):
    monkeypatch.setattr(start, "code_stamp", lambda: 1000.0)
    monkeypatch.setattr(start, "studio_process", lambda port: proc)
    monkeypatch.setattr(start, "say", lambda *a: None)


def test_outdated_idle_copy_is_replaced(monkeypatch):
    proc = _Proc(created=900.0)
    _setup(monkeypatch, proc)
    assert start.replace_if_outdated(7860) is True and proc.stopped


def test_current_copy_is_reused(monkeypatch):
    proc = _Proc(created=1100.0)
    _setup(monkeypatch, proc)
    assert start.replace_if_outdated(7860) is False and not proc.stopped


def test_busy_or_unknown_copy_is_left_alone(monkeypatch):
    proc = _Proc(created=900.0, cpu=180.0)
    _setup(monkeypatch, proc)
    assert start.replace_if_outdated(7860) is False and not proc.stopped
    _setup(monkeypatch, None)       # not identifiable as our app
    assert start.replace_if_outdated(7860) is False


def test_code_stamp_sees_the_scripts():
    assert start.code_stamp() > 0
