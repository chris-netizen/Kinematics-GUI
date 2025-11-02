# core/worker.py
from PySide6.QtCore import QRunnable, Slot, QObject, Signal, QThreadPool


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(object)


class Worker(QRunnable):
    def __init__(self, fn, on_done=None, on_error=None):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()
        if on_done:
            self.signals.finished.connect(on_done)
        if on_error:
            self.signals.error.connect(on_error)

    @Slot()
    def run(self):
        try:
            result = self.fn()
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(e)

    def start(self):
        QThreadPool.globalInstance().start(self)
