# from collections.abc import Callable

from threading import Thread, Event
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Callable
from numpy.typing import NDArray
import numpy as np
from queue import Queue
from time import sleep


class Pluggable(Thread, ABC):
    def __init__(self, source: "Pluggable | None" = None) -> None:
        self.source = source
        self.output_queue: Queue[NDArray[np.float64]] = Queue(maxsize=10)
        self.stop_event = Event()
        self.watcher = Thread(target=self.source_watcher, daemon=True)
        return super().__init__()

    def start(self) -> None:
        if self.source is not None:
            if not self.source.is_alive():
                self.source.start()
            self.watcher.start()
        return super().start()

    def output(self, block: bool = True) -> NDArray[np.float64]:
        a = self.output_queue.get(block)
        self.output_queue.task_done()
        return a

    def stop(self) -> None:
        self.stop_event.set()

    def source_watcher(self) -> None:
        if self.source is not None:
            while self.source.is_alive():
                sleep(0.1)
            self.stop_event.set()
