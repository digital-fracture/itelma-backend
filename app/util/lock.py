import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


class AsyncRWLock:
    """
    Simple asyncio-compatible read/write lock.

    - multiple concurrent readers
    - single writer exclusive
    - writer gets exclusive access (no new readers while writer waits)
    """

    def __init__(self) -> None:
        self._reader_count = 0
        self._readers_lock = asyncio.Lock()
        self._writer_lock = asyncio.Lock()  # only one writer at a time
        self._reader_queue_lock = asyncio.Lock()  # to block readers when writer is waiting

        self._no_reader_event = asyncio.Event()
        self._no_writer_event = asyncio.Event()
        self._no_reader_event.set()  # initially no reader
        self._no_writer_event.set()  # initially no writer

    @asynccontextmanager
    async def read(self) -> AsyncGenerator[None]:
        async with self._reader_queue_lock, self._readers_lock:
            self._reader_count += 1

            if self._reader_count == 1:
                await self._no_writer_event.wait()  # ensure there is no active writer
                self._no_reader_event.clear()

        try:
            yield

        finally:
            async with self._readers_lock:
                self._reader_count -= 1

                if self._reader_count == 0:
                    self._no_reader_event.set()

    @asynccontextmanager
    async def write(self) -> AsyncGenerator[None]:
        async with self._writer_lock:
            await self._reader_queue_lock.acquire()  # prevent new readers from starting
            await self._no_reader_event.wait()  # wait for active readers to finish
            self._no_writer_event.clear()

            try:
                yield

            finally:
                self._no_writer_event.set()
                self._reader_queue_lock.release()
