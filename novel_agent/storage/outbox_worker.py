"""Background delivery loop for derived-state Outbox events."""

import asyncio
import uuid


class OutboxWorker:
    """Poll and deliver Outbox events without blocking the async server."""

    def __init__(
        self,
        manager,
        *,
        poll_interval: float = 1.0,
        max_retries: int = 3,
        batch_size: int = 10,
        lease_seconds: int = 60,
    ):
        self.manager = manager
        self.poll_interval = max(poll_interval, 0.01)
        self.max_retries = max(max_retries, 0)
        self.batch_size = max(batch_size, 1)
        self.lease_seconds = max(lease_seconds, 1)
        self.owner = f"outbox-worker-{uuid.uuid4()}"
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    async def run(self) -> None:
        while not self._stop.is_set():
            processed = await self.process_once()
            if processed == 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), self.poll_interval)
                except TimeoutError:
                    pass

    async def process_once(self) -> int:
        events = await asyncio.to_thread(
            self.manager.claim_outbox_events,
            self.owner,
            limit=self.batch_size,
            lease_seconds=self.lease_seconds,
            max_retries=self.max_retries,
        )
        processed = 0
        for event in events:
            if self._stop.is_set():
                break
            await asyncio.to_thread(
                self.manager.process_outbox_event,
                event["id"],
                self.owner,
            )
            processed += 1
        return processed
