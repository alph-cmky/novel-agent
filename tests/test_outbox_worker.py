import asyncio

from novel_agent.storage.outbox_worker import OutboxWorker


class _FakeManager:
    def __init__(self, events):
        self.events = {event["id"]: dict(event) for event in events}
        self.processed = []
        self.retried = []

    def list_all_outbox_events(self, status=None):
        values = list(self.events.values())
        return [event for event in values if status is None or event["status"] == status]

    def retry_outbox_event(self, event_id):
        self.retried.append(event_id)
        self.events[event_id]["status"] = "pending"
        return self.events[event_id]

    def claim_outbox_events(self, owner, *, limit, lease_seconds, max_retries):
        claimed = []
        for event in self.events.values():
            if event["status"] == "pending" or (
                event["status"] == "failed" and event["retry_count"] < max_retries
            ):
                event["status"] = "processing"
                event["lease_owner"] = owner
                claimed.append(event)
        return claimed[:limit]

    def process_outbox_event(self, event_id, owner=None):
        self.processed.append(event_id)
        self.events[event_id]["status"] = "done"
        return self.events[event_id]


def test_worker_processes_pending_and_retryable_failed_events():
    manager = _FakeManager([
        {"id": "pending", "status": "pending", "retry_count": 0},
        {"id": "failed", "status": "failed", "retry_count": 1},
        {"id": "exhausted", "status": "failed", "retry_count": 3},
    ])
    worker = OutboxWorker(manager, max_retries=3)

    processed = asyncio.run(worker.process_once())

    assert processed == 2
    assert manager.processed == ["pending", "failed"]
    assert manager.retried == []
    assert manager.events["exhausted"]["status"] == "failed"


def test_worker_start_stop_is_idempotent():
    manager = _FakeManager([])
    worker = OutboxWorker(manager, poll_interval=0.01)

    async def exercise():
        await worker.start()
        await worker.start()
        await worker.stop()
        await worker.stop()

    asyncio.run(exercise())
