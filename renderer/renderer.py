import threading
import torch
from concurrent.futures import ThreadPoolExecutor, Future, wait
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class JobID:
    pipeline: int 
    value: int       

    def __str__(self):
        return f"JobID(pipeline={self.pipeline}, id={self.value})"

class RenderOp(ABC):
    """
    Abstract base class for all rendering operations.
    """
    @abstractmethod
    def execute(self, slot: int, slots: dict[int, torch.Tensor]) -> torch.Tensor | None:
        pass

class RenderQueue:
    def __init__(self, renderer: 'Renderer', pipeline_id: int = 0):
        self.renderer = renderer
        self.pipeline_id = pipeline_id
        self.jobs: list[tuple[JobID, RenderOp]] = []
        self.dependencies: dict[JobID, list[JobID]] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def submit(self, op: RenderOp, wait_for: list[JobID] | None = None) -> JobID:
        with self._lock:
            job_id = JobID(self.pipeline_id, self._counter)
            self._counter += 1
        self.jobs.append((job_id, op))
        self.dependencies[job_id] = wait_for or []
        return job_id

    def clear(self):
        self.jobs.clear()
        self.dependencies.clear()
        self._counter = 0


class Renderer:
    def __init__(self, device: torch.device = torch.device('cpu'), max_workers: int = 4):
        self.device = device
        self.pipeline_counter = 0
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def begin(self) -> RenderQueue:
        with self._lock:
            pipeline_id = self.pipeline_counter
            self.pipeline_counter += 1
        return RenderQueue(self, pipeline_id)

    def render(self, queue: RenderQueue, output: list[JobID] | None = None) -> dict[JobID, torch.Tensor]:
        if not queue.jobs:
            return {}

        slots: dict[int, torch.Tensor] = {}
        outputs: dict[JobID, torch.Tensor] = {}
        futures: dict[JobID, Future] = {}
        slot_locks: dict[int, threading.Lock] = {}
        last_slot: int | None = None

        # Determine which job outputs to return
        if output is None:
            output = [queue.jobs[-1][0]]
        output_set: set[JobID] = set(output)

        # Preprocess jobs to determine fallback slot if unspecified
        slot_map: dict[JobID, int] = {}
        for job_id, op in queue.jobs:
            slot = getattr(op, 'slot', None)
            if slot is not None:
                last_slot = slot
            elif last_slot is not None:
                slot_map[job_id] = last_slot

        def submit_job(job_id: JobID, op: RenderOp):
            # Wait for dependencies
            wait([futures[dep] for dep in queue.dependencies.get(job_id, [])])

            # Determine working slot
            slot = getattr(op, 'slot', None)
            if slot is None:
                slot = slot_map.get(job_id, 0)

            if slot not in slot_locks:
                slot_locks[slot] = threading.Lock()

            with slot_locks[slot]:
                try: 
                    result = op.execute(slot=slot, slots=slots)
                except Exception as e:
                    print(f"[ERROR] Job {job_id} failed: {e}")
                    raise
                if result is None: return

                slots[slot] = result
                if job_id not in output_set: return

                with self._lock:
                    outputs[job_id] = result.detach().clone()

        for job_id, op in queue.jobs:
            futures[job_id] = self._executor.submit(submit_job, job_id, op)

        wait(list(futures.values()))
        return outputs
