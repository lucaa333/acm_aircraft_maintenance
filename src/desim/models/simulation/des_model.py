from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import logging
from random import Random
import math

import numpy as np

from desim.domain.distributions import sample_service_time
from desim.domain.metrics import ORIGIN_FIELDS, ModelResult, t_critical_95
from desim.domain.scenario import Scenario
from desim.experiments.replications import confidence_interval
from desim.models.base import Model

logger = logging.getLogger(__name__)


class InvariantViolation(RuntimeError):
    """The simulation reached a state that the model rules do not allow."""


@dataclass(slots=True)
class Task:
    task_id: int
    kind: str
    priority: int
    arrival_time: float
    required_mechanics: int
    remaining_time: float
    escalated: bool = False
    queued_at: float = 0.0
    service_started_at: float | None = None
    interrupted_at: float | None = None
    completion_token: int = 0
    total_wait_time: float = 0.0
    window_wait: float = 0.0  # time queued inside the observation window, for class queue lengths
    observed_arrival: bool = False
    completed: bool = False


def origin_class(task: Task) -> str:
    """Class a task is reported under: fixed by origin, so escalation cannot move statistics between classes."""
    return "PM_escalated" if task.escalated else task.kind


class DiscreteEventSimulationModel(Model):
    """Simulate maintenance tasks competing for hangar slots and mechanics."""
    name = "Discrete Event Simulation"

    def run(self, scenario: Scenario) -> ModelResult:
        """Run independent replications and summarize their KPI values."""
        return _average_results(self.name, scenario.name, self.run_replications(scenario))

    def run_replications(self, scenario: Scenario) -> list[ModelResult]:
        """Run all replications of a scenario and log every failed plausibility check."""
        results = [self._run_once(scenario, scenario.simulation.seed + idx) for idx in range(scenario.simulation.replications)]
        for warning in post_check_warnings(scenario, results):
            logger.warning("%s: %s", scenario.name, warning)
        return results

    def _run_once(
        self, scenario: Scenario, seed: int, trace: list | None = None, interruptions: list | None = None
    ) -> ModelResult:
        """Run one event-driven replication; ``trace`` collects (time, class, wait, CM still waiting, task id) per start.

        With preemption a task can start several times; ``wait`` is then the wait before this start.
        ``interruptions`` collects (time, trigger id, victim id, ids of the PM that could still be interrupted).
        """
        rng = Random(seed)
        clock = 0.0
        end_time = scenario.simulation.run_length
        warmup = scenario.simulation.warmup
        mechanics_available = scenario.resources.mechanics
        slots_available = scenario.resources.hangar_slots
        # Heap entries are ordered by time, then by insertion order.
        # The optional token invalidates stale completion events after preemption.
        event_queue: list[tuple[float, int, str, int, int | None]] = []
        active: dict[int, Task] = {}
        tasks: dict[int, Task] = {}
        hp_queue: deque[int] = deque()
        lp_queue: deque[int] = deque()
        cm_ahead_of_escalated = scenario.escalation.queue_position == "behind_cm"
        event_id = 0
        task_id = 0
        runs_in_service = 0
        slot_busy_area = 0.0
        mechanics_busy_area = 0.0
        last_area_time = 0.0
        completed = {"CM": 0, "PM": 0, "PM_escalated": 0}
        total_wait = dict.fromkeys(completed, 0.0)
        total_cycle = dict.fromkeys(completed, 0.0)
        escalated_count = 0
        preemptions = {"by_cm": 0, "by_escalated_pm": 0, "pm_after_deadline": 0, "cascades": 0}
        preempted_tasks: set[int] = set()
        setup_hours = 0.0
        blocked_area = 0.0
        blocked_cm_area = 0.0
        observed_pm = 0

        def schedule(time: float, event_type: str, ref: int, token: int | None = None) -> None:
            nonlocal event_id
            event_id += 1
            heapq.heappush(event_queue, (time, event_id, event_type, ref, token))

        def update_areas(new_time: float) -> None:
            nonlocal last_area_time, slot_busy_area, mechanics_busy_area, blocked_area, blocked_cm_area
            # Accumulate the state that existed since the previous event, starting at warm-up.
            start = max(last_area_time, warmup)
            stop = max(min(new_time, end_time), warmup)
            dt = max(0.0, stop - start)
            busy_slots = scenario.resources.hangar_slots - slots_available
            busy_mechanics = scenario.resources.mechanics - mechanics_available
            for queued_id in (*hp_queue, *lp_queue):
                tasks[queued_id].window_wait += dt
            slot_busy_area += dt * busy_slots
            mechanics_busy_area += dt * busy_mechanics
            if dt > 0 and slots_available > 0 and (hp_queue or lp_queue):
                # A bay is free but the dispatch rule keeps every waiting task out.
                blocked_area += dt
                head = tasks[hp_queue[0]] if hp_queue else tasks[lp_queue[0]]
                if head.kind == "CM" and any(item.kind == "CM" for item in active.values()):
                    blocked_cm_area += dt
            last_area_time = new_time

        def enqueue(task: Task) -> None:
            task.queued_at = clock
            if task.priority != 0:
                lp_queue.append(task.task_id)
            elif cm_ahead_of_escalated and not task.escalated:
                hp_queue.insert(sum(1 for queued_id in hp_queue if not tasks[queued_id].escalated), task.task_id)
            else:
                hp_queue.append(task.task_id)

        def can_start(task: Task) -> bool:
            return mechanics_available >= task.required_mechanics and slots_available >= 1

        def start_task(task: Task) -> None:
            nonlocal mechanics_available, slots_available, runs_in_service
            slots_available -= 1
            mechanics_available -= task.required_mechanics
            runs_in_service += 1
            task.service_started_at = clock
            task.total_wait_time += max(0.0, clock - task.queued_at)
            if trace is not None:
                waiting_cm = sum(1 for queued_id in hp_queue if not tasks[queued_id].escalated)
                trace.append((clock, origin_class(task), clock - task.queued_at, waiting_cm, task.task_id))
            task.completion_token += 1
            active[task.task_id] = task
            schedule(clock + task.remaining_time, "complete", task.task_id, task.completion_token)

        def release_task_resources(task: Task) -> None:
            nonlocal mechanics_available, slots_available, runs_in_service
            mechanics_available += task.required_mechanics
            slots_available += 1
            runs_in_service -= 1

        def escalate(task: Task, behind_head: bool = False) -> None:
            # A waiting PM moves to the HP queue; it stays a PM (kind, service time, 6 engineers), only its rank changes.
            # behind_head: called during dispatch, where the current HP head must stay first.
            nonlocal escalated_count
            task.escalated = True
            task.priority = 0
            lp_queue.remove(task.task_id)
            if scenario.escalation.queue_position == "front":
                hp_queue.insert(1 if behind_head and hp_queue else 0, task.task_id)
            else:
                hp_queue.append(task.task_id)
            if task.observed_arrival:
                escalated_count += 1

        def maybe_preempt_for(task: Task) -> None:
            nonlocal setup_hours
            # Only an HP head preempts; in FCFS mode every task is in the LP queue, so nothing is ever interrupted.
            if task.priority != 0 or can_start(task):
                return
            # Only never-escalated PM tasks can be paused; CM and escalated PM always run to completion.
            lp_active = [item for item in active.values() if item.kind == "PM" and not item.escalated]
            threshold = scenario.priority.escalate_threshold
            # EP7: first the PM still within their deadline (arrival + X), among them the one started last, which has
            # made the least progress; a PM past its deadline only if there is no other. Ties keep the active order.
            # Without escalation no deadline passes, so this is the PM started last.
            lp_active.sort(key=lambda item: (threshold is not None and clock >= item.arrival_time + threshold, -item.service_started_at))
            # Pause nothing unless the pauses really let this task start.
            slots = slots_available + len(lp_active)
            mechanics = mechanics_available + sum(item.required_mechanics for item in lp_active)
            if slots < 1 or mechanics < task.required_mechanics:
                return
            for index, victim in enumerate(lp_active):
                if interruptions is not None:
                    interruptions.append((clock, task.task_id, victim.task_id, tuple(item.task_id for item in lp_active[index:])))
                elapsed = clock - (victim.service_started_at or clock)
                # The set-up time is extra work after resuming; it occupies the bay and the engineers like service.
                victim.remaining_time = max(0.001, victim.remaining_time - elapsed) + scenario.priority.preemption_setup_time
                if clock >= warmup:
                    preemptions["by_escalated_pm" if task.escalated else "by_cm"] += 1
                    setup_hours += scenario.priority.preemption_setup_time
                    # A cascade: the interrupting escalated PM was itself interrupted in this instant.
                    preemptions["cascades"] += task.interrupted_at == clock
                victim.interrupted_at = clock
                if victim.observed_arrival:
                    preempted_tasks.add(victim.task_id)
                victim.service_started_at = None
                del active[victim.task_id]
                # The token makes the victim's old completion event harmless when it is reached.
                victim.completion_token += 1
                # The aircraft leaves its bay: bay and engineers are free for the waiting task.
                release_task_resources(victim)
                enqueue(victim)
                # Resume before fresh PMs: head-only dispatch would strand a paused PM behind them.
                lp_queue.remove(victim.task_id)
                lp_queue.appendleft(victim.task_id)
                # Its escalation event has already passed while it ran: a PM past its deadline goes to the HP queue now.
                if threshold is not None and clock >= victim.arrival_time + threshold:
                    escalate(victim, behind_head=True)
                    if clock >= warmup:
                        preemptions["pm_after_deadline"] += 1
                if can_start(task):
                    return

        def check_invariants() -> None:
            busy_bays = scenario.resources.hangar_slots - slots_available
            busy_mechanics = scenario.resources.mechanics - mechanics_available
            if not 0 <= busy_bays <= scenario.resources.hangar_slots or not 0 <= busy_mechanics <= scenario.resources.mechanics:
                raise InvariantViolation(f"t={clock}: resources out of range, bays {busy_bays}, mechanics {busy_mechanics}")
            if busy_bays != len(active) or busy_mechanics != sum(task.required_mechanics for task in active.values()):
                raise InvariantViolation(f"t={clock}: {len(active)} running tasks hold {busy_bays} bays, {busy_mechanics} mechanics")
            if runs_in_service != len(active):
                raise InvariantViolation(f"t={clock}: {runs_in_service} tasks started but unfinished, {len(active)} running")
            head = tasks[hp_queue[0]] if hp_queue else tasks[lp_queue[0]] if lp_queue else None
            if head is not None and (can_start(head) or not active):
                raise InvariantViolation(f"t={clock}: standstill, task {head.task_id} heads a queue and nothing runs or starts")

        def try_start_tasks() -> None:
            changed = True
            while changed:
                changed = False
                # Only the head of each queue is considered; a blocked HP head keeps LP work waiting.
                if hp_queue:
                    candidate = tasks[hp_queue[0]]
                    maybe_preempt_for(candidate)
                    if can_start(candidate):
                        hp_queue.popleft()
                        start_task(candidate)
                        changed = True
                        continue
                if hp_queue:
                    continue
                if lp_queue:
                    candidate = tasks[lp_queue[0]]
                    if can_start(candidate):
                        lp_queue.popleft()
                        start_task(candidate)
                        changed = True

        def create_task(kind: str) -> None:
            nonlocal task_id, observed_pm
            # Arrivals schedule both this task and the next arrival of the same class.
            task_id += 1
            if kind == "PM":
                service = sample_service_time(scenario.pm_service, rng)
                task = Task(task_id, "PM", 1, clock, scenario.pm_mechanics, service)
                task.observed_arrival = clock >= warmup
                tasks[task_id] = task
                if task.observed_arrival:
                    observed_pm += 1
                if scenario.priority.escalate_threshold is not None:
                    schedule(clock + scenario.priority.escalate_threshold, "escalate", task_id)
                enqueue(task)
                schedule(clock + rng.expovariate(scenario.arrival_rate_pm), "arrival_pm", 0)
            else:
                service = sample_service_time(scenario.cm_service, rng)
                # FCFS keeps every task in the LP queue, which is then plain arrival order.
                priority = 1 if scenario.priority.mode == "fcfs" else 0
                task = Task(task_id, "CM", priority, clock, scenario.cm_mechanics, service)
                task.observed_arrival = clock >= warmup
                tasks[task_id] = task
                enqueue(task)
                schedule(clock + rng.expovariate(scenario.arrival_rate_cm), "arrival_cm", 0)
            try_start_tasks()

        if scenario.arrival_rate_pm > 0:
            schedule(rng.expovariate(scenario.arrival_rate_pm), "arrival_pm", 0)
        if scenario.arrival_rate_cm > 0:
            schedule(rng.expovariate(scenario.arrival_rate_cm), "arrival_cm", 0)

        while event_queue:
            time, _, event_type, ref, token = heapq.heappop(event_queue)
            if time > end_time:
                update_areas(end_time)
                break
            update_areas(time)
            if time < clock:
                raise InvariantViolation(f"event at t={time} lies before the clock at t={clock}")
            # Account for the previous system state before applying this event's changes.
            clock = time
            if event_type == "arrival_pm":
                create_task("PM")
            elif event_type == "arrival_cm":
                create_task("CM")
            elif event_type == "escalate":
                task = tasks[ref]
                # Only a PM still waiting after X hours escalates; a task in service needs no urgency.
                if task.task_id in lp_queue:
                    escalate(task)
                    try_start_tasks()
            elif event_type == "complete":
                task = tasks[ref]
                if task.task_id not in active or token != task.completion_token:
                    continue
                if task.completed:
                    raise InvariantViolation(f"t={clock}: task {task.task_id} completed twice")
                del active[task.task_id]
                task.completed = True
                release_task_resources(task)
                if task.observed_arrival:
                    group = origin_class(task)
                    completed[group] += 1
                    total_wait[group] += task.total_wait_time
                    total_cycle[group] += clock - task.arrival_time
                try_start_tasks()
            if scenario.simulation.check_invariants:
                check_invariants()

        # Customer averages include observed arrivals that completed before the horizon.
        # Time-average queues and utilization instead use the full post-warm-up interval.
        observed_time = max(1e-9, end_time - warmup)

        def per_task(totals: dict[str, float], group: str) -> float | None:
            # None, not 0: a replication without escalations has no escalated-PM average.
            return totals[group] / completed[group] if completed[group] else None

        queued = dict.fromkeys(completed, 0.0)
        for task in tasks.values():
            queued[origin_class(task)] += task.window_wait / observed_time
        queue_pm = queued["PM"] + queued["PM_escalated"]
        counted_pm = completed["PM"] + completed["PM_escalated"]
        wait_pm = total_wait["PM"] + total_wait["PM_escalated"]
        return ModelResult(
            model_name=self.name,
            scenario_name=scenario.name,
            waiting_time_total=sum(total_wait.values()) / max(1, sum(completed.values())),
            queue_length_total=sum(queued.values()),
            waiting_time_hp=per_task(total_wait, "CM") or 0.0,
            waiting_time_lp=wait_pm / counted_pm if counted_pm else 0.0,
            queue_length_hp=queued["CM"],
            queue_length_lp=queue_pm,
            mttr_cm=per_task(total_cycle, "CM") or 0.0,
            hangar_utilization=(slot_busy_area / observed_time) / max(1, scenario.resources.hangar_slots),
            mechanics_utilization=(mechanics_busy_area / observed_time) / max(1, scenario.resources.mechanics),
            escalation_rate=escalated_count / observed_pm if observed_pm else 0.0,
            waiting_time_pm_regular=per_task(total_wait, "PM"),
            waiting_time_pm_escalated=per_task(total_wait, "PM_escalated"),
            queue_length_pm_regular=queued["PM"],
            queue_length_pm_escalated=queued["PM_escalated"],
            cycle_time_pm_regular=per_task(total_cycle, "PM"),
            cycle_time_pm_escalated=per_task(total_cycle, "PM_escalated"),
            tasks_cm=completed["CM"],
            tasks_pm_regular=completed["PM"],
            tasks_pm_escalated=completed["PM_escalated"],
            escalations_total=escalated_count,
            preemption_rate=len(preempted_tasks) / observed_pm if observed_pm else 0.0,
            preemptions_total=preemptions["by_cm"] + preemptions["by_escalated_pm"],
            preemptions_by_cm=preemptions["by_cm"],
            preemptions_by_escalated_pm=preemptions["by_escalated_pm"],
            preemptions_after_deadline=preemptions["pm_after_deadline"],
            preemption_cascades=preemptions["cascades"],
            setup_hours=setup_hours,
            blocked_bay_share=blocked_area / observed_time,
            blocked_cm_behind_cm_share=blocked_cm_area / observed_time,
        )


def post_check_warnings(scenario: Scenario, results: list[ModelResult]) -> list[str]:
    """Compare replication KPIs with values known from theory; each test uses the 95 % interval as tolerance."""
    pm, cm, resources = scenario.pm_service, scenario.cm_service, scenario.resources
    pm_work = scenario.arrival_rate_pm * pm.moment_mean
    cm_work = scenario.arrival_rate_cm * cm.moment_mean
    rho_hangar = (pm_work + cm_work) / resources.hangar_slots
    rho_mechanics = (pm_work * scenario.pm_mechanics + cm_work * scenario.cm_mechanics) / resources.mechanics
    if len(results) < 2 or rho_hangar >= 1 or rho_mechanics >= 1:
        return []
    observed = scenario.simulation.run_length - scenario.simulation.warmup
    # Set-up time after a preemption is extra work the theoretical values do not contain.
    setup = [(result.setup_hours or 0.0) / observed for result in results]
    checks = [
        ("hangar utilization", [r.hangar_utilization - s / resources.hangar_slots for r, s in zip(results, setup)], rho_hangar),
        ("mechanics utilization", [r.mechanics_utilization - s * scenario.pm_mechanics / resources.mechanics for r, s in zip(results, setup)], rho_mechanics),
        ("MTTR minus CM wait", [result.mttr_cm - result.waiting_time_hp for result in results], cm.moment_mean),
    ]
    warnings = []
    for label, values, expected in checks:
        mean, lower, upper = confidence_interval(values)
        if not lower <= expected <= upper:
            warnings.append(f"{label} {mean:.4f} [{lower:.4f}; {upper:.4f}] does not contain {expected:.4f}")
    return warnings


def _average_results(model_name: str, scenario_name: str, results: list[ModelResult]) -> ModelResult:
    """Average replication KPIs and attach Student-t intervals to selected fields."""
    # Compute mean and uncertainty for all fields
    fields = [
        "waiting_time_total",
        "queue_length_total",
        "waiting_time_hp",
        "waiting_time_lp",
        "queue_length_hp",
        "queue_length_lp",
        "mttr_cm",
        "hangar_utilization",
        "mechanics_utilization",
        "escalation_rate",
        *ORIGIN_FIELDS,
    ]
    
    # Fields for which we compute full uncertainty statistics
    uncertainty_fields = {
        "waiting_time_total",
        "queue_length_total",
        "waiting_time_hp",
        "waiting_time_lp",
        "queue_length_hp",
        "queue_length_lp",
        "mttr_cm",
        "escalation_rate",
        "mechanics_utilization",
        "hangar_utilization",
        *ORIGIN_FIELDS,
    }
    
    averages: dict[str, float | None] = {}
    extra: dict[str, float | str] = {}
    
    for field in fields:
        values = [getattr(result, field) for result in results if getattr(result, field) is not None]
        if not values:
            averages[field] = None
            continue
        
        # Store mean
        mean_val = np.mean(values)
        averages[field] = mean_val
        
        # Compute and store uncertainty metrics for specified fields
        if field in uncertainty_fields and len(values) > 1:
            std_val = float(np.std(values, ddof=1))
            ci_margin = t_critical_95(len(values) - 1) * std_val / math.sqrt(len(values))
            ci_lower = mean_val - ci_margin
            ci_upper = mean_val + ci_margin
            
            extra[f"{field}_mean"] = float(mean_val)
            extra[f"{field}_std"] = std_val
            extra[f"{field}_ci95_lower"] = ci_lower
            extra[f"{field}_ci95_upper"] = ci_upper
    
    return ModelResult(
        model_name=model_name,
        scenario_name=scenario_name,
        extra=extra,
        **averages
    )
