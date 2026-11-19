"""Closed-loop episode over a degradation scenario (the `pis.harness.run_scenario` loop)."""
from __future__ import annotations

from typing import Callable

from pis.adapters import initial_policy
from pis.harness import Controller
from pis.phase_b.scenarios import DegradedSim, Scenario


def run_episode(scenario: Scenario, controller: Controller, policy_source: str,
                on_window: Callable[[DegradedSim, dict], None] | None = None) -> list[dict]:
    """One record per window: harness keys {"t", "tel", "policy", "info"} plus "applied" (the masked
    policy that actually ran in the window that produced `tel`) and "source" (`policy_source`).

    `on_window(dsim, record)` is called after each record, with the simulator at the window's end.
    Controllers with `attach_sim` get the underlying simulator. Controllers never see the
    "phase_b" block of the telemetry (the degradation is not announced); records keep it.
    """
    ds = DegradedSim(scenario)
    if hasattr(controller, "attach_sim"):
        controller.attach_sim(ds.sim)
    current = initial_policy()
    records: list[dict] = []
    history: list[tuple[dict, dict]] = []
    tel = ds.run_window(current)
    while True:
        seen = _hidden(tel)
        view = history + [(seen, current)]
        policy, info = controller.decide(seen, current, view)
        rec = {"t": tel["t"], "tel": tel, "policy": policy, "info": info,
               "applied": ds.last_applied, "source": policy_source}
        records.append(rec)
        if on_window is not None:
            on_window(ds, rec)
        history.append((seen, policy))
        if ds.done():
            break
        current = policy
        tel = ds.run_window(current)
    return records


def _hidden(tel: dict) -> dict:
    return {k: v for k, v in tel.items() if k != "phase_b"}
