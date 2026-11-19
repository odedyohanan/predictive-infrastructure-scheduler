import numpy as np
import pytest

from pis.phase_b import incidents
from pis.phase_b.incidents import Incident, IncidentRule, label_series

RULE = IncidentRule(k=3, interval=10, majority=0.5)
NO_MAJORITY = IncidentRule(k=3, interval=1000)


def _series(n, *runs):
    v = np.zeros(n, dtype=bool)
    for a, b in runs:
        v[a:b] = True
    return v


def _off(n):
    return np.zeros(n, dtype=bool)


def test_run_of_k_is_incident():
    v = _series(60, (30, 33))
    lab = label_series(v, _off(60), NO_MAJORITY)
    assert lab.in_incident.tolist() == v.tolist()
    assert lab.incidents == [Incident(30, 33, "sudden")]
    assert lab.time_in_incident_s == pytest.approx(1.5)


def test_run_of_k_minus_1_is_not():
    v = _series(60, (30, 32), (40, 42))
    lab = label_series(v, _off(60), NO_MAJORITY)
    assert not lab.in_incident.any() and lab.incidents == [] and lab.time_in_incident_s == 0.0


def test_majority_rule():
    v = np.zeros(60, dtype=bool)
    v[[30, 31, 33, 34, 36, 37]] = True              # 6 violations, no run of 3
    lab = label_series(v, _off(60), RULE)
    # the trailing 10 windows hold all 6 (> half) for windows 37..39 only
    assert lab.in_incident.nonzero()[0].tolist() == [37, 38, 39]
    assert lab.incidents == [Incident(37, 40, "sudden")]
    v[39] = True                                    # window 40 now sees 31..40 -> 6 violations
    lab = label_series(v, _off(60), RULE)
    assert lab.in_incident.nonzero()[0].tolist() == [37, 38, 39, 40]


def test_exact_majority_is_not_incident():
    v = np.zeros(60, dtype=bool)
    v[[30, 31, 33, 34, 36]] = True                  # 5 of 10: not more than half
    assert not label_series(v, _off(60), RULE).in_incident.any()


def test_onset_in_warmup_ignored():
    v = _series(60, (15, 25), (40, 45))
    lab = label_series(v, _off(60), NO_MAJORITY, warmup=20)
    assert lab.incidents == [Incident(40, 45, "sudden")]
    assert lab.in_incident[15:25].all()             # warm-up windows still count as time in incident
    assert lab.time_in_incident_s == pytest.approx(15 * 0.5)


def test_in_incident_from_start():
    v = np.ones(40, dtype=bool)
    lab = label_series(v, _off(40), RULE, warmup=20)
    assert lab.incidents == [] and lab.in_incident.all()
    assert lab.time_in_incident_s == pytest.approx(20.0)
    assert incidents.is_chronic(lab, 40)
    assert not incidents.onset_within(lab, 0, 40)


def test_no_incident_episode():
    lab = label_series(_off(600), _off(600), RULE)
    assert lab.incidents == [] and lab.time_in_incident_s == 0.0
    assert not incidents.is_chronic(lab, 600)
    assert not incidents.onset_within(lab, 100, 50)


def test_empty_series():
    lab = label_series(_off(0), _off(0), RULE)
    assert lab.incidents == [] and lab.time_in_incident_s == 0.0


def test_cause_from_active_flag():
    v = _series(100, (30, 35), (60, 70))
    active = np.zeros(100, dtype=bool)
    active[50:] = True
    lab = label_series(v, active, NO_MAJORITY)
    assert [i.cause for i in lab.incidents] == ["sudden", "building_up"]


def test_lengths_must_match():
    with pytest.raises(ValueError):
        label_series(_off(10), _off(9), RULE)


def test_chronic():
    v = _series(100, (20, 71))
    assert incidents.is_chronic(label_series(v, _off(100), NO_MAJORITY), 100)
    v = _series(100, (20, 70))
    assert not incidents.is_chronic(label_series(v, _off(100), NO_MAJORITY), 100)


def test_onset_within_boundaries():
    lab = label_series(_series(100, (50, 55)), _off(100), NO_MAJORITY)
    assert incidents.onset_within(lab, 49, 1)
    assert incidents.onset_within(lab, 45, 5)
    assert not incidents.onset_within(lab, 45, 4)          # (45, 49] misses 50
    assert not incidents.onset_within(lab, 50, 10)         # onset at t is not in (t, t+h]


def _record(late=0, ontime=100, overdue=0, active=None):
    tel = {"t": 0.0, "a": {"completed_ontime": ontime, "late_new": late, "queue_overdue": overdue}}
    if active is not None:
        tel["phase_b"] = {"active": active}
    return {"tel": tel}


def test_violations_slo_v2():
    recs = [_record(), _record(late=2, ontime=100), _record(late=1, ontime=200),
            _record(ontime=0, overdue=3), _record(ontime=0, overdue=0)]
    assert incidents.violations(recs).tolist() == [False, True, False, True, False]


def test_label_wrapper_reads_active_flag():
    recs = [_record(active=False)] * 30 + [_record(late=50, active=True)] * 5 + [_record()] * 5
    lab = incidents.label(recs, NO_MAJORITY)
    assert lab.incidents == [Incident(30, 35, "building_up")]
    plain = [_record()] * 30 + [_record(late=50)] * 5          # no phase_b block: not active
    assert incidents.label(plain, NO_MAJORITY).incidents == [Incident(30, 35, "sudden")]


def test_early_signal():
    q = [0, 0, 2, 4, 6, 6, 6]
    recs = [{"tel": {"a": {"queue_overdue": x}}} for x in q]
    s = incidents.early_signal(recs, w=3)
    assert s.dtype == float and len(s) == len(q)
    assert s[0] == 0.0
    assert s[1] == pytest.approx(0.0)
    assert s[2] == pytest.approx(1.0)                      # 0, 0, 2
    assert s[3] == pytest.approx(2.0)                      # 0, 2, 4
    assert s[6] == pytest.approx(0.0)                      # 6, 6, 6


def test_label_real_episode():
    import dataclasses

    from pis.adapters import initial_policy
    from pis.phase_b.episode import run_episode
    from pis.phase_b.scenarios import MAGNITUDES, Degradation, scenario_for

    class Fixed:
        def decide(self, tel, current, history):
            return initial_policy(), {"decide_ms": 0.0}

    sc = dataclasses.replace(scenario_for(3101), duration_s=10.0,
                             degradation=Degradation("leak", 2.0, 2.0, MAGNITUDES["leak"]))
    recs = run_episode(sc, Fixed(), "fixed")
    lab = incidents.label(recs, RULE, warmup=2)
    assert lab.in_incident.shape == (20,)
    assert len(incidents.early_signal(recs)) == 20
    assert all(i.cause == "building_up" for i in lab.incidents if i.onset >= 4)
