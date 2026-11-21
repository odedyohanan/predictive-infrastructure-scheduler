"""A drop-in, compiled version of the simulator that produces bit-for-bit identical telemetry.

`FastSimulator` subclasses `sim.simulator.Simulator` and replaces `run_window`. The Python code before the
step loop (eligibility, shares, migration bookkeeping, A's L3 domain mask) is the parent's; the step loop
and the window-end pass over the A queue (ages, overdue marking, work left) run in a numba kernel; the
telemetry dict is built in Python from the kernel's results with the parent's expressions. The kernel
receives the simulator's own `np.random.Generator` (numba draws the same streams from it and advances its
state in place) and plain arrays. The A queue is kept as parallel arrays between windows (a deque view is
available as `queue`), so even a queue of hundreds of thousands of requests costs no conversion.

Bitwise identity rests on doing every floating-point operation in the same order as the Python code:
  - numpy sums over a contiguous float64 array use pairwise summation (`_pairwise`), starting from 0.0;
  - numpy sums over the 3 workload rows (axis 0) add rows one after another, starting from 0.0;
  - Python's `min(a, b)` returns `a` unless `b < a` (matters only for which of two equal values wins);
  - Python's builtin `sum` adds left to right.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from numba import njit

from pis.config import setup_imports

setup_imports()

from sim.config import A, B, C  # noqa: E402
from sim.simulator import Simulator  # noqa: E402

# Layout of the float parameter vector handed to the kernel (read from `self.spec` every window, so a
# spec replaced between windows takes effect).
(P_BASE_RATE, P_WAVE_AMP, P_WAVE_PERIOD, P_BURST_LEN, P_BURST_EXTRA, P_SERVICE_MS, P_SLO,
 P_INTERFERENCE, P_BUFFER_CAP, P_CPU_MS_PER_BATCH, P_GPU_MS_PER_BATCH, P_GPU_CORE_EQUIV, P_GPU_POWER,
 P_AMBIENT, P_HEAT, P_COOL, P_T_THROTTLE, P_T_MAX, P_MIN_SPEED, P_MIG_EFF, P_DELAY_MEAN, P_DT) = range(22)
N_PARAMS = 22


@njit(cache=True)
def _pairwise_block(a, lo, n):
    """numpy's pairwise sum of a[lo:lo+n] for n <= 128 (one unrolled block, no further splitting)."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    stop = n - n % 8
    while i < stop:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(cache=True)
def _pairwise(a, lo, n):
    """numpy's pairwise sum of a[lo:lo+n] (float64, contiguous), as `a[lo:lo+n].sum()` computes it.

    Above 128 elements numpy splits the range in two (the first part a multiple of 8 long) and adds the
    two sums. That recursion is evaluated here with an explicit stack: numba's on-disk cache crashes on
    recursive functions.
    """
    if n <= 128:
        return _pairwise_block(a, lo, n)
    depth = 64
    s_lo = np.empty(depth, np.int64)
    s_n = np.empty(depth, np.int64)
    s_stage = np.empty(depth, np.int64)
    vals = np.empty(depth)
    s_lo[0], s_n[0], s_stage[0] = lo, n, 0
    sp = 1
    vp = 0
    while sp > 0:
        top = sp - 1
        l, m = s_lo[top], s_n[top]
        if m <= 128:
            vals[vp] = _pairwise_block(a, l, m)
            vp += 1
            sp -= 1
            continue
        n2 = m // 2
        n2 -= n2 % 8
        if s_stage[top] == 0:          # evaluate the first part
            s_stage[top] = 1
            s_lo[sp], s_n[sp], s_stage[sp] = l, n2, 0
            sp += 1
        elif s_stage[top] == 1:        # then the second part
            s_stage[top] = 2
            s_lo[sp], s_n[sp], s_stage[sp] = l + n2, m - n2, 0
            sp += 1
        else:                          # both done: first + second
            vals[vp - 2] = vals[vp - 2] + vals[vp - 1]
            vp -= 1
            sp -= 1
    return vals[0]


@njit(cache=True)
def _pymin(a, b):
    """Python's min(a, b): `a` unless `b < a`."""
    return b if b < a else a


@njit(cache=True)
def _throttle(temp, t_throttle, t_max, min_speed):
    if temp <= t_throttle:
        return 1.0
    x = _pymin(1.0, (temp - t_throttle) / (t_max - t_throttle))
    return 1.0 - x * (1.0 - min_speed)


@njit(cache=True)
def _allocate(elig, demand, weights, spare, usage, got, w_mat, col, offer, total, take, frac, used, row):
    """The simulator's `allocate` (cpu_cap 1.0, 6 rounds), writing into the given work arrays.

    Returns nothing; `got` and `usage` hold the result.
    """
    n_w, n_k = elig.shape
    for k in range(n_k):
        spare[k] = 1.0
        for w in range(n_w):
            usage[w, k] = 0.0
    for w in range(n_w):
        got[w] = 0.0
    for _ in range(6):
        any_active = False
        aw0 = 0.0
        aw1 = 0.0
        aw2 = 0.0
        for w in range(n_w):
            want = demand[w] - got[w]
            act = want > 1e-9
            if act:
                any_active = True
            v = (1.0 if act else 0.0) * weights[w]
            if w == 0:
                aw0 = v
            elif w == 1:
                aw1 = v
            else:
                aw2 = v
        mx = spare[0]
        for k in range(1, n_k):
            if spare[k] > mx:
                mx = spare[k]
        if not any_active or mx <= 1e-12:
            break
        for k in range(n_k):
            w_mat[0, k] = (1.0 if elig[0, k] else 0.0) * aw0
            w_mat[1, k] = (1.0 if elig[1, k] else 0.0) * aw1
            w_mat[2, k] = (1.0 if elig[2, k] else 0.0) * aw2
            c = ((0.0 + w_mat[0, k]) + w_mat[1, k]) + w_mat[2, k]
            if c == 0:
                c = 1.0
            col[k] = c
        for w in range(n_w):
            for k in range(n_k):
                offer[w, k] = w_mat[w, k] / col[k] * spare[k]
                row[k] = offer[w, k]
            total[w] = _pairwise(row, 0, n_k)
        for w in range(n_w):
            want = demand[w] - got[w]
            t = total[w]
            take[w] = want if want < t else t
            frac[w] = take[w] / t if t > 0 else 0.0
        for w in range(n_w):
            for k in range(n_k):
                used[w, k] = offer[w, k] * frac[w]
                usage[w, k] += used[w, k]
            got[w] += take[w]
        for k in range(n_k):
            spare[k] -= ((0.0 + used[0, k]) + used[1, k]) + used[2, k]


@njit(cache=True)
def _grow(arr, n):
    out = np.empty(n, arr.dtype)
    out[:arr.shape[0]] = arr
    return out


@njit(cache=True)
def _run_steps(rng, n_steps, t, p, bursts, elig, weights, gpu_owner, a_dom, migrating_until, prev_usage,
               temps, b_buffer, q_t, q_rem, q_delay, q_late, q_h, q_len):
    """The per-step loop of `Simulator.run_window`, then the window-end pass over the queue.

    The queue is entries q_h .. q_h+q_len-1 of (q_t, q_rem, q_delay, q_late), head first; the arrays are
    updated in place, or replaced by larger ones when they fill up. `prev_usage` and `temps` are updated in
    place. Returns (t, b_buffer, queue arrays, queue start, queue length, arrivals, b_batches, c_work,
    usage sum, gpu power sum, completed latencies, late_new, queue_overdue, queue_oldest_age_ms,
    queue_work_ms).
    """
    dt = p[P_DT]
    dt_ms = dt * 1000.0
    n_k = elig.shape[1]
    n_g = temps.shape[0]
    n_dom = 0
    for k in range(n_k):
        if a_dom[k]:
            n_dom += 1
    n_dom_div = n_dom if n_dom > 1 else 1      # max(a_dom.sum(), 1)
    n_a_cpus = 0
    for k in range(n_k):
        if elig[0, k]:
            n_a_cpus += 1
    n_bursts = bursts.shape[0]

    # work arrays
    spare = np.empty(n_k)
    usage = np.zeros((3, n_k))
    got = np.zeros(3)
    w_mat = np.empty((3, n_k))
    col = np.empty(n_k)
    offer = np.empty((3, n_k))
    total = np.empty(3)
    take = np.empty(3)
    frac = np.empty(3)
    used = np.empty((3, n_k))
    row = np.empty(n_k)
    demand = np.empty(3)
    others = np.empty(n_k)
    sel = np.empty(n_k)
    power = np.empty(n_g)
    eff = np.empty(3)
    unf_t = np.empty(n_k + 4)
    unf_rem = np.empty(n_k + 4)
    unf_delay = np.empty(n_k + 4)
    unf_late = np.empty(n_k + 4, np.bool_)

    usage_acc = np.zeros((3, n_k))
    power_acc = np.zeros(n_g)
    lat = np.empty(1024)
    n_lat = 0
    late_new = 0
    arrivals = 0
    b_batches = 0.0
    c_work = 0.0

    h = q_h
    cap_q = q_t.shape[0]

    for _ in range(n_steps):
        # 1. A arrivals
        r = p[P_BASE_RATE] + p[P_WAVE_AMP] * np.sin(2 * np.pi * t / p[P_WAVE_PERIOD])
        if n_bursts:
            hit = False
            for i in range(n_bursts):
                b = bursts[i]
                if t >= b and t < b + p[P_BURST_LEN]:
                    hit = True
                    break
            if hit:
                r += p[P_BURST_EXTRA]
        if r < 0.0:                               # max(r, 0.0)
            r = 0.0
        n_arr = rng.poisson(r * dt)
        if n_arr:
            for k in range(n_k):
                others[k] = (0.0 + prev_usage[1, k]) + prev_usage[2, k]
            m = 0
            for k in range(n_k):
                if a_dom[k]:
                    sel[m] = others[k]
                    m += 1
            busy = _pairwise(sel, 0, m) / n_dom_div
            inflate = 1.0 + p[P_INTERFERENCE] * _pymin(busy, 1.0)
            cnt = 0.0
            for k in range(n_k):
                if elig[0, k] and others[k] > 0.05:
                    cnt += 1.0
            shared_frac = cnt / n_a_cpus
            svc = rng.exponential(p[P_SERVICE_MS], n_arr)
            u = rng.random(n_arr)
            ex = rng.exponential(p[P_DELAY_MEAN], n_arr)
            need = h + q_len + n_arr
            if need > cap_q:
                if q_len + n_arr <= cap_q // 2:
                    # compact to the front
                    for i in range(q_len):
                        q_t[i] = q_t[h + i]
                        q_rem[i] = q_rem[h + i]
                        q_delay[i] = q_delay[h + i]
                        q_late[i] = q_late[h + i]
                    h = 0
                else:
                    new_cap = 2 * (q_len + n_arr) + 64
                    nt = np.empty(new_cap)
                    nr = np.empty(new_cap)
                    nd = np.empty(new_cap)
                    nl = np.empty(new_cap, np.bool_)
                    for i in range(q_len):
                        nt[i] = q_t[h + i]
                        nr[i] = q_rem[h + i]
                        nd[i] = q_delay[h + i]
                        nl[i] = q_late[h + i]
                    q_t, q_rem, q_delay, q_late = nt, nr, nd, nl
                    h = 0
                    cap_q = new_cap
            j = h + q_len
            for i in range(n_arr):
                q_t[j] = t
                q_rem[j] = svc[i] * inflate
                q_delay[j] = ex[i] if u[i] < shared_frac else 0.0
                q_late[j] = False
                j += 1
            q_len += n_arr
            arrivals += n_arr

        # 2. demands
        n_head = q_len if q_len < n_k + 1 else n_k + 1
        s = 0.0
        for i in range(n_head):
            s += _pymin(q_rem[h + i], dt_ms)
        demand[0] = _pymin(s / dt_ms, float(n_k))
        demand[1] = _pymin((p[P_BUFFER_CAP] - b_buffer) * p[P_CPU_MS_PER_BATCH] / dt_ms, float(n_k))
        demand[2] = float(n_k)
        _allocate(elig, demand, weights, spare, usage, got, w_mat, col, offer, total, take, frac, used, row)
        for w in range(3):
            eff[w] = p[P_MIG_EFF] if t < migrating_until[w] else 1.0

        # 3. serve A: FIFO, at most one core per request
        cap = got[0] * eff[0] * dt_ms
        end = h + q_len
        i = h
        n_unf = 0
        while cap > 1e-12 and i < end:
            rem = q_rem[i]
            give = _pymin(_pymin(rem, dt_ms), cap)
            rem -= give
            cap -= give
            if rem <= 1e-9:
                lt = (t + dt - q_t[i]) * 1000.0 + q_delay[i]
                if n_lat == lat.shape[0]:
                    lat = _grow(lat, 2 * n_lat)
                lat[n_lat] = lt
                n_lat += 1
                if lt > p[P_SLO] and not q_late[i]:
                    late_new += 1
            else:
                if n_unf == unf_t.shape[0]:
                    sz = 2 * n_unf
                    unf_t = _grow(unf_t, sz)
                    unf_rem = _grow(unf_rem, sz)
                    unf_delay = _grow(unf_delay, sz)
                    unf_late = _grow(unf_late, sz)
                unf_t[n_unf] = q_t[i]
                unf_rem[n_unf] = rem
                unf_delay[n_unf] = q_delay[i]
                unf_late[n_unf] = q_late[i]
                n_unf += 1
            i += 1
        h = i - n_unf
        for j in range(n_unf):
            q_t[h + j] = unf_t[j]
            q_rem[h + j] = unf_rem[j]
            q_delay[h + j] = unf_delay[j]
            q_late[h + j] = unf_late[j]
        q_len = end - h

        # 4. B: CPUs fill the batch buffer, GPUs drain it.  5. C: CPU work plus GPU work.
        b_buffer = _pymin(p[P_BUFFER_CAP], b_buffer + got[1] * eff[1] * dt_ms / p[P_CPU_MS_PER_BATCH])
        c_work += got[2] * eff[2] * dt_ms
        for g in range(n_g):
            power[g] = 0.0
            f = _throttle(temps[g], p[P_T_THROTTLE], p[P_T_MAX], p[P_MIN_SPEED])
            owner = gpu_owner[g]
            if owner == 1:
                can = dt_ms / p[P_GPU_MS_PER_BATCH] * f
                done = _pymin(can, b_buffer)
                b_buffer -= done
                b_batches += done
                power[g] = done / (dt_ms / p[P_GPU_MS_PER_BATCH])
            elif owner == 2:
                c_work += p[P_GPU_CORE_EQUIV] * f * dt_ms
                power[g] = p[P_GPU_POWER] * f

        # 6. GPU temperatures
        for g in range(n_g):
            temps[g] += dt * (p[P_HEAT] * power[g] - p[P_COOL] * (temps[g] - p[P_AMBIENT]))

        for w in range(3):
            for k in range(n_k):
                usage_acc[w, k] += usage[w, k]
                prev_usage[w, k] = usage[w, k]
        for g in range(n_g):
            power_acc[g] += power[g]
        t += dt

    # Window end: requests still queued. Age so far (+ the wake-up delay they will pay) is a lower bound on
    # their latency; overdue ones are late already and counted once. The work left is summed left to right
    # (Python's builtin sum).
    slo = p[P_SLO]
    n_overdue = 0
    oldest = 0.0
    work = 0.0
    for i in range(h, h + q_len):
        age = (t - q_t[i]) * 1000.0 + q_delay[i]
        if i == h or age > oldest:
            oldest = age
        if age > slo:
            n_overdue += 1
            if not q_late[i]:
                q_late[i] = True
                late_new += 1
        work += q_rem[i]
    return (t, b_buffer, q_t, q_rem, q_delay, q_late, h, q_len, arrivals, b_batches, c_work,
            usage_acc, power_acc, lat[:n_lat].copy(), late_new, n_overdue, oldest, work)


def _params(spec, a_delay_mean: float) -> np.ndarray:
    p = np.empty(N_PARAMS)
    a, b, c, th = spec.a, spec.b, spec.c, spec.thermal
    p[P_BASE_RATE] = a.base_rate
    p[P_WAVE_AMP] = a.wave_amp
    p[P_WAVE_PERIOD] = a.wave_period
    p[P_BURST_LEN] = a.burst_len
    p[P_BURST_EXTRA] = a.burst_extra
    p[P_SERVICE_MS] = a.service_ms
    p[P_SLO] = a.slo_p99_ms
    p[P_INTERFERENCE] = a.interference
    p[P_BUFFER_CAP] = b.buffer_cap
    p[P_CPU_MS_PER_BATCH] = b.cpu_ms_per_batch
    p[P_GPU_MS_PER_BATCH] = b.gpu_ms_per_batch
    p[P_GPU_CORE_EQUIV] = c.gpu_core_equiv
    p[P_GPU_POWER] = c.gpu_power
    p[P_AMBIENT] = th.ambient
    p[P_HEAT] = th.heat
    p[P_COOL] = th.cool
    p[P_T_THROTTLE] = th.t_throttle
    p[P_T_MAX] = th.t_max
    p[P_MIN_SPEED] = th.min_speed
    p[P_MIG_EFF] = spec.costs.migration_efficiency
    p[P_DELAY_MEAN] = a_delay_mean
    p[P_DT] = spec.dt
    return p


class FastSimulator(Simulator):
    """`Simulator` with the per-step loop compiled; same telemetry, same generator state.

    The A queue lives in parallel arrays (arrival time, remaining ms, wake-up delay, counted late) between
    windows, so a long queue costs no per-window conversion. `queue` reads as a fresh deque of
    [arrival_t, remaining_ms, extra_delay_ms, counted_late] lists (a snapshot: changing it has no effect;
    assign a new queue instead).
    """

    @property
    def queue(self) -> deque:
        sl = slice(self._q_h, self._q_h + self._q_n)
        return deque(map(list, zip(self._q_t[sl].tolist(), self._q_rem[sl].tolist(),
                                   self._q_delay[sl].tolist(), self._q_late[sl].tolist())))

    @queue.setter
    def queue(self, q) -> None:
        rows = list(q)
        n = len(rows)
        cap = max(2 * n, 4096)
        self._q_t, self._q_rem, self._q_delay = np.empty(cap), np.empty(cap), np.empty(cap)
        self._q_late = np.zeros(cap, dtype=bool)
        if n:
            arr = np.array([r[:3] for r in rows], dtype=float)
            self._q_t[:n], self._q_rem[:n], self._q_delay[:n] = arr[:, 0], arr[:, 1], arr[:, 2]
            self._q_late[:n] = [bool(r[3]) for r in rows]
        self._q_h, self._q_n = 0, n

    def run_window(self, policy) -> dict:
        spec, dt = self.spec, self.spec.dt
        elig = policy.eligibility(self.n_cpus)
        weights = policy.shares()
        gpu_owner = policy.gpu_owner(self.n_gpus)
        a_class = policy.workloads[A].latency_class
        a_delay_mean = spec.sched.delay_ms[a_class]

        # Changing a workload's cpuset costs it some efficiency (cold caches, migrations).
        if self.prev_elig is not None:
            changed = (elig != self.prev_elig).any(axis=1)
            self.migrating_until[changed] = self.t + spec.costs.migration_penalty_s
        self.prev_elig = elig

        # CPUs in the L3 domains A runs in: other workloads there slow A down.
        a_dom = np.zeros(self.n_cpus, dtype=bool)
        for dom in spec.machine.l3_domains:
            if elig[A, list(dom)].any():
                a_dom[list(dom)] = True

        n_steps = int(round(spec.policy_period_s / dt))
        owners = np.array([-1 if o is None else o for o in gpu_owner], dtype=np.int64)
        self.prev_usage = np.array(self.prev_usage, dtype=float)       # own, contiguous copy
        self.temps = np.ascontiguousarray(self.temps, dtype=float)
        (t, b_buffer, self._q_t, self._q_rem, self._q_delay, self._q_late, self._q_h, self._q_n, arrivals,
         b_batches, c_work, usage_sum, power_sum, lat, late_new, n_overdue, oldest, work) = _run_steps(
            self.rng, n_steps, float(self.t), _params(spec, a_delay_mean),
            np.asarray(self.bursts, dtype=float), np.ascontiguousarray(elig),
            np.ascontiguousarray(weights, dtype=float), owners, a_dom,
            np.asarray(self.migrating_until, dtype=float), self.prev_usage, self.temps, float(self.b_buffer),
            self._q_t, self._q_rem, self._q_delay, self._q_late, self._q_h, self._q_n)
        self.t = float(t)
        self.b_buffer = float(b_buffer)
        arrivals = int(arrivals)
        b_batches = float(b_batches)
        c_work = float(c_work)

        # The parent's telemetry, from the same values.
        self.all_latencies.append(lat)
        slo = spec.a.slo_p99_ms
        ontime = int((lat <= slo).sum())
        win = spec.policy_period_s
        u = usage_sum / n_steps
        return {
            "t": round(self.t, 3),
            "window_s": win,
            "a": {
                "arrivals": arrivals,
                "rate": arrivals / win,
                "p50_ms": float(np.percentile(lat, 50)) if len(lat) else 0.0,
                "p99_ms": float(np.percentile(lat, 99)) if len(lat) else 0.0,
                "p99_valid": bool(len(lat)),
                "completions": int(len(lat)),
                "completed_ontime": ontime,
                "late_new": int(late_new),
                "queue_overdue": int(n_overdue),
                "queue_oldest_age_ms": float(oldest) if self._q_n else 0.0,
                "queue_len": int(self._q_n),
                "queue_work_ms": float(work),
                "cpu_cores": float(u[A].sum()),
            },
            "b": {"batches": b_batches, "rate": b_batches / win,
                  "buffer": self.b_buffer, "cpu_cores": float(u[B].sum())},
            "c": {"work_core_ms": c_work, "cpu_cores": float(u[C].sum())},
            "cpu_util": u.sum(axis=0).round(3).tolist(),
            "cpu_usage_by_workload": u.round(3).tolist(),
            "gpu_temp": self.temps.round(2).tolist(),
            "gpu_power": (power_sum / n_steps).round(3).tolist(),
        }
