"""P4: канал связи, знание наблюдателей, правила CBBA, децентрализованные и гибридные прогоны."""
import numpy as np
import pytest

from swarmsim import apply_overrides, run
from swarmsim.alloc.cbba import DONE, RUN, CBBANode, cbba_actions
from swarmsim.knowledge import RUNNING, KnowledgeBase
from swarmsim.network import Network

NET = {"kind": "network", "latency": 0.05, "jitter": 0.02, "loss": 0.1}


def _net(small, alloc, **comms):
    return apply_overrides(small, {"allocator": {"kind": alloc}, "comms": {**NET, **comms}})


# ------------------------------------------------------------------ канал
def test_link_matrix_range_and_outage():
    net = Network(3, np.array([0.0, 0.0]), {"range": 100, "range_gcs": 300, "gcs_outages": [[10, 20]]},
                  np.random.default_rng(0), ideal=False)
    net.update_positions(np.array([[0, 0], [90, 0], [250, 0]], float), np.ones(3, bool))
    ok = net.link_matrix(0.0)
    assert ok[0, 1] and not ok[0, 2] and not ok[1, 2]       # агент↔агент — дальность 100
    assert ok[3, 2] and ok[2, 3]                             # станция дальше — 300
    ok = net.link_matrix(15.0)
    assert not ok[3].any() and not ok[:, 3].any()            # разрыв станции
    assert not ok.diagonal().any()


def test_messages_arrive_with_latency_and_are_lost_at_rate():
    net = Network(2, np.zeros(2), {"latency": 0.1, "jitter": 0.0, "loss": 0.3}, np.random.default_rng(1), ideal=False)
    net.update_positions(np.zeros((2, 2)), np.ones(2, bool))
    link = net.link_matrix(0.0)
    for _ in range(2000):
        net.send(0.0, 0, [1], "event", None, 1, link)
    assert net.deliver(0.05) == []                          # ещё в пути
    got = len(net.deliver(10.0))
    assert got / 2000 == pytest.approx(0.7, abs=0.04)


# ------------------------------------------------------------------ знание
def _hb(cur=-1, wp=0, plan=(), done=()):
    return {"p": np.zeros(2), "v": np.zeros(2), "e": 1.0, "mode": 1, "cur": cur, "wp": wp,
            "lowbat": False, "vc": 10.0, "pv": 0, "plan": list(plan), "done": np.array(done, int)}


def test_done_is_monotonic_and_propagates_transitively():
    kb = KnowledgeBase(3, 4, timeout=1.5)
    got = np.zeros((4, 4), bool)
    got[0, 1] = True                                         # 1 слышит 0
    kb.on_heartbeats(1.0, got, [_hb(done=[2]), None, None])
    assert kb.status[1, 2] == 3
    got = np.zeros((4, 4), bool)
    got[1, 2] = True                                         # 2 слышит 1 (но не 0)
    kb.on_heartbeats(2.0, got, [None, _hb(done=np.flatnonzero(kb.status[1] == 3)), None])
    assert kb.status[2, 2] == 3                              # транзитом
    kb.on_heartbeats(3.0, np.ones((4, 4), bool), [_hb(plan=[2]), None, None])
    assert kb.status[1, 2] == 3                              # «выполнена» не откатывается


def test_running_owner_not_overwritten_by_conflict_loser():
    kb = KnowledgeBase(3, 2, timeout=1.5)
    got = np.zeros((4, 4), bool)
    got[0, 2] = got[1, 2] = True
    kb.on_heartbeats(1.0, got, [_hb(cur=0, wp=5), _hb(cur=0, wp=1), None])
    assert kb.status[2, 0] == RUNNING and kb.owner[2, 0] == 0   # победитель — больший прогресс
    kb.on_heartbeats(1.2, got, [None, _hb(), None])             # проигравший уступил
    assert kb.owner[2, 0] == 0                                   # задача не «освободилась»


def test_timeouts_matrix_controls_loss():
    kb = KnowledgeBase(2, 1, timeout=1.5)
    kb.last_heard[:] = 0.0
    tmo = np.full((3, 2), 60.0)
    tmo[0, 1] = 1.5                                          # 0 должен слышать 1 напрямую
    out = kb.detect(5.0, timeouts=tmo)
    assert (0, 1, "lost") in out and (2, 1, "lost") not in out  # станция: вне зоны — не отказ


# ------------------------------------------------------------------ CBBA
def test_cbba_rules_basic():
    si = np.zeros(3)
    sk = np.ones(3)
    # отправитель 1 считает победителем себя с большей ставкой, у меня (0) — я: принять
    act = cbba_actions(0, 1, np.array([0.5]), np.array([1]), np.array([0.3]), np.array([0]), sk, si)
    assert act[0] == 1
    # меньшая ставка — оставить своё
    act = cbba_actions(0, 1, np.array([0.2]), np.array([1]), np.array([0.3]), np.array([0]), sk, si)
    assert act[0] == 0
    # отправитель считает победителем меня, а я — его: сбросить
    act = cbba_actions(0, 1, np.array([0.2]), np.array([0]), np.array([0.3]), np.array([1]), sk, si)
    assert act[0] == 2


def test_run_and_done_bids_cannot_be_outbid():
    nd = CBBANode(0, 2, 3, bundle_size=2, hysteresis=0.05)
    nd.set_current(1)
    nd.set_done(2, 1)
    y = np.array([0.0, 5.0, 5.0])
    nd.merge(1, 1.0, y, np.array([-1, 1, 1]), np.ones(2))
    assert nd.y[1] == RUN and nd.z[1] == 0
    assert nd.y[2] == DONE


def test_lost_peer_claims_are_filtered():
    nd = CBBANode(0, 3, 2, bundle_size=2, hysteresis=0.05)
    nd.release_peer(2)
    nd.merge(1, 1.0, np.array([0.4, 0.0]), np.array([2, -1]), np.ones(3))
    assert nd.z[0] == -1                                     # заявка «мёртвого» не воскресла


# ------------------------------------------------------------------ прогоны
@pytest.mark.parametrize("alloc", ["cbba", "hybrid", "central_greedy"])
def test_lossy_network_completes_without_duplicates(small, alloc):
    s = run(_net(small, alloc, range=350), 1).summary
    assert s["valid"] and s["CR"] == 1.0
    assert s["N_dup"] == 0 and s["N_lost"] == 0
    assert s["N_msg"] > 0 and s["B_per_agent"] > 0


def test_gcs_outage_decentral_continues_central_stalls(small):
    out = {"gcs_outages": [[20, 400]]}
    fault = {"faults.scheduled": [{"t": 60, "agent": 1, "detect": "silent"}]}
    hyb = run(apply_overrides(_net(small, "hybrid", **out), fault), 1).summary
    cen = run(apply_overrides(_net(small, "central_greedy", **out), fault), 1).summary
    assert hyb["CR"] == 1.0 and hyb["N_dup"] == 0 and hyb["recovered_frac"] == 1.0
    assert hyb["T_recovery_max"] < 5.0                      # подхватили без станции
    assert cen["T_recovery_max"] is None or cen["T_recovery_max"] > 300   # ждали станцию
    assert hyb["N_false_loss"] == 0                          # станция во время разрыва никого не «теряет»


def test_ideal_network_cbba_is_deterministic(small):
    a = run(_net(small, "cbba", loss=0.0), 3).summary
    b = run(_net(small, "cbba", loss=0.0), 3).summary
    assert a["CR"] == b["CR"] and a["makespan"] == b["makespan"] and a["N_msg"] == b["N_msg"]
