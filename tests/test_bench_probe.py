from experiments import probe as pb

TASKS = {f"retail:{t}" for t in range(111)}


def test_split3_is_disjoint_complete_and_sized():
    tr, ca, te = pb.split3(TASKS, 0, pb.PREREG["splits"]["primary"])
    assert (len(tr), len(ca), len(te)) == (44, 33, 34)
    assert not (tr & ca or tr & te or ca & te) and tr | ca | te == TASKS
    assert [len(s) for s in pb.split3(TASKS, 0, pb.PREREG["splits"]["secondary"])] == [55, 27, 29]


def test_split3_is_deterministic_and_seed_dependent():
    assert pb.split3(TASKS, 7, (0.4, 0.3, 0.3)) == pb.split3(set(sorted(TASKS, reverse=True)), 7, (0.4, 0.3, 0.3))
    assert pb.split3(TASKS, 7, (0.4, 0.3, 0.3)) != pb.split3(TASKS, 8, (0.4, 0.3, 0.3))


def test_calib_heavy_shares_test_tasks_with_primary():
    for seed in range(5):
        assert pb.split3(TASKS, seed, (0.4, 0.3, 0.3))[2] == pb.split3(TASKS, seed, (0.2, 0.5, 0.3))[2]


def r(task, p=0.99, ok=True):
    return {"task_id": task, "p_max": p, "choice": "a", "label": "a" if ok else "b"}


def test_serve_and_aggregate():
    cal = [r(f"t{i}") for i in range(60)]
    test = [r("x"), r("y", ok=False), r("z", p=0.1)]
    s = pb.serve(cal, test, 0.05, 0.1, "hb")
    assert (s["served"], s["wrong"], s["lam"], s["cal_sessions"]) == (2, 1, 0.5, 60)
    assert pb.serve(cal[:33], test, 0.05, 0.1, "hb")["lam"] is None  # 33 clean sessions cannot pass HB at α=0.05
    m = pb.aggregate([s, {**s, "served": 0, "wrong": 0, "lam": None}], 0.05)
    assert m["coverage"] == 2 / 6 and m["held"] == 0 and m["serving"] == 1 and m["pooled_disagreement"] == 0.5


def test_ece_and_quality():
    assert pb.ece([1.0, 1.0], [True, True]) == 0.0
    assert abs(pb.ece([0.9, 0.9], [True, False]) - 0.4) < 1e-9
    q = pb.quality([r("a"), r("a", ok=False), r("a", p=0.5)])
    assert q["top1"] == 2 / 3 and q["disagreement_ge_0.9"] == 0.5
