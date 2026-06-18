"""
Testy jednostkowe symulacji policera Token Bucket.

Uruchomienie:
    python3 -m unittest -v test_token_bucket_policer.py

Każdy test weryfikuje jedno z założeń algorytmu opisanych w schemacie blokowym,
a nie tylko fakt, że kod się wykonuje.
"""

import math
import unittest

from token_bucket_policer import Params, Result, TokenBucketSim, sweep


def base_params(**over) -> Params:
    """Sensowny scenariusz bazowy; pola można nadpisać argumentami."""
    cfg = dict(
        BS=50_000, R=4_000_000, k=1_000, C=10_000_000,
        T_ON=0.1, T_OFF=0.1, T_sim=1.0, start_ON=True,
    )
    cfg.update(over)
    return Params(**cfg)


class TestInvariants(unittest.TestCase):
    """Niezmienniki, które muszą zachodzić w KAŻDYM przebiegu."""

    def test_counters_sum(self):
        # N_all = N_pass + N_drop -- każdy pakiet jest albo przepuszczony, albo odrzucony.
        res = TokenBucketSim(base_params()).run()
        self.assertEqual(res.N_all, res.N_pass + res.N_drop)

    def test_pdrop_formula(self):
        # P_drop = N_drop / N_all * 100 %.
        res = TokenBucketSim(base_params()).run()
        self.assertAlmostEqual(res.P_drop, res.N_drop / res.N_all * 100.0, places=9)

    def test_pdrop_in_range(self):
        # P_drop zawsze w [0, 100].
        res = TokenBucketSim(base_params()).run()
        self.assertGreaterEqual(res.P_drop, 0.0)
        self.assertLessEqual(res.P_drop, 100.0)


class TestTokenAccounting(unittest.TestCase):
    """Poprawność gospodarki tokenami w kubełku."""

    def test_no_drops_when_rate_sufficient(self):
        # R >= C oraz duży kubełek -> nic nie powinno być odrzucone.
        res = TokenBucketSim(base_params(R=20_000_000, BS=1_000_000)).run()
        self.assertEqual(res.N_drop, 0)
        self.assertEqual(res.N_pass, res.N_all)
        self.assertEqual(res.P_drop, 0.0)

    def test_only_initial_bucket_passes_without_refill(self):
        # Praktycznie brak napełniania (R~0) i BS = 3*k -> przejdą dokładnie 3 pakiety
        # (te, które mieszczą się w początkowym, pełnym kubełku), reszta odrzucona.
        p = base_params(R=1e-9, BS=3_000, k=1_000)
        res = TokenBucketSim(p).run()
        self.assertEqual(res.N_pass, 3)
        self.assertGreater(res.N_drop, 0)

    def test_drop_leaves_bucket_unchanged(self):
        # Po odrzuceniu kubełek nie jest pomniejszany: przy R~0 i BS=2.5*k przejdą
        # 2 pakiety, a poziom kubełka nigdy nie spadnie poniżej 0 ani nie "przeskoczy".
        p = base_params(R=1e-9, BS=2_500, k=1_000)
        res = TokenBucketSim(p, record=True).run()
        self.assertEqual(res.N_pass, 2)
        # poziom kubełka w logu nigdy ujemny
        for _, _, tb in res.log:
            self.assertGreaterEqual(tb, -1e-6)

    def test_token_cap_at_BS(self):
        # Po długim OFF (R*T_OFF >> BS) kubełek nasyca się do BS, nie więcej.
        # Pierwszy pakiet każdego okna ON powinien przejść, a poziom po nim = BS - k.
        p = base_params(BS=50_000, R=4_000_000, T_OFF=0.1)
        res = TokenBucketSim(p, record=True).run()
        log = res.log
        # pierwsze przybycia okien ON = poprzedzone dużą przerwą (start nowego okna)
        starts = [log[0]]
        for prev, cur in zip(log, log[1:]):
            if cur[0] - prev[0] > p.T_OFF / 2:   # duża luka -> nowe okno ON
                starts.append(cur)
        self.assertGreaterEqual(len(starts), 2)
        for t, decision, tb in starts:
            self.assertEqual(decision, "pass")
            self.assertAlmostEqual(tb, p.BS - p.k, delta=1.0)


class TestSchedulingAtBeginning(unittest.TestCase):
    """Kluczowe założenie: kolejne przybycie planowane NA POCZĄTKU obsługi,
    niezależnie od decyzji policera -> odrzucenia NIE urywają strumienia ruchu."""

    def test_drops_do_not_break_arrival_chain(self):
        # Kubełek na jeden pakiet (BS=k) i znikome napełnianie -> pakiet 1 przechodzi,
        # a praktycznie wszystkie kolejne są odrzucane. Mimo to przybycia muszą
        # napływać przez cały czas symulacji.
        p = base_params(BS=1_000, k=1_000, R=1e-9)
        res = TokenBucketSim(p).run()
        # gdyby planowanie było na ścieżce "przepuść", łańcuch urwałby się po ~1 pakiecie
        self.assertGreater(res.N_all, 1000)
        self.assertGreater(res.N_drop, 0)
        self.assertGreaterEqual(res.N_pass, 1)

    def test_arrival_count_independent_of_drops(self):
        # Liczba przybyć (N_all) zależy tylko od źródła (C, T_ON, T_OFF, T_sim),
        # a nie od tego, ile pakietów przeszło. Ten sam ruch przy różnym R -> to samo N_all.
        n_low = TokenBucketSim(base_params(R=1e-9)).run().N_all      # prawie same dropy
        n_high = TokenBucketSim(base_params(R=20_000_000)).run().N_all  # zero dropów
        self.assertEqual(n_low, n_high)


class TestDeterminism(unittest.TestCase):
    """Źródło deterministyczne -> przebieg w pełni powtarzalny."""

    def test_reproducible(self):
        r1 = TokenBucketSim(base_params()).run()
        r2 = TokenBucketSim(base_params()).run()
        self.assertEqual((r1.N_all, r1.N_pass, r1.N_drop), (r2.N_all, r2.N_pass, r2.N_drop))
        self.assertEqual(r1.P_drop, r2.P_drop)


class TestSourceModel(unittest.TestCase):
    """Model źródła ON-OFF: stała przepływność (CBR) i poprawne fazy."""

    def test_constant_interarrival_in_ON(self):
        # W obrębie okna ON odstępy między przybyciami są stałe i równe dt = k/C.
        p = base_params()
        dt = p.k / p.C
        res = TokenBucketSim(p, record=True).run()
        times = [t for t, _, _ in res.log]
        gaps = [b - a for a, b in zip(times, times[1:])]
        in_window = [g for g in gaps if g < p.T_ON / 2]   # pomiń przerwy OFF
        self.assertTrue(in_window)
        for g in in_window:
            self.assertAlmostEqual(g, dt, delta=dt * 1e-6)

    def test_start_off_no_traffic_before_first_on(self):
        # Start w OFF -> brak przybyć przed pierwszym przejściem OFF->ON (t = T_OFF).
        p = base_params(start_ON=False, T_OFF=0.2)
        res = TokenBucketSim(p, record=True).run()
        first_arrival = res.log[0][0]
        self.assertGreaterEqual(first_arrival, p.T_OFF - 1e-9)

    def test_packets_per_on_window(self):
        # Liczba pakietów na okno ON ~ T_ON / dt (z tolerancją na granicę/FP).
        p = base_params(T_sim=0.1, T_ON=0.1, T_OFF=0.1)  # jedno okno ON
        dt = p.k / p.C
        res = TokenBucketSim(p).run()
        expected = p.T_ON / dt
        self.assertAlmostEqual(res.N_all, expected, delta=2)


class TestAnalyticAgreement(unittest.TestCase):
    """Zgodność z analizą w stanie ustalonym (walidacja modelu)."""

    def test_overflow_regime(self):
        # R*T_OFF >> BS: tokeny z OFF się marnują.
        # P_drop ~ 1 - (BS + R*T_ON)/(C*T_ON).
        p = base_params(BS=50_000, R=4_000_000, T_ON=0.1, T_OFF=0.1, C=10_000_000)
        res = TokenBucketSim(p).run()
        avail = min(p.BS, p.R * p.T_OFF) + p.R * p.T_ON
        est = max(0.0, 1.0 - avail / (p.C * p.T_ON)) * 100.0
        self.assertAlmostEqual(res.P_drop, est, delta=1.0)  # <1 pkt proc.

    def test_no_overflow_regime(self):
        # R*T_OFF < BS i kubełek opróżniany w ON -> P_drop ~ 1 - R/śr_wej.
        # Dużo cykli, by zaniknął wpływ pełnego kubełka na starcie.
        p = base_params(BS=600_000, R=4_000_000, T_ON=0.1, T_OFF=0.1,
                        C=10_000_000, T_sim=4.0)
        res = TokenBucketSim(p).run()
        avg_in = p.C * p.T_ON / (p.T_ON + p.T_OFF)
        est = max(0.0, 1.0 - p.R / avg_in) * 100.0
        self.assertAlmostEqual(res.P_drop, est, delta=2.0)

    def test_pdrop_monotonic_in_R(self):
        # Większe R (więcej tokenów) -> nie więcej strat. P_drop nierosnące względem R.
        data = sweep(base_params(), "R", [1e6, 2e6, 4e6, 6e6, 8e6, 10e6])
        pds = [pd for _, pd in data]
        for a, b in zip(pds, pds[1:]):
            self.assertGreaterEqual(a + 1e-9, b)


class TestValidation(unittest.TestCase):
    """Walidacja parametrów wejściowych."""

    def test_packet_larger_than_bucket(self):
        with self.assertRaises(ValueError):
            TokenBucketSim(base_params(k=2_000, BS=1_000)).run()

    def test_nonpositive_param(self):
        for bad in (dict(R=0), dict(BS=-1), dict(C=0), dict(T_ON=0)):
            with self.assertRaises(ValueError):
                TokenBucketSim(base_params(**bad)).run()


if __name__ == "__main__":
    unittest.main(verbosity=2)