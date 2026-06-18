"""
Symulacja zdarzeniowa policera Token Bucket
Sieci Wielousługowe -- projekt (Jan Gajownik, Mikita Zhydko)

Założenia (zgodne ze schematem blokowym):
  * Symulacja zdarzeniowa (discrete-event) z kalendarzem zdarzeń (kopiec, heapq).
  * Trzy typy zdarzeń: przybycie pakietu, zmiana stanu źródła, koniec symulacji.
    Po pobraniu najbliższego zdarzenia sterowanie idzie jedną z trzech "nóg".
  * Źródło ON-OFF DETERMINISTYCZNE: stałe T_ON i T_OFF (bez losowania) -> przebieg
    jest w pełni powtarzalny.
  * STAŁA przepływność w stanie ON (CBR): pakiety o stałym rozmiarze k w stałym
    odstępie  dt = k / C, gdzie C to przepływność szczytowa źródła w ON.
  * Kolejne przybycie planowane jest NA POCZĄTKU obsługi przybycia -- niezależnie
    od decyzji policera (przepuść/odrzuć), o ile mieści się w bieżącym oknie ON
    (t + dt < t_OFF). Dzięki temu odrzucenie pakietu nie urywa łańcucha ruchu.
  * Token bucket:  TB = min(BS, TB + R * (t - t_prev)); na starcie kubełek pełny.
    Generator tokenów działa w sposób CIĄGŁY (również w czasie OFF).
  * P_drop = N_drop / N_all * 100 %.

Jednostki: spójnie w bitach i sekundach (BS, k w bitach; R, C w bit/s; czasy w s).
Można równie dobrze liczyć w bajtach -- byle konsekwentnie.
"""

from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
#  Parametry i wynik
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    BS: float           # pojemność kubełka                     [bit]
    R: float            # tempo napełniania tokenami            [bit/s]
    k: float            # rozmiar pakietu                       [bit]
    C: float            # przepływność źródła w ON (peak rate)  [bit/s]
    T_ON: float         # czas trwania fazy ON                  [s]
    T_OFF: float        # czas trwania fazy OFF                 [s]
    T_sim: float        # czas symulacji                        [s]
    start_ON: bool = True   # czy źródło startuje w stanie ON

    def validate(self) -> None:
        for name in ("BS", "R", "k", "C", "T_ON", "T_OFF", "T_sim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Parametr {name} musi być dodatni.")
        if self.k > self.BS:
            raise ValueError("k > BS: pojedynczy pakiet nigdy się nie zmieści -> "
                             "100% strat. Zwiększ BS lub zmniejsz k.")


@dataclass
class Result:
    N_all: int
    N_pass: int
    N_drop: int
    P_drop: float                       # [%]
    log: list | None = None             # opcjonalny ślad: (t, 'pass'/'drop', TB_po)

    def summary(self, p: Params | None = None) -> str:
        lines = [
            "=" * 52,
            "  WYNIK SYMULACJI -- policer Token Bucket",
            "=" * 52,
            f"  N_all  (wszystkie pakiety) : {self.N_all}",
            f"  N_pass (przepuszczone)     : {self.N_pass}",
            f"  N_drop (odrzucone)         : {self.N_drop}",
            f"  P_drop                     : {self.P_drop:.3f} %",
        ]
        if p is not None:
            avg_in = p.C * p.T_ON / (p.T_ON + p.T_OFF)
            # Dolne ograniczenie strat (kubełek "nieskończony" -> brak utraty tokenów):
            #   w długim czasie nie da się przepuścić więcej niż R.
            p_lower = max(0.0, 1.0 - p.R / avg_in) * 100.0
            # Oszacowanie stanu ustalonego z uwzględnieniem PRZEPEŁNIENIA kubełka w OFF:
            #   poziom na początku okna ON  B0 = min(BS, R*T_OFF)  (kubełek opróżniony w ON),
            #   tokeny dostępne w oknie ON  = B0 + R*T_ON,  ruch wejściowy = C*T_ON.
            B0 = min(p.BS, p.R * p.T_OFF)
            avail = B0 + p.R * p.T_ON
            inp = p.C * p.T_ON
            p_est = max(0.0, 1.0 - avail / inp) * 100.0 if inp > 0 else 0.0
            lines += [
                "-" * 52,
                f"  śr. przepływność wej.      : {avg_in:.3e} bit/s",
                f"  P_drop (dolne ogr.)        : {p_lower:.3f} %  (kubełek nieskończony)",
                f"  P_drop (stan ustalony)     : {p_est:.3f} %  (z przepełnieniem w OFF)",
            ]
        lines.append("=" * 52)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Symulator
# --------------------------------------------------------------------------- #
class TokenBucketSim:
    """Symulacja zdarzeniowa policera Token Bucket ze źródłem ON-OFF (CBR)."""

    # Typy zdarzeń
    ARRIVAL = "arrival"
    STATE   = "state"
    END     = "end"

    # Priorytet przy IDENTYCZNYM czasie zdarzenia (mniejszy = wcześniej):
    # zmiana stanu przed przybyciem (by stan był aktualny), koniec na końcu.
    _PRIO = {STATE: 0, ARRIVAL: 1, END: 2}

    EPS = 1e-12  # margines na błędy zmiennoprzecinkowe przy porównaniach czasu

    def __init__(self, p: Params, record: bool = False):
        p.validate()
        self.p = p
        self.record = record

    # --- kalendarz zdarzeń --------------------------------------------------
    def _schedule(self, etype: str, time: float) -> None:
        heapq.heappush(self._heap, (time, self._PRIO[etype], self._seq, etype))
        self._seq += 1

    # --- główna pętla -------------------------------------------------------
    def run(self) -> Result:
        p = self.p

        # ---------- Inicjalizacja (zegar=0, TB=BS, liczniki=0) ----------
        self.t = 0.0
        self.tb = p.BS                       # kubełek na starcie pełny
        self.t_prev = 0.0                    # czas ostatniej aktualizacji tokenów
        self.N_all = self.N_pass = self.N_drop = 0
        self.dt = p.k / p.C                  # stały odstęp przybyć (CBR w ON)
        self.state = "ON" if p.start_ON else "OFF"
        self._heap: list = []
        self._seq = 0
        # t_OFF -- czas najbliższego przejścia ON->OFF (strażnik planowania przybyć)
        self.t_OFF = p.T_ON if self.state == "ON" else math.inf

        log: list | None = [] if self.record else None

        # ---------- Zaplanuj zdarzenia początkowe ----------
        self._schedule(self.END, p.T_sim)
        if self.state == "ON":
            self._schedule(self.ARRIVAL, 0.0)        # pierwsze przybycie
            self._schedule(self.STATE,  p.T_ON)      # pierwsza zmiana ON->OFF
        else:
            self._schedule(self.STATE,  p.T_OFF)     # pierwsza zmiana OFF->ON

        # ---------- Pętla: pobierz najbliższe zdarzenie -> dispatch ----------
        while self._heap:
            t, _, _, etype = heapq.heappop(self._heap)
            self.t = t                                # przesuń zegar
            if etype == self.END:                     # noga 3
                break
            elif etype == self.ARRIVAL:               # noga 1
                self._on_arrival(t, log)
            else:                                     # noga 2
                self._on_state_change(t)

        # ---------- Generator raportu ----------
        P_drop = (self.N_drop / self.N_all * 100.0) if self.N_all else 0.0
        return Result(self.N_all, self.N_pass, self.N_drop, P_drop, log)

    # --- NOGA 1: przybycie pakietu -----------------------------------------
    def _on_arrival(self, t: float, log: list | None) -> None:
        p = self.p

        # 1) NA POCZĄTKU: zaplanuj kolejne przybycie (niezależnie od decyzji),
        #    o ile mieści się jeszcze w bieżącym oknie ON.
        t_next = t + self.dt
        if self.state == "ON" and t_next < self.t_OFF - self.EPS:
            self._schedule(self.ARRIVAL, t_next)

        # 2) Uzupełnij tokeny (napełnianie ciągłe, z ograniczeniem do BS)
        self.tb = min(p.BS, self.tb + p.R * (t - self.t_prev))
        self.t_prev = t

        # 3) Zlicz wszystkie pakiety
        self.N_all += 1

        # 4) Decyzja policera
        if p.k <= self.tb:                    # dość tokenów -> przepuść
            self.tb -= p.k
            self.N_pass += 1
            decision = "pass"
        else:                                 # za mało tokenów -> odrzuć (TB bez zmian)
            self.N_drop += 1
            decision = "drop"

        if log is not None:
            log.append((t, decision, self.tb))

    # --- NOGA 2: zmiana stanu źródła (deterministyczna) --------------------
    def _on_state_change(self, t: float) -> None:
        p = self.p
        if self.state == "ON":
            # ON -> OFF
            self.state = "OFF"
            self.t_OFF = math.inf
            self._schedule(self.STATE, t + p.T_OFF)
        else:
            # OFF -> ON
            self.state = "ON"
            self.t_OFF = t + p.T_ON
            self._schedule(self.ARRIVAL, t)              # pierwsze przybycie okna ON
            self._schedule(self.STATE,  t + p.T_ON)      # następna zmiana ON->OFF


# --------------------------------------------------------------------------- #
#  Narzędzie do badań: przemiatanie parametru
# --------------------------------------------------------------------------- #
def sweep(base: Params, param: str, values) -> list[tuple[float, float]]:
    """Zwraca listę (wartość_parametru, P_drop[%]) dla zadanego zakresu wartości."""
    out = []
    for v in values:
        kwargs = base.__dict__.copy()
        kwargs[param] = v
        res = TokenBucketSim(Params(**kwargs)).run()
        out.append((v, res.P_drop))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Symulacja policera Token Bucket dla źródła ON-OFF."
    )
    parser.add_argument("--BS", type=float, default=50_000, help="Pojemność kubełka [bit]")
    parser.add_argument("--R", type=float, default=4_000_000, help="Tempo napełniania tokenami [bit/s]")
    parser.add_argument("--k", type=float, default=1_000, help="Rozmiar pakietu [bit]")
    parser.add_argument("--C", type=float, default=10_000_000, help="Przepływność źródła w ON [bit/s]")
    parser.add_argument("--T-ON", dest="T_ON", type=float, default=0.1, help="Czas trwania fazy ON [s]")
    parser.add_argument("--T-OFF", dest="T_OFF", type=float, default=0.1, help="Czas trwania fazy OFF [s]")
    parser.add_argument("--T-sim", dest="T_sim", type=float, default=1.0, help="Czas symulacji [s]")
    parser.add_argument("--start-OFF", dest="start_ON", action="store_false", help="Start źródła w stanie OFF")
    return parser.parse_args()


def save_plot(x_values, y_values, xlabel: str, title: str, filename: str, vline: float | None = None,
             vline_label: str | None = None) -> None:
    """Zapisuje prosty wykres liniowy do pliku PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.2))
    plt.plot(x_values, y_values, "o-", color="#2b6cb0")
    if vline is not None:
        plt.axvline(vline, ls="--", color="#888", label=vline_label)
    plt.xlabel(xlabel)
    plt.ylabel("P_drop  [%]")
    plt.title(title)
    plt.grid(alpha=0.3)
    if vline_label is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = parse_args()

    # Przykładowy scenariusz, ale z możliwością wyboru R i C z CLI.
    p = Params(
        BS=args.BS,
        R=args.R,
        k=args.k,
        C=args.C,
        T_ON=args.T_ON,
        T_OFF=args.T_OFF,
        T_sim=args.T_sim,
        start_ON=args.start_ON,
    )

    res = TokenBucketSim(p, record=True).run()
    print(res.summary(p))

    # --- przykładowe badanie: P_drop w funkcji R (tempa tokenów) ---
    Rs = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 10e6]
    data = sweep(p, "R", Rs)
    print("\nP_drop(R):")
    for R, pd in data:
        print(f"  R = {R/1e6:5.1f} Mbit/s  ->  P_drop = {pd:6.2f} %")

    # --- przykładowe badanie: P_drop w funkcji C (przepływności źródła) ---
    Cs = [2e6, 4e6, 6e6, 8e6, 10e6, 12e6, 14e6, 16e6, 18e6, 20e6]
    data_c = sweep(p, "C", Cs)
    print("\nP_drop(C):")
    for C, pd in data_c:
        print(f"  C = {C/1e6:5.1f} Mbit/s  ->  P_drop = {pd:6.2f} %")

    # --- przykładowe badanie: P_drop w funkcji BS (pojemności kubełka) ---
    BSs = [10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 80_000, 100_000, 150_000, 200_000]
    data_bs = sweep(p, "BS", BSs)
    print("\nP_drop(BS):")
    for BS, pd in data_bs:
        print(f"  BS = {BS/1e3:6.1f} kbit  ->  P_drop = {pd:6.2f} %")

    # Wykres (jeśli dostępny matplotlib)
    try:
        xs = [R / 1e6 for R, _ in data]
        ys = [pd for _, pd in data]
        save_plot(
            xs,
            ys,
            "R  [Mbit/s]",
            "P_drop w funkcji tempa tokenów R",
            "pdrop_vs_R.png",
            vline=p.C * p.T_ON / (p.T_ON + p.T_OFF) / 1e6,
            vline_label="śr. przepływność wejściowa",
        )

        xs_c = [C / 1e6 for C, _ in data_c]
        ys_c = [pd for _, pd in data_c]
        save_plot(
            xs_c,
            ys_c,
            "C  [Mbit/s]",
            "P_drop w funkcji przepływności źródła C",
            "pdrop_vs_C.png",
        )

        xs_bs = [BS / 1e3 for BS, _ in data_bs]
        ys_bs = [pd for _, pd in data_bs]
        save_plot(
            xs_bs,
            ys_bs,
            "BS  [kbit]",
            "P_drop w funkcji pojemności kubełka BS",
            "pdrop_vs_BS.png",
        )
        print("\nZapisano wykresy: pdrop_vs_R.png, pdrop_vs_C.png, pdrop_vs_BS.png")
    except Exception as e:
        print(f"\n(pominięto wykres: {e})")