import math
from dataclasses import dataclass
from typing import Optional


# =====================================================================
# SELEKCIJA - pristupni algoritmi (BP2 - 5. Obrada upita, slajd 5-9)
# =====================================================================

def linear_scan_cost(b_r: int, is_key: bool = False) -> float:
    """A1 - linearna pretraga (file scan). Slajd 5.
    Ako je selekcija po ključnom atributu, pretraga može stati na pola."""
    return (b_r / 2 if is_key else b_r) + 1  # +1 pretraga (search)


def clustering_index_key_cost(h_i: int) -> float:
    """A2 - grupišući indeks, jednakost po ključnom atributu. Slajd 6.
    h_i mora već biti umanjeno za pretpostavku da su unutrašnji čvorovi
    stabla u memoriji (vidi napomenu o konvenciji ispod)."""
    return h_i + 1


def clustering_index_nonkey_cost(h_i: int, b: int) -> float:
    """A3 - grupišući indeks, jednakost po ne-ključnom atributu. Slajd 6.
    b = broj blokova koji sadrže zapise koji zadovoljavaju uslov."""
    return h_i + b


def secondary_index_cost(h_i: int, n: int, is_key: bool = False) -> float:
    """A4 - sekundarni indeks. Slajd 6.
    is_key=True -> dohvata se jedan zapis (n se ignoriše).
    is_key=False -> n zapisa koji zadovoljavaju uslov (mogu biti u različitim blokovima!)."""
    if is_key:
        return h_i + 1
    return h_i + n


def hash_index_cost(n: int, fudge_factor: float = 1.2) -> float:
    """Pristup preko HASH indeksa. Konstanta 1.2 je "fudge factor" koji
    profesor koristi za dolazak do ulaza u indeks (vidi primer na slajdu 34,
    BP2 - 5, i pominjanje istog faktora kod hash-joina, slajd 26 BP2 - 6)."""
    return fudge_factor + n


# NAPOMENA O KONVENCIJI ZA h_i:
# Na primeru sa slajda 34 (BP2 - 5. Obrada upita), profesor za indeks
# dubine (treeHeight) = 2 koristi h_i = 1 u formulama, uz obrazloženje
# "pretpostavka da su unutrašnji čvorovi u memoriji, onda je h=1".
# Znači: pri pozivu ovih funkcija iz operations.py/optimizer.py,
# prosleđuj h_i = index.tree_height - 1 (nikad sirov treeHeight iz JSON-a).
def effective_tree_height(tree_height: int) -> int:
    """Pretvara 'sirovi' treeHeight iz šeme u h_i koji ide u formule cene,
    po konvenciji da su unutrašnji čvorovi B+ stabla već u memoriji."""
    return max(tree_height - 1, 0)


# =====================================================================
# SLOŽENE SELEKCIJE (BP2 - 5, slajd 8-10) - napomena, ne formula cene
# =====================================================================
# A7 (konjukcija preko indeksa): odabrati uslov/algoritam najmanje cene,
#     ostale uslove testirati nad dohvaćenim zapisima (cena testiranja = 0,
#     pipelining).
# A8 (konjukcija preko kompozitnog indeksa): koristiti isti
#     clustering/secondary_index_cost ali sa n/b izračunatim za spojeni uslov.
# Ove odluke idu u operations.py/optimizer.py - cost_formulas.py samo
# obezbeđuje osnovne "cene pristupa" od kojih se te odluke prave.


# =====================================================================
# SORTIRANJE - spoljno sort-merge sortiranje (BP2 - 5, slajd 12-16)
# =====================================================================

def external_sort_cost(b_r: int, M: int, b_b: int = 1) -> float:
    """Cena spoljnog sort-merge sortiranja u broju blok transfera.
    Ako relacija staje u M blokova memorije, sortira se u jednom prolazu
    (quicksort u memoriji) i NE piše se na disk (izlaz ide dalje kroz
    pipeline) - cena je samo br (čitanje), ne 2*br.
    """
    if b_r <= M:
        return b_r

    merge_factor = math.floor(M / b_b) - 1
    if merge_factor < 1:
        raise ValueError("M/b_b mora biti > 1 da bi merge bio moguć")

    num_passes = math.ceil(math.log(b_r / M, merge_factor))
    # finalni upis se ne računa (rezultat ide dalje kroz pipeline)
    return b_r * (2 * num_passes + 1)


def external_sort_seeks(b_r: int, M: int, b_b: int = 1) -> float:
    """Broj pretraga (seeks) za spoljno sortiranje. Slajd 16, BP2 - 5."""
    if b_r <= M:
        return 2  # 1 seek za čitanje + 1 za upis (ako je potreban upis)

    merge_factor = math.floor(M / b_b) - 1
    num_passes = math.ceil(math.log(b_r / M, merge_factor))
    return 2 * math.ceil(b_r / M) + math.ceil(b_r / b_b) * (2 * num_passes - 1)


# =====================================================================
# SPAJANJE (BP2 - 6, slajd 17-27)
# =====================================================================

def nested_loop_cost(n_r: int, b_s: int, b_r: int) -> float:
    """Spajanje ugnježdenom petljom, najgori slučaj (1 blok memorije).
    r = spoljna relacija, s = unutrašnja. Slajd 18."""
    return n_r * b_s + b_r


def block_nested_loop_cost(b_r: int, b_s: int, M: int) -> float:
    """Spajanje blok ugnježdenom petljom, sa M-2 bafer blokova za spoljnu
    relaciju. Slajd 20."""
    return math.ceil(b_r / (M - 2)) * b_s + b_r


def indexed_nested_loop_cost(b_r: int, cost_per_lookup: float, n_r: int) -> float:
    """Spajanje indeksiranom ugnježdenom petljom. Slajd 21.
    cost_per_lookup = cena JEDNE selekcije nad unutrašnjom relacijom preko
    indeksa (tipično clustering_index_*_cost ili secondary_index_cost/
    hash_index_cost, zavisno od tipa indeksa nad atributom spajanja)."""
    return b_r + n_r * cost_per_lookup


def merge_join_cost(b_r: int, b_s: int) -> float:
    """Objedinjeno spajanje (merge-join), pod pretpostavkom da su obe
    relacije već sortirane po atributu spajanja (ili se sortiranje računa
    posebno preko external_sort_cost i sabira). Slajd 23."""
    return b_r + b_s


def hash_join_cost(b_r: int, b_s: int, M: int) -> float:
    """Heš spajanje (hash-join). Slajd 24 i 26, BP2 - 6.

    - Ako build relacija (manja, s) staje kompletno u memoriju: br + bs,
      particionisanje nije potrebno.
    - Ako ne staje, ali M > sqrt(b_s): rekurzivno particionisanje NIJE
      potrebno -> cena 3*(br+bs) (4*n_h se po konvenciji sa primera
      profesora zanemaruje, vidi slajd 26: "zanemareno 4nh").
    - Inače (M <= sqrt(b_s)): potrebno je rekurzivno particionisanje.
    """
    if b_s <= M:
        return b_r + b_s

    if M > math.sqrt(b_s):
        return 3 * (b_r + b_s)

    n_h = math.ceil(math.log(b_s / M, M - 1))
    return 2 * (b_r + b_s) * n_h + b_r + b_s


def hash_join_seeks(b_r: int, b_s: int, M: int, b_b: int = 1) -> float:
    """Broj pretraga za hash-join, prati istu granu logike kao cena. Slajd 26."""
    if b_s <= M:
        return 2

    base = 2 * (math.ceil(b_r / b_b) + math.ceil(b_s / b_b))
    if M > math.sqrt(b_s):
        return base  # + 2*n_h se po istoj konvenciji zanemaruje

    n_h = math.ceil(math.log(b_s / M, M - 1))
    return base * n_h


# =====================================================================
# PROJEKCIJA (BP2 - 6, slajd 28)
# =====================================================================

def projection_cost(b_r: int, needs_dedup: bool, M: int, b_b: int = 1) -> float:
    """Cena projekcije. Ako SELECT lista sadrži sve atribute nekog
    kandidat-ključa, duplikati su nemogući (needs_dedup=False) i cena
    eliminacije duplikata je 0 (samo skeniranje, pipelining).
    Ako needs_dedup=True, eliminacija duplikata se svodi na sortiranje."""
    if not needs_dedup:
        return 0  # pipelining - projekcija sama po sebi ne dodaje cenu
    return external_sort_cost(b_r, M, b_b)


# =====================================================================
# PROCENA VELIČINE / SELEKTIVNOSTI (BP2 - 6, slajd 25-27)
# Neophodno da bi optimizator mogao da poredi planove za upite sa
# više od jedne selekcije/spajanja (procena međurezultata).
# =====================================================================

def selectivity_equality(n_r: int, v_a_r: int) -> float:
    """Selekcija po jednakosti: n_r / V(A,r). Slajd 25."""
    if v_a_r == 0:
        return 0.0
    return n_r / v_a_r


def size_equality_on_key(is_unique: bool) -> Optional[int]:
    """Ako je uslov jednakosti po ključnom (unique) atributu, procena
    veličine je tačno 1."""
    return 1 if is_unique else None


def size_comparison(n_r: int, v: float, min_a: float, max_a: float) -> float:
    """Selekcija po uslovu nejednakosti (A <= v ili A >= v, simetrično).
    Slajd 25. Ako min/max nisu poznati, koristiti c = n_r / 2 (fallback)."""
    if v < min_a:
        return 0.0
    if max_a == min_a:
        return n_r  # degenerativan slučaj
    return n_r * (v - min_a) / (max_a - min_a)


def size_conjunction(n_r: int, selectivities: list[float]) -> float:
    """Konjukcija uslova (WHERE ... AND ... AND ...). Slajd 26.
    size = n_r * s1 * s2 * ... * sn  (svaka selektivnost već je s_i = s_i/n_r
    normalizovana, tj. prosleđuje se lista selektivnosti [0,1], ne s_i)."""
    result = n_r
    for s in selectivities:
        result *= s
    return result


def size_disjunction(n_r: int, selectivities: list[float]) -> float:
    product = 1.0
    for s in selectivities:
        product *= (1 - s)
    return n_r * (1 - product)


def size_negation(n_r: int, size_theta: float) -> float:
    return n_r - size_theta


def join_size_key_reference(n_r: int, n_s: int, r_has_key: bool) -> float:
    return n_s if r_has_key else n_r


def join_size_foreign_key(n_ref_side: int) -> float:
    """Ako je zajednički atribut strani ključ koji referiše drugu relaciju,
    broj redova u spajanju je tačno jednak broju redova na strani stranog
    ključa. Slajd 27."""
    return n_ref_side


def join_size_general(n_r: int, n_s: int, v_a_s: int, v_a_r: int) -> float:
    est_via_s = (n_r * n_s) / v_a_s if v_a_s else float("inf")
    est_via_r = (n_r * n_s) / v_a_r if v_a_r else float("inf")
    return min(est_via_s, est_via_r)


def distinct_values_after_selection(v_a_r: int, n_selected: int) -> int:
    return min(v_a_r, n_selected)


def distinct_values_after_join(v_a_r: int, n_join: float) -> float:
    return min(v_a_r, n_join)