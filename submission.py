"""
Kaggriculture submission -- control-theoretic / market-microstructure agent.

Architecture: rolling-horizon MPC.
  * Ground-truth engine constants (CROPS/ANIMALS/MARKET_PARAMS/market_price)
    imported directly from kaggle_environments so the internal economics
    model can never drift from the real transition function, with an exact
    hardcoded fallback if the import path ever changes.
  * Once per day (hour==0): investment decisions (HIRE / BUY_LAND /
    BUY_ANIMAL / BUILD_COOP / BUILD_PASTURE), a proportional-fair crop-mix
    quota weighted by live price and market-depth fragility, and a
    boustrophedon-sweep route partitioned into per-unit contiguous chunks
    (tier-sorted: protect sunk capital > realize value > grow > polish).
  * Every turn: live tile-state re-evaluation at the unit's current stop
    (robust to mid-day decay), plus a marginal-price sell-throttle using
    the real price function against the live observed market inventory.
  * Phase controller: ACCEL -> THROUGHPUT -> LIQUIDATE, transitioning off
    of remaining turns rather than a hardcoded script.
"""
import math

# ============================================================================
# ECONOMICS: engine constants + derived crop/animal/market statistics
# ============================================================================
try:
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        CROPS, ANIMALS, MARKET_PARAMS, SHOPS, PRODUCTS, TOWN_CENTER_PRODUCTS,
        LAND_ORDER, LAND_PRICES, FARM_HAND_COST_MULT, MAX_SHOP_INSTANCES,
        market_price as _engine_market_price,
    )
except Exception:
    CROPS = {
        "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
        "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
        "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
        "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
        "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
    }
    ANIMALS = {
        "GOOSE": {"cost": 300, "structure": "COOP", "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
        "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
        "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
    }
    PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
    MARKET_I0 = 10000
    MARKET_PARAMS = {
        "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
        "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
        "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
        "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
        "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
        "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
        "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
        "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
        "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
    }
    SHOPS = {
        "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
        "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
        "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
    }
    TOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]
    LAND_ORDER = ["NE", "SW", "SE"]
    LAND_PRICES = [1000, 2000, 4000]
    FARM_HAND_COST_MULT = 1
    MAX_SHOP_INSTANCES = 8
    HINGE_GAIN = 8.0

    def _shape(func, x, T=None):
        import math
        x = max(0.0, x)
        if func == "linear": return x
        if func == "sq": return x * x
        if func == "sqrt": return math.sqrt(x)
        if func == "log": return math.log(1.0 + x)
        if func == "log10": return math.log10(1.0 + x)
        if func == "hinge":
            if not T or T <= 0:
                return x
            u = x / T
            return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
        return x

    def _engine_market_price(item, inventory, params=None):
        p = (params or MARKET_PARAMS)[item]
        base, I0, T = p["base"], p["I0"], p["T"]
        if inventory < I0:
            f = p["below_func"]
            amp = p["below_target"] * base / _shape(f, T, T)
            price = base + amp * _shape(f, I0 - inventory, T)
        else:
            f = p["above_func"]
            amp = p["above_target"] * base / _shape(f, T, T)
            price = base - amp * _shape(f, inventory - I0, T)
        return max(1, int(round(price)))


def market_price(item, inventory, params=None):
    return _engine_market_price(item, inventory, params)


def fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(n_already_today, mult=FARM_HAND_COST_MULT):
    return mult * fib(n_already_today)


I0_DEFAULT = MARKET_PARAMS["WHEAT"]["I0"]


def find_floor_x(item, params=None):
    """Oversupply x above I0 at which price(I0+x) hits the $1 floor. None if it never floors in range."""
    p = (params or MARKET_PARAMS)[item]
    I0 = p["I0"]
    lo, hi = 0, 5_000_000
    if market_price(item, I0 + hi, params) > 1:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if market_price(item, I0 + mid, params) <= 1:
            hi = mid
        else:
            lo = mid
    return hi


# Precomputed once at import: market depth (oversupply units before price floors).
# None => effectively bottomless (wheat, egg -> log decay never reaches $1).
MARKET_DEPTH = {item: find_floor_x(item) for item in PRODUCTS}
_DEPTH_FOR_SORT = {k: (v if v is not None else 10 ** 9) for k, v in MARKET_DEPTH.items()}

# Rough $/tile/day at BASE price, ignoring market impact -- used only as a
# starting prior before live prices are known (turn 0).
def _one_time_crop_stats(name, d):
    cycle_days = d["max_yield_day"] + 1
    window_start = (d["max_yield_day"] + 1) // 2
    bonus_days = d["max_yield_day"] - window_start + 1
    units = min(d["max_yield"], 1 + bonus_days)
    return cycle_days, units


def _ongoing_crop_stats(name, d):
    cycle_days = d["first_yield_day"] + (d["max_yield"] - 1) * max(d["interval"], 1) + 1
    units = d["max_yield"]
    return cycle_days, units


CROP_STATS = {}
for _name, _d in CROPS.items():
    if _d["ongoing"]:
        _cd, _u = _ongoing_crop_stats(_name, _d)
    else:
        _cd, _u = _one_time_crop_stats(_name, _d)
    CROP_STATS[_name] = {"cycle_days": _cd, "units_per_cycle": _u,
                          "base_rev_per_day": _u * MARKET_PARAMS[_name]["base"] / _cd}

ANIMAL_STATS = {}
for _name, _d in ANIMALS.items():
    _rate = 1.0 / _d["interval"]
    ANIMAL_STATS[_name] = {"steady_rate": _rate,
                            "base_rev_per_day": _rate * MARKET_PARAMS[_d["product"]]["base"]}


# ============================================================================
# CORE: state parsing, zoning, tile-decision logic, day-route planner
# ============================================================================

import math

TURNS_PER_DAY_DEFAULT = 24
BOARD_SIZE_DEFAULT = 10
EPISODE_STEPS_DEFAULT = 720
SHED_CAP_DEFAULT = 100

MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

# Fragility ranking (smaller MARKET_DEPTH = crashes faster = grow less of it).
# None (wheat, egg) treated as very large (bottomless).
def _depth_val(item):
    d = MARKET_DEPTH.get(item)
    return d if d is not None else 10 ** 9


CROPS_LIST = list(CROPS)
ANIMALS_LIST = list(ANIMALS)

# ---------------------------------------------------------------------------
# Global persistent cache (survives across turns within one episode process;
# every value is also cheaply re-derivable so a cold cache is not fatal).
# ---------------------------------------------------------------------------
_STATE = {
    "planned_day": -1,
    "routes": {},          # unit_idx -> list of stops (dict)
    "route_ptr": {},       # unit_idx -> int index into routes[unit_idx]
    "crop_zone": {},       # (x,y) -> crop name, for empty tiles this day
    "animal_zone": set(),  # set of (x,y) reserved for coop/pasture
    "opp_prev_money": None,
    "our_prev_money": None,
    "phase": "ACCEL",
}


def _get(d, key, default):
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def reset_state():
    _STATE["planned_day"] = -1
    _STATE["routes"] = {}
    _STATE["route_ptr"] = {}
    _STATE["crop_zone"] = {}
    _STATE["animal_zone"] = set()


# ---------------------------------------------------------------------------
# Farm parsing helpers
# ---------------------------------------------------------------------------

def shed_access_tiles(board_size):
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def quadrant_of(x, y, board_size):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def owned_tiles(farm, board_size):
    """List of (x, y) tiles in unlocked quadrants."""
    unlocked = set(farm.get("unlocked_quadrants", ["NW"]))
    out = []
    for y in range(board_size):
        for x in range(board_size):
            if quadrant_of(x, y, board_size) in unlocked:
                out.append((x, y))
    return out


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Economics: live marginal value estimates using observed market state
# ---------------------------------------------------------------------------

def crop_value_per_tile_day(crop, prices, fertilizer_free=False):
    stats = CROP_STATS[crop]
    p = prices.get(crop, MARKET_PARAMS[crop]["base"])
    return stats["units_per_cycle"] * p / stats["cycle_days"]


def animal_value_per_tile_day(animal, prices):
    stats = ANIMAL_STATS[animal]
    product = ANIMALS[animal]["product"]
    p = prices.get(product, MARKET_PARAMS[product]["base"])
    feed_cost = prices.get("WHEAT", MARKET_PARAMS["WHEAT"]["base"]) * 0  # wheat usually self-supplied; ignore cash cost here
    return stats["steady_rate"] * p - feed_cost


def crop_priority_score(crop, prices, remaining_days):
    """$/tile/day at current price, discounted by market fragility and
    whether the crop can even mature before season end."""
    stats = CROP_STATS[crop]
    cd = CROPS[crop]
    lead = cd["max_yield_day"] if not cd["ongoing"] else cd["first_yield_day"]
    if remaining_days < lead + 1:
        return -1.0
    val = crop_value_per_tile_day(crop, prices)
    # Fragility discount: thinner market depth -> temper enthusiasm so we
    # don't over-zone into a crop that will crash the moment we scale it.
    depth = _depth_val(crop)
    frag_factor = min(1.0, 0.15 + depth / 900.0)
    return val * frag_factor


def animal_priority_score(animal, prices, remaining_days):
    ad = ANIMALS[animal]
    if remaining_days < ad["first_yield_day"] + 2:
        return -1.0
    val = animal_value_per_tile_day(animal, prices)
    depth = _depth_val(ad["product"])
    frag_factor = min(1.0, 0.15 + depth / 900.0)
    return val * frag_factor


# ---------------------------------------------------------------------------
# Marginal-price sell planner (uses the *exact* real price function)
# ---------------------------------------------------------------------------

def plan_sell_qty(item, inventory, available, floor_price, params, cap=None):
    """Greedy: sell units while the marginal (per-unit) price stays >= floor_price."""
    if available <= 0:
        return 0
    k = 0
    limit = available if cap is None else min(available, cap)
    inv = inventory
    while k < limit:
        p = market_price(item, inv + k, params)
        if p < floor_price:
            break
        k += 1
    return k


def sell_floor_fraction(item, phase, remaining_turns):
    """Fraction of base price below which we stop selling this item today.
    Thin-depth (fragile) items get a HIGHER floor (sell less / hold more),
    robust items get a lower floor (sell freely). Relaxes to 0 in LIQUIDATE.
    """
    base = MARKET_PARAMS[item]["base"]
    depth = _depth_val(item)
    # fragile (small depth) -> hold threshold high; robust -> low
    frag = max(0.0, min(1.0, 1.0 - depth / 900.0))
    frac = 0.15 + 0.55 * frag  # 0.15 (robust) .. 0.70 (fragile)
    if phase == "LIQUIDATE":
        # Relax linearly to 0 as we approach turn 720.
        urgency = max(0.0, min(1.0, remaining_turns / 240.0))
        frac *= urgency
    return frac * base


# ---------------------------------------------------------------------------
# Phase controller
# ---------------------------------------------------------------------------

def compute_phase(day, remaining_turns, total_days=30):
    if remaining_turns <= 6 * TURNS_PER_DAY_DEFAULT:
        return "LIQUIDATE"
    if day <= 8:
        return "ACCEL"
    return "THROUGHPUT"


# ---------------------------------------------------------------------------
# Tile task decision (live, at time of arrival)
# ---------------------------------------------------------------------------

def decide_tile_action(tile, day, crop_for_empty, carried, can_fertilize_value_crops, build_for_empty=None):
    """Return farmer op (list) given the live tile dict/None/'LOCKED' and
    what we're carrying, plus the crop we'd plant / structure we'd build if
    it's empty (build_for_empty takes priority when set)."""
    if tile is None:
        if build_for_empty is not None:
            return ["BUILD_COOP" if build_for_empty == "COOP" else "BUILD_PASTURE"]
        if crop_for_empty is not None:
            return ["PLANT", crop_for_empty]
        return None
    if tile == "LOCKED":
        return None
    kind = tile.get("kind")
    if kind == "WEED":
        return ["DIG"]
    if kind == "PLANT":
        if tile.get("yield_units", 0) > 0 and day - tile["planted_day"] >= CROPS[tile["crop"]]["first_yield_day"]:
            return ["HARVEST"]
        if not tile.get("watered_today", False):
            return ["WATER"]
        if (can_fertilize_value_crops and carried.get("FERTILIZER", 0) > 0
                and tile.get("fertilized_until_day", -1) < day):
            cd = CROPS[tile["crop"]]
            age = day - tile["planted_day"]
            window_start = (cd["max_yield_day"] + 1) // 2 if not cd["ongoing"] else 0
            if age >= window_start:
                return ["FERTILIZE"]
        return None
    if "animal" in tile:
        if tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
        if not tile.get("fed_today", False) and carried.get("WHEAT", 0) > 0:
            return ["FEED"]
        if tile.get("fertilizer_available", False):
            return ["COLLECT_FERTILIZER"]
        if not tile.get("cared_today", False):
            return ["CARE"]
        return None
    if kind in ("COOP", "PASTURE") and "animal" not in tile:
        for aname, ad in ANIMALS.items():
            if ad["structure"] == kind and carried.get(aname, 0) > 0:
                return ["PLACE", aname]
        return None
    return None


# ---------------------------------------------------------------------------
# Crop-mix quota (proportional fair allocation, recomputed once per day)
# ---------------------------------------------------------------------------

def compute_crop_fractions(prices, remaining_days):
    scores = {c: max(0.0, crop_priority_score(c, prices, remaining_days)) for c in CROPS_LIST}
    total = sum(scores.values())
    if total <= 0:
        eligible = [c for c in CROPS_LIST if remaining_days >= (CROPS[c]["max_yield_day"] if not CROPS[c]["ongoing"] else CROPS[c]["first_yield_day"]) + 1]
        if not eligible:
            return {c: 0.0 for c in CROPS_LIST}
        frac = 1.0 / len(eligible)
        return {c: (frac if c in eligible else 0.0) for c in CROPS_LIST}
    frac = {c: scores[c] / total for c in CROPS_LIST}
    # Wheat floor: self-sufficiency for animal feed + it is a bottomless-depth
    # staple, always worth a baseline allocation.
    floor = 0.15 if remaining_days >= CROPS["WHEAT"]["max_yield_day"] + 1 else 0.0
    if frac.get("WHEAT", 0) < floor and floor > 0:
        deficit = floor - frac.get("WHEAT", 0)
        others = [c for c in CROPS_LIST if c != "WHEAT" and frac[c] > 0]
        other_total = sum(frac[c] for c in others)
        if other_total > 0:
            for c in others:
                frac[c] -= deficit * (frac[c] / other_total)
        frac["WHEAT"] = floor
    return frac


def pick_crop_to_plant(fractions, counts, total_planted):
    """Largest-remainder proportional allocation, live during route execution."""
    eligible = [c for c in CROPS_LIST if fractions.get(c, 0) > 0]
    if not eligible:
        return None
    n = total_planted + 1
    best_c, best_score = None, None
    for c in eligible:
        target = fractions[c] * n
        deficit = target - counts.get(c, 0)
        if best_score is None or deficit > best_score:
            best_score, best_c = deficit, c
    return best_c


def animal_targets(money, remaining_days, num_owned_tiles, prices):
    """How many of each animal we WANT eventually, given farm scale."""
    if remaining_days < ANIMALS["GOOSE"]["first_yield_day"] + 3:
        return {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    animal_tile_budget = max(0, int(num_owned_tiles * 0.20))
    scores = {a: max(0.0, animal_priority_score(a, prices, remaining_days)) for a in ANIMALS_LIST}
    total = sum(scores.values())
    if total <= 0 or animal_tile_budget <= 0:
        return {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    targets = {}
    for a in ANIMALS_LIST:
        targets[a] = max(0, int(round(animal_tile_budget * scores[a] / total)))
    return targets


# ---------------------------------------------------------------------------
# Boustrophedon (lawnmower) sweep partitioned into per-unit contiguous
# chunks. Far more turn-efficient than a global nearest-pair matcher, which
# tends to send units on long crisscrossing trips and starves tier-1 (must
# water/feed) tasks on tiles that never win a "globally closest" contest.
# ---------------------------------------------------------------------------

def serpentine_tiles(farm, board_size):
    unlocked = set(farm.get("unlocked_quadrants", ["NW"]))
    out = []
    for y in range(board_size):
        xs = range(board_size) if y % 2 == 0 else range(board_size - 1, -1, -1)
        for x in xs:
            if quadrant_of(x, y, board_size) in unlocked:
                out.append((x, y))
    return out


def classify_tile_tier(tile):
    """Tier 1 = protect sunk capital, 2 = realize value, 3 = grow, 4 = polish."""
    if tile is None:
        return 3
    if tile == "LOCKED":
        return None
    kind = tile.get("kind")
    if kind == "PLANT":
        if tile.get("yield_units", 0) > 0:
            return 2
        if not tile.get("watered_today", False):
            return 1
        return 4
    if kind == "WEED":
        return 2  # reclaiming a wasted tile beats greenfield expansion (tier 3)
    if "animal" in tile:
        if tile.get("yield_units", 0) > 0:
            return 2
        if not tile.get("fed_today", False):
            return 1
        return 4
    if kind in ("COOP", "PASTURE"):
        return 3
    return 4


def build_day_routes(farm, board_size, day, turns_per_day, unit_positions, budget_override=None):
    """unit_positions: dict uidx -> (x,y) at day start (or current position,
    for a mid-day replan). Partitions a single boustrophedon sweep of all
    owned tiles into contiguous, roughly-equal chunks (one per unit), then
    within each unit's chunk sorts by urgency tier so that if the chunk is
    too big for the day's budget, only low-priority "polish" tasks get
    dropped -- watering/feeding never starve for units doing their own
    local segment.
    """
    sweep = serpentine_tiles(farm, board_size)
    unit_ids = sorted(unit_positions.keys())
    n = max(1, len(unit_ids))
    chunk_size = math.ceil(len(sweep) / n) if sweep else 0
    routes = {}
    for i, uidx in enumerate(unit_ids):
        chunk = sweep[i * chunk_size:(i + 1) * chunk_size]
        tiered = []
        for (x, y) in chunk:
            tile = farm["tiles"][y][x]
            tier = classify_tile_tier(tile)
            if tier is not None:
                tiered.append((tier, (x, y)))
        tiered.sort(key=lambda p: p[0])
        budget = turns_per_day if budget_override is None else budget_override
        pos = unit_positions[uidx]
        route = []
        for tier, t in tiered:
            d = manhattan(pos, t) + 1
            if d > budget:
                continue
            budget -= d
            pos = t
            route.append(t)
        routes[uidx] = route
    return routes


# ============================================================================
# TURN CONTROLLER: investment/seed/sell decisions + agent() entrypoint
# ============================================================================


MAX_ORDERS = 10


def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _plan_investments(farm, opp_farm, private, day, remaining_turns, board_size, phase, prices):
    """Called once at hour==0. Returns list of (priority, market_order)."""
    orders = []
    money = farm["money"]
    tiles = owned_tiles(farm, board_size)
    n_owned = len(tiles)
    n_hands_existing = 0  # hands reset to [] each day; hires_today reset to 0

    # --- Land purchase --------------------------------------------------
    n_extra = len(farm["unlocked_quadrants"]) - 1
    n_owned_planned = n_owned
    if phase != "LIQUIDATE" and n_extra < len(LAND_ORDER):
        cost = LAND_PRICES[n_extra]
        # Payback check: remaining days * rough achievable $/tile/day (using a
        # conservative blended value) must clear the cost with margin.
        blended_value_per_tile_day = 12.0
        payback_days = cost / (25 * blended_value_per_tile_day)
        remaining_days = remaining_turns / TURNS_PER_DAY_DEFAULT
        if money - cost >= 50 and remaining_days > payback_days * 1.5:
            orders.append((900, ["BUY_LAND"]))
            money -= cost  # reserve
            n_owned_planned += 25  # plan hiring/seed capacity for the bigger farm now

    # --- Hiring -----------------------------------------------------------
    capacity_per_unit = 10  # empirical: ~2 turns/tile incl. movement + action
    target_units = max(1, math.ceil(n_owned_planned / capacity_per_unit))
    target_hands = max(0, target_units - 1)
    cum_cost = 0
    for n in range(target_hands):
        c = hire_cost(n)
        cum_cost += c
        if money - cum_cost < 0:
            break
        orders.append((800 - n, ["HIRE"]))

    # --- Animals ------------------------------------------------------
    remaining_days = remaining_turns / TURNS_PER_DAY_DEFAULT
    targets = animal_targets(money, remaining_days, n_owned, prices)
    owned_animals = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    unplaced = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and "animal" in t and t.get("animal"):
                owned_animals[t["animal"]] += 1
    shed = private.get("shed", {})
    for a in unplaced:
        unplaced[a] = shed.get(a, 0)
    if phase != "LIQUIDATE":
        # buy at most 1 animal per day to keep cashflow smooth
        best_a, best_deficit = None, 0
        for a, tgt in targets.items():
            deficit = tgt - owned_animals[a] - unplaced[a]
            if deficit > best_deficit:
                best_deficit, best_a = deficit, a
        if best_a is not None:
            cost = ANIMALS[best_a]["cost"]
            if money - cost >= 200:
                orders.append((500, ["BUY_ANIMAL", best_a, 1]))
                money -= cost

    # --- Feed safety net: top up wheat if animals would otherwise starve ---
    n_animals_alive = sum(
        1 for row in farm["tiles"] for t in row
        if isinstance(t, dict) and "animal" in t and t.get("animal")
    )
    wheat_have = private.get("shed", {}).get("WHEAT", 0)
    deficit = n_animals_alive - wheat_have
    if deficit > 0:
        wheat_price = prices.get("WHEAT", MARKET_PARAMS["WHEAT"]["base"])
        afford = int(money // wheat_price) if wheat_price > 0 else 0
        buy_n = min(deficit, afford)
        if buy_n > 0:
            orders.append((950, ["BUY_PRODUCT", "WHEAT", buy_n]))

    return orders, target_units


def _sell_orders(private, prices, market_inv, phase, remaining_turns):
    orders = []
    shed = private.get("shed", {})
    for item in PRODUCTS:
        qty = shed.get(item, 0)
        if qty <= 0:
            continue
        floor = sell_floor_fraction(item, phase, remaining_turns)
        params = None
        k = plan_sell_qty(item, market_inv.get(item, MARKET_PARAMS[item]["I0"]), qty, floor, params)
        if k > 0:
            value = k * prices.get(item, MARKET_PARAMS[item]["base"])
            orders.append((600 + min(200, value / 10), ["SELL", item, k]))
    return orders


def _seed_orders(farm, private, day, remaining_turns, prices, board_size, target_units=None):
    orders = []
    money = farm["money"]
    tiles = owned_tiles(farm, board_size)
    n_empty = sum(1 for (x, y) in tiles if farm["tiles"][y][x] is None)
    if n_empty <= 0:
        return orders
    remaining_days = remaining_turns / TURNS_PER_DAY_DEFAULT
    fractions = compute_crop_fractions(prices, remaining_days)
    # NB: farm["hands"] is always [] at hour==0 (reset daily, hired hands
    # spawn with a 1-turn lag) -- use the day's *planned* unit count, not
    # the stale live count, or this chronically under-buys seed.
    n_units = target_units if target_units is not None else (1 + len(farm.get("hands", [])))
    labor_cap = n_units * 10
    plan_count = min(n_empty, labor_cap)
    seeds = private.get("seeds", {})
    for c, frac in fractions.items():
        if frac <= 0:
            continue
        need = int(math.ceil(frac * plan_count)) - seeds.get(c, 0)
        if need <= 0:
            continue
        cost_each = CROPS[c]["seed"]
        afford = int(money // cost_each) if cost_each > 0 else 0
        buy_n = max(0, min(need, afford))
        if buy_n > 0:
            orders.append((300, ["BUY_SEED", c, buy_n]))
            money -= buy_n * cost_each
    return orders


def compute_struct_deficit(farm, private, targets):
    owned_animals = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    unfilled = {"COOP": 0, "PASTURE": 0}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict):
                if "animal" in t and t.get("animal"):
                    owned_animals[t["animal"]] += 1
                elif t.get("kind") in ("COOP", "PASTURE") and "animal" not in t:
                    unfilled[t["kind"]] += 1
    shed = private.get("shed", {})
    unplaced = {a: shed.get(a, 0) for a in ANIMALS}
    goose_need = max(0, targets.get("GOOSE", 0) - owned_animals["GOOSE"] - unplaced["GOOSE"] - unfilled["COOP"])
    pasture_have = owned_animals["COW"] + owned_animals["SHEEP"] + unplaced["COW"] + unplaced["SHEEP"] + unfilled["PASTURE"]
    pasture_target = targets.get("COW", 0) + targets.get("SHEEP", 0)
    pasture_need = max(0, pasture_target - pasture_have)
    return {"COOP": goose_need, "PASTURE": pasture_need}


def _live_action_for_unit(farm, private, uidx, board_size, day, crop_fracs):
    """Given cached route, produce this unit's action for THIS turn."""
    pos = farm["farmer"] if uidx == 0 else farm["hands"][uidx - 1]
    fx, fy = pos
    route = _STATE["routes"].get(uidx, [])
    ptr = _STATE["route_ptr"].get(uidx, 0)
    inv = private["inventories"][uidx] if uidx < len(private.get("inventories", [])) else {}

    while ptr < len(route):
        tx, ty = route[ptr]
        if (fx, fy) != (tx, ty):
            step = _step_toward(fx, fy, tx, ty)
            return step if step else "PASS"
        tile = farm["tiles"][ty][tx]
        crop_choice = None
        build_choice = None
        if tile is None:
            deficit = _STATE.setdefault("struct_deficit", {"COOP": 0, "PASTURE": 0})
            if deficit.get("COOP", 0) > 0:
                build_choice = "COOP"
            elif deficit.get("PASTURE", 0) > 0:
                build_choice = "PASTURE"
            else:
                crop_choice = pick_crop_to_plant(
                    crop_fracs, _STATE.setdefault("crop_counts", {}), _STATE.get("total_planted", 0)
                )
        act = decide_tile_action(tile, day, crop_choice, inv, can_fertilize_value_crops=True, build_for_empty=build_choice)
        if act is None:
            ptr += 1
            _STATE["route_ptr"][uidx] = ptr
            continue
        if act[0] == "PLANT" and crop_choice is not None:
            _STATE["crop_counts"][crop_choice] = _STATE["crop_counts"].get(crop_choice, 0) + 1
            _STATE["total_planted"] = _STATE.get("total_planted", 0) + 1
        elif act[0] in ("BUILD_COOP", "BUILD_PASTURE") and build_choice is not None:
            _STATE["struct_deficit"][build_choice] -= 1
        # Don't advance ptr: same tile may need another action next turn
        # (e.g. harvest this turn, plant next turn once empty).
        return act
    return "PASS"


def _pickup_needs(farm, private, board_size):
    """Compute (item -> qty) the farmer should pick up at the shed this
    morning: wheat for feeding live animals, fertilizer for top crops."""
    needs = {}
    n_animals_alive = 0
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and "animal" in t and t.get("animal"):
                n_animals_alive += 1
    shed = private.get("shed", {})
    if n_animals_alive > 0 and shed.get("WHEAT", 0) > 0:
        needs["WHEAT"] = min(n_animals_alive, shed["WHEAT"])
    if shed.get("FERTILIZER", 0) > 0:
        needs["FERTILIZER"] = min(6, shed["FERTILIZER"])
    return needs


def _live_animal_tiles(farm, board_size):
    out = []
    for (x, y) in owned_tiles(farm, board_size):
        t = farm["tiles"][y][x]
        if isinstance(t, dict) and "animal" in t and t.get("animal"):
            out.append((x, y))
    return out


def _order_nearest(start, coords):
    remaining = list(coords)
    pos = start
    ordered = []
    while remaining:
        remaining.sort(key=lambda c: manhattan(pos, c))
        nxt = remaining.pop(0)
        ordered.append(nxt)
        pos = nxt
    return ordered


def _plan_animal_quest(farm, private, board_size):
    """Find a single coordinated pickup->walk->place (or walk->build) task
    for the farmer, so a purchased animal never just rots in the shed.
    Returns (target_tile_or_None, animal_to_carry_or_None)."""
    shed = private.get("shed", {})
    tiles = owned_tiles(farm, board_size)
    for a in ANIMALS:
        if shed.get(a, 0) > 0:
            struct_kind = ANIMALS[a]["structure"]
            for (x, y) in tiles:
                t = farm["tiles"][y][x]
                if isinstance(t, dict) and t.get("kind") == struct_kind and "animal" not in t:
                    return (x, y), a
    deficit = _STATE.get("struct_deficit", {})
    if sum(deficit.values()) > 0:
        fx, fy = farm["farmer"]
        best = None
        for (x, y) in tiles:
            if farm["tiles"][y][x] is None:
                d = manhattan((fx, fy), (x, y))
                if best is None or d < best[0]:
                    best = (d, (x, y))
        if best:
            return best[1], None
    return None, None


def agent(obs):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    opp = farms[1 - player] if len(farms) > 1 else farm
    board_size = len(farm["tiles"])
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    step = obs.get("step", day * TURNS_PER_DAY_DEFAULT + hour)
    remaining_turns = max(1, EPISODE_STEPS_DEFAULT - step)
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    inv = market.get("inventory", {}) or {}

    phase = compute_phase(day, remaining_turns)
    _STATE["phase"] = phase

    replanned_full = False
    if _STATE["planned_day"] != day:
        # New day: reset counters, plan farmer-only route for hour 0.
        _STATE["planned_day"] = day
        _STATE["crop_counts"] = {}
        _STATE["total_planted"] = 0
        remaining_days0 = remaining_turns / TURNS_PER_DAY_DEFAULT
        targets0 = animal_targets(farm["money"], remaining_days0, len(owned_tiles(farm, board_size)), prices)
        _STATE["struct_deficit"] = compute_struct_deficit(farm, private, targets0)
        quest_tile, quest_carry = _plan_animal_quest(farm, private, board_size)
        _STATE["quest_tile"] = quest_tile
        _STATE["quest_carry"] = quest_carry
        _STATE["quest_picked_up"] = quest_carry is None
        fx, fy = farm["farmer"]
        animal_tiles = _live_animal_tiles(farm, board_size)
        _STATE["feed_stops"] = _order_nearest((fx, fy), animal_tiles)
        _STATE["need_wheat_pickup"] = len(animal_tiles) > 0
        _STATE["routes"] = build_day_routes(farm, board_size, day, TURNS_PER_DAY_DEFAULT, {0: (fx, fy)})
        priority = ([quest_tile] if quest_tile is not None else []) + _STATE["feed_stops"]
        if priority:
            _STATE["routes"][0] = priority + [t for t in _STATE["routes"][0] if t not in priority]
        _STATE["route_ptr"] = {0: 0}
        _STATE["pending_full_replan"] = True

    if _STATE.get("pending_full_replan") and len(farm.get("hands", [])) > 0:
        # Hands have now spawned (1-turn hire lag) -- rebuild the full route
        # including them, from current live positions, with remaining budget.
        positions = {0: tuple(farm["farmer"])}
        for i, hp in enumerate(farm["hands"]):
            positions[i + 1] = tuple(hp)
        budget_left = max(1, TURNS_PER_DAY_DEFAULT - hour)
        _STATE["routes"] = build_day_routes(farm, board_size, day, TURNS_PER_DAY_DEFAULT, positions, budget_override=budget_left)
        qt = _STATE.get("quest_tile")
        priority = ([qt] if qt is not None else []) + _STATE.get("feed_stops", [])
        if priority:
            _STATE["routes"][0] = priority + [t for t in _STATE["routes"][0] if t not in priority]
        _STATE["route_ptr"] = {u: 0 for u in positions}
        _STATE["pending_full_replan"] = False

    remaining_days = remaining_turns / TURNS_PER_DAY_DEFAULT
    crop_fracs = compute_crop_fractions(prices, remaining_days)

    market_orders = []
    if hour == 0:
        inv_orders, target_units = _plan_investments(farm, opp, private, day, remaining_turns, board_size, phase, prices)
        market_orders.extend(inv_orders)
        _STATE["target_units"] = target_units
        market_orders.extend(_seed_orders(farm, private, day, remaining_turns, prices, board_size, target_units=target_units))
    elif hour <= 4:
        # Hire orders (up to ~9 of the 10 slots) can crowd seed purchases
        # out of hour 0 entirely; retry any still-unmet seed need over the
        # next few hours once hiring is done for the day.
        market_orders.extend(_seed_orders(farm, private, day, remaining_turns, prices, board_size,
                                           target_units=_STATE.get("target_units")))
    market_orders.extend(_sell_orders(private, prices, inv, phase, remaining_turns))

    market_orders.sort(key=lambda o: -o[0])
    market_actions = [o[1] for o in market_orders[:MAX_ORDERS]]

    n_units = 1 + len(farm.get("hands", []))
    farmer_action = _live_action_for_unit(farm, private, 0, board_size, day, crop_fracs)
    hands_actions = []
    if farm.get("hands"):
        needs = _pickup_needs(farm, private, board_size) if hour == 0 else {}
        for i in range(len(farm["hands"])):
            act = _live_action_for_unit(farm, private, i + 1, board_size, day, crop_fracs)
            hands_actions.append(act)

    # Route farmer through pickups at day start if it's sitting on the shed
    # and there is something useful to grab (cheap: farmer always starts
    # the day on a shed-access tile).
    if hour == 0:
        carried = private["inventories"][0] if private.get("inventories") else {}
        quest_carry = _STATE.get("quest_carry")
        if quest_carry is not None and not _STATE.get("quest_picked_up") and carried.get(quest_carry, 0) == 0:
            qty = private.get("shed", {}).get(quest_carry, 0)
            if qty > 0:
                farmer_action = ["PICKUP", quest_carry, qty]
                _STATE["quest_picked_up"] = True
        elif _STATE.get("need_wheat_pickup") and carried.get("WHEAT", 0) == 0:
            qty = min(len(_STATE.get("feed_stops", [])), private.get("shed", {}).get("WHEAT", 0))
            if qty > 0:
                farmer_action = ["PICKUP", "WHEAT", qty]
                _STATE["need_wheat_pickup"] = False
        else:
            needs = _pickup_needs(farm, private, board_size)
            for item, qty in needs.items():
                if carried.get(item, 0) == 0 and qty > 0:
                    farmer_action = ["PICKUP", item, qty]
                    break

    return {
        "farmer": farmer_action if isinstance(farmer_action, list) else [farmer_action],
        "hands": [a if isinstance(a, list) else [a] for a in hands_actions],
        "market": market_actions,
    }
