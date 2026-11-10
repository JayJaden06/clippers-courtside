"""Hand-drawn SVG graphics for the dashboard, in the style of the clippers-home-court project.

Each function returns an HTML string (a ``<div class="gfx">`` wrapping an inline SVG) that the
dashboard renders with ``st.markdown(..., unsafe_allow_html=True)``. They are pictures of a
number, not charts: a court split into halves, The Wall behind one basket, players drawn to
scale, and a 10x10 grid of missed shots. Every number comes in as an argument, so the graphics
update with the data, and every picture has an ``aria-label`` that says what it shows.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class Ink:
    """Colors for the dark theme (the same steps the charts use)."""

    ink: str = "#f0efec"
    muted: str = "#a3a29a"
    floor: str = "#2b2620"
    courtline: str = "#7a6a52"
    paint: str = "#2a3a55"
    good: str = "#3987e5"  # blue = better
    bad: str = "#d95926"  # orange = worse (the same orange as the focus and the second half)
    focus: str = "#d95926"
    gray: str = "#73716b"  # context, 3.8:1 on the page
    empty: str = "#33322f"


DARK = Ink()  # for drawings that sit inside dark cards (the card mini-charts) and the scoreboard
# For drawings on the light page: a maple floor, navy ink. Blue and orange are the same validated pair.
PAGE = Ink(ink="#1b2333", muted="#5f6372", floor="#efe4cf", courtline="#b39a72", paint="#d9e2f2",
           gray="#7f7d77", empty="#e3e0d8")


def _signed(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


def _ft_in(inches: float) -> str:
    return f"{int(inches) // 12}'{int(inches) % 12}\""


def _svg(width: int, height: int, body: str, label: str) -> str:
    return (f'<div class="gfx"><svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}" '
            f'xmlns="http://www.w3.org/2000/svg" font-family="inherit">{body}</svg></div>')


def _text(x: float, y: float, s: str, size: int, fill: str, weight: int = 400, anchor: str = "middle",
          spacing: float = 0) -> str:
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}"{ls}>{escape(s)}</text>')


def _court(c: Ink, left_tint: str | None = None, right_tint: str | None = None, left_paint: str | None = None) -> str:
    """A 760 x 300 court at the origin. Tints color each half; ``left_paint`` colors the left key."""
    lines = f'fill="none" stroke="{c.courtline}" stroke-width="3"'
    out = [f'<rect x="0" y="0" width="760" height="300" rx="6" fill="{c.floor}"/>']
    if left_tint:
        out.append(f'<rect x="0" y="0" width="380" height="300" rx="6" fill="{left_tint}" fill-opacity="0.22"/>')
    if right_tint:
        out.append(f'<rect x="380" y="0" width="380" height="300" rx="6" fill="{right_tint}" fill-opacity="0.22"/>')
    out += [
        f'<rect x="0" y="0" width="760" height="300" rx="6" {lines}/>',
        f'<line x1="380" y1="0" x2="380" y2="300" stroke="{c.courtline}" stroke-width="3"/>',
        f'<circle cx="380" cy="150" r="45" {lines}/>',
        f'<rect x="0" y="95" width="150" height="110" fill="{left_paint or c.floor}" stroke="{c.courtline}" '
        'stroke-width="3"/>',
        f'<rect x="610" y="95" width="150" height="110" fill="{c.floor}" stroke="{c.courtline}" stroke-width="3"/>',
        f'<circle cx="150" cy="150" r="45" {lines}/>',
        f'<circle cx="610" cy="150" r="45" {lines}/>',
        f'<path d="M0 30 L50 30 A140 140 0 0 1 50 270 L0 270" {lines}/>',
        f'<path d="M760 30 L710 30 A140 140 0 0 0 710 270 L760 270" {lines}/>',
        f'<circle cx="22" cy="150" r="9" fill="none" stroke="{c.focus}" stroke-width="4"/>',
        f'<circle cx="738" cy="150" r="9" fill="none" stroke="{c.focus}" stroke-width="4"/>',
    ]
    return "".join(out)


# ------------------------------------------------------------------ the two halves

def court_halves(h1: float, h2: float, c: Ink = PAGE) -> str:
    """The court split at half court: the average score margin in first halves and in second halves."""
    tint = lambda v: c.good if v > 0 else c.bad if v < 0 else None  # noqa: E731
    body = [f'<g transform="translate(20,20)">{_court(c, tint(h1), tint(h2))}</g>']
    for x, label, v in ((210, "1ST HALF", h1), (590, "2ND HALF", h2)):
        color = c.good if v > 0 else c.bad if v < 0 else c.ink
        # a soft panel in the floor color, so the court lines don't run through the numbers
        body += [f'<rect x="{x - 118}" y="84" width="236" height="164" rx="16" fill="{c.floor}" '
                 'fill-opacity="0.9"/>',
                 _text(x, 112, label, 22, c.muted, 700, spacing=3),
                 _text(x, 192, _signed(v), 76, color, 800),
                 _text(x, 232, "points a game", 22, c.muted)]
    body.append(_text(400, 356, "Average Clippers score margin in each half this season", 22, c.muted))
    return _svg(800, 370, "".join(body),
                f"Basketball court split in two: first halves {_signed(h1)} points a game, "
                f"second halves {_signed(h2)}.")


# ------------------------------------------------------------------ The Wall

def wall_court(ft_before: float, ft_after: float, c: Ink = PAGE) -> str:
    """Intuit Dome's court with The Wall behind the left basket and opponents' FT% by half."""
    rows = []
    for i in range(12):  # 12 bands of heads stand in for 51 rows
        y = 52 + i * 26
        rows += [f'<circle cx="{40 + j * 22 + (i % 2) * 11}" cy="{y + 9}" r="7" fill="{c.good}" '
                 f'opacity="{0.45 + 0.045 * (11 - i):.2f}"/>' for j in range(7)]
    body = ["".join(rows),
            _text(105, 34, "THE WALL", 30, c.ink, 800, spacing=2),
            _text(105, 396, "51 rows of fans", 22, c.muted),
            f'<g transform="translate(200,60)">{_court(c, left_paint=c.paint)}</g>',
            f'<circle cx="350" cy="210" r="8" fill="{c.focus}"/>']
    for x, label, v, color in ((470, "BEFORE HALFTIME", ft_before, c.good),
                               (770, "AFTER HALFTIME", ft_after, c.focus)):
        body += [_text(x, 118, label, 18, c.muted, 700, spacing=2),
                 _text(x, 186, f"{v:.0%}", 64, color, 800),
                 _text(x, 220, "free throws made", 19, c.muted)]
    body.append(_text(580, 396, "Opponents at the line, at Intuit Dome", 22, c.muted))
    return _svg(1000, 410, "".join(body),
                f"Intuit Dome court with The Wall behind one basket. Opponents make {ft_before:.0%} of free throws "
                f"before halftime and {ft_after:.0%} after.")


# ------------------------------------------------------------------ players to scale

def _head(cx: float, cy: float, r: float, color: str, photo: str | None) -> str:
    """A round head; with ``photo`` (an NBA headshot URL) the player's face, ringed in the figure's color."""
    if not photo:
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{color}"/>'
    uid = "h" + hashlib.md5(f"{photo}{cx}{cy}".encode()).hexdigest()[:8]
    w = 2 * r * 1.6
    return (f'<clipPath id="{uid}"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"/></clipPath>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#d9d6ce"/>'
            f'<image href="{photo}" x="{cx - w / 2:.1f}" y="{cy - r * 1.05:.1f}" width="{w:.1f}" '
            f'height="{w / 1.368:.1f}" clip-path="url(#{uid})"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{color}" stroke-width="3"/>')


def _person(cx: float, ground: float, height_in: float, weight_lb: float, color: str, px_per_in: float = 3.2,
            photo: str | None = None) -> str:
    """A simple standing figure: height to scale, shoulder width scaled by weight; ``photo`` puts a face on it."""
    h = height_in * px_per_in
    top = ground - h
    head_r = h * 0.07
    shoulders = h * 0.25 * (weight_lb / 200) ** 0.9
    torso_top, torso_bot = top + head_r * 2 + 4, top + h * 0.56
    leg_w = shoulders * 0.36
    body = (f'<rect x="{cx - shoulders / 2:.1f}" y="{torso_top:.1f}" width="{shoulders:.1f}" '
            f'height="{torso_bot - torso_top:.1f}" rx="{shoulders * 0.28:.1f}" fill="{color}"/>'
            f'<rect x="{cx - shoulders * 0.42:.1f}" y="{torso_bot - 2:.1f}" width="{leg_w:.1f}" '
            f'height="{ground - torso_bot + 2:.1f}" rx="{leg_w / 2:.1f}" fill="{color}"/>'
            f'<rect x="{cx + shoulders * 0.42 - leg_w:.1f}" y="{torso_bot - 2:.1f}" width="{leg_w:.1f}" '
            f'height="{ground - torso_bot + 2:.1f}" rx="{leg_w / 2:.1f}" fill="{color}"/>')
    # a face is drawn bigger than the plain head, and last, so it sits on top of the shoulders
    r = head_r * (1.45 if photo else 1)
    return body + _head(cx, top + head_r - (r - head_r), r, color, photo)


def height_matchup(name: str, height_in: float, weight_lb: float, every_tall: float, every_short: float,
                   prior_every_tall: float, prior_every_short: float, tall_in: float = 78,
                   c: Ink = PAGE, photo: str | None = None) -> str:
    """The player next to a bigger and a shorter defender, with how often he turns it over against each."""
    ground = 300
    body = []
    panels = ((210, f"Guarded by someone {_ft_in(tall_in)} or taller", tall_in + 2, every_tall, prior_every_tall,
               c.bad),
              (610, "Guarded by someone shorter", tall_in - 4, every_short, prior_every_short, c.good))
    for cx, title, d_height, every, prior, color in panels:
        body += [_person(cx - 50, ground, height_in, weight_lb, c.focus, photo=photo),
                 _person(cx + 50, ground, d_height, 215, c.gray),
                 f'<line x1="{cx - 130}" y1="{ground}" x2="{cx + 130}" y2="{ground}" stroke="{c.courtline}" '
                 'stroke-width="3"/>',
                 _text(cx - 50, ground + 26, name, 18, c.ink, 700),
                 _text(cx + 50, ground + 26, "Defender", 18, c.muted),
                 _text(cx, 32, title, 21, c.ink, 700),
                 _text(cx, 372, f"1 turnover every {every:.0f} times", 30, color, 800),
                 _text(cx, 402, f"last season: every {prior:.0f} times", 19, c.muted)]
    body.append(f'<line x1="410" y1="20" x2="410" y2="400" stroke="{c.empty}" stroke-width="2"/>')
    return _svg(820, 415, "".join(body),
                f"{name} turns it over once every {every_tall:.0f} times when a defender {_ft_in(tall_in)} or taller "
                f"guards him, and once every {every_short:.0f} times against shorter defenders.")


def size_matchup(name: str, height_in: float, weight_lb: float, opp_height_in: float, opp_weight_lb: float,
                 fg_allowed: float, team_fg: float, c: Ink = PAGE, photo: str | None = None) -> str:
    """The player next to the average player he guards, drawn wider by weight."""
    ground = 290
    gap = opp_weight_lb - weight_lb
    body = [_person(250, ground, height_in, weight_lb, c.focus, photo=photo),
            _person(450, ground, opp_height_in, opp_weight_lb, c.gray),
            f'<line x1="150" y1="{ground}" x2="550" y2="{ground}" stroke="{c.courtline}" stroke-width="3"/>',
            _text(250, ground + 30, name, 20, c.ink, 700),
            _text(250, ground + 56, f"{weight_lb:.0f} lb", 26, c.focus, 800),
            _text(450, ground + 30, "Who he guards (avg)", 20, c.muted),
            _text(450, ground + 56, f"{opp_weight_lb:.0f} lb", 26, c.ink, 800),
            _text(700, 120, f"{_signed(gap, 0)} lb", 58, c.bad if gap > 0 else c.good, 800),
            _text(700, 152, "heavier than him", 20, c.muted),
            _text(700, 232, f"{fg_allowed:.0%}", 46, c.bad if fg_allowed > team_fg else c.ink, 800),
            _text(700, 262, f"of shots go in on him (team {team_fg:.0%})", 18, c.muted)]
    return _svg(860, 360, "".join(body),
                f"{name}, {weight_lb:.0f} lb, next to the average player he guards, {opp_weight_lb:.0f} lb. "
                f"Opponents shoot {fg_allowed:.0%} against him.")


# ------------------------------------------------------------------ the shot map

ZONE_SPOTS = {"At the rim": (0, 26), "Short range": (0, 112), "Mid-range": (-150, 150), "Three": (0, 300)}


def _half_court(c: Ink) -> str:
    """NBA half court in shot-chart units (tenths of a foot), basket at (0, 0), drawn 500 x 470 with the
    basket near the top. Translate by (250, 52.5) to place it."""
    ln = f'fill="none" stroke="{c.courtline}" stroke-width="2.5"'
    corner = 89.5  # where the 22-ft corner three meets the 23.75-ft arc
    return "".join([
        f'<rect x="-250" y="-52.5" width="500" height="470" rx="4" fill="{c.floor}"/>',
        f'<rect x="-250" y="-52.5" width="500" height="470" rx="4" {ln}/>',
        f'<rect x="-80" y="-52.5" width="160" height="190" {ln}/>',
        f'<circle cx="0" cy="137.5" r="60" {ln}/>',
        f'<path d="M-40 0 A40 40 0 0 0 40 0" {ln}/>',
        f'<line x1="-30" y1="-7.5" x2="30" y2="-7.5" {ln}/>',
        f'<circle cx="0" cy="0" r="7.5" fill="none" stroke="{c.focus}" stroke-width="2.5"/>',
        f'<line x1="-220" y1="-52.5" x2="-220" y2="{corner}" {ln}/>',
        f'<line x1="220" y1="-52.5" x2="220" y2="{corner}" {ln}/>',
        f'<path d="M-220 {corner} A237.5 237.5 0 0 0 220 {corner}" {ln}/>',
        f'<path d="M-60 417.5 A60 60 0 0 1 60 417.5" {ln}/>',
    ])


def shot_map(before: list[tuple[int, int, bool]], after: list[tuple[int, int, bool]],
             zones: list[tuple[str, float, float, int, int]], c: Ink = PAGE) -> str:
    """Every shot before and after halftime on two half courts: made = filled dot, missed = small ring.

    ``before``/``after`` are (loc_x, loc_y, made); ``zones`` are (zone, pct_before, pct_after, n_before,
    n_after). Each court labels each zone with its percentage; the second court adds the change, orange
    when it fell and blue when it rose.
    """
    body = []
    for i, (title, shots, color) in enumerate((("Before halftime", before, c.muted),
                                               ("After halftime", after, c.focus))):
        ox = 250 + i * 560
        body.append(_text(ox, 34, title, 26, c.ink, 700))
        g = [_half_court(c)]
        for x, y, made in shots:
            if y > 410:
                continue
            g.append(f'<circle cx="{x}" cy="{y}" r="5.5" fill="{color}" fill-opacity="0.85"/>' if made else
                     f'<circle cx="{x}" cy="{y}" r="4" fill="none" stroke="{c.muted}" stroke-opacity="0.45" '
                     'stroke-width="1.5"/>')
        for zone, p0, p1, n0, n1 in zones:
            zx, zy = ZONE_SPOTS[zone]
            pct = p1 if i else p0
            change = p1 - p0
            sub = (f"{zone.lower()} · {'▼' if change < 0 else '▲'} {abs(change) * 100:.0f}"
                   if i and abs(change) >= 0.005 else zone.lower())
            sub_color = (c.bad if change < 0 else c.good) if i and abs(change) >= 0.005 else c.ink
            g += [f'<rect x="{zx - 78}" y="{zy - 30}" width="156" height="56" rx="10" fill="{c.floor}" '
                  'fill-opacity="0.88"/>',
                  _text(zx, zy + 2, f"{pct:.0%}", 30, c.ink, 800),
                  _text(zx, zy + 20, sub, 14, sub_color, 600)]
        body.append(f'<g transform="translate({ox},{80 + 52.5})">{"".join(g)}</g>')
    legend_y = 80 + 470 + 34
    body += [f'<circle cx="380" cy="{legend_y - 5}" r="6" fill="{c.focus}"/>',
             _text(394, legend_y, "made", 17, c.muted, anchor="start"),
             f'<circle cx="480" cy="{legend_y - 5}" r="4.5" fill="none" stroke="{c.muted}" stroke-width="1.5"/>',
             _text(494, legend_y, "missed", 17, c.muted, anchor="start"),
             _text(600, legend_y, "▼ ▲ = change in percentage points after halftime", 17, c.muted,
                   anchor="start")]
    return _svg(1060, legend_y + 14, "".join(body),
                "Clippers shots on two half courts, before and after halftime, with the shooting percentage "
                "in each zone: " + "; ".join(f"{z}: {p0:.0%} before, {p1:.0%} after" for z, p0, p1, _, _ in zones))


# ------------------------------------------------------------------ 100 missed shots

def _waffle(x0: float, y0: float, n: int, color: str, c: Ink, cell: int = 22, gap: int = 4) -> str:
    out = []
    for i in range(100):
        r, col = divmod(i, 10)
        fill = color if i < n else c.empty
        out.append(f'<rect x="{x0 + col * (cell + gap)}" y="{y0 + r * (cell + gap)}" width="{cell}" '
                   f'height="{cell}" rx="4" fill="{fill}"/>')
    return "".join(out)


def missed_shots(before: float, after: float, last_season: float, c: Ink = PAGE) -> str:
    """Two 10x10 grids: of every 100 shots the opponents miss, how many they rebound themselves."""
    nb, na = round(before * 100), round(after * 100)
    body = [_text(130, 30, "BEFORE HALFTIME", 18, c.muted, 700, spacing=2),
            _waffle(0, 48, nb, c.gray, c),
            _text(130, 336, f"{nb} of 100", 32, c.ink, 800),
            _text(470, 30, "AFTER HALFTIME", 18, c.muted, 700, spacing=2),
            _waffle(340, 48, na, c.bad, c),
            _text(470, 336, f"{na} of 100", 32, c.bad, 800),
            _text(300, 376, f"Last season after halftime: {round(last_season * 100)} of 100", 19, c.muted)]
    return _svg(600, 390, "".join(body),
                f"Of every 100 shots opponents miss, they grab back {nb} before halftime and {na} after.")


# ------------------------------------------------------------------ how the turnovers happened

def turnover_strip(own: int, second: int, no_steal: int, c: Ink = PAGE) -> str:
    """One square per turnover, grouped: stolen by his own defender, by a second defender, or no steal."""
    groups = ((own, c.gray, "Stolen by his defender"), (second, c.focus, "Stolen by a 2nd defender"),
              (no_steal, c.empty, "No steal"))
    total = own + second + no_steal
    per_row = 20
    cell, gap = 26, 6
    body, i = [], 0
    for n, color, _ in groups:
        for _ in range(n):
            r, col = divmod(i, per_row)
            stroke = f' stroke="{c.muted}" stroke-width="1.5"' if color == c.empty else ""
            body.append(f'<rect x="{col * (cell + gap)}" y="{r * (cell + gap)}" width="{cell}" height="{cell}" '
                        f'rx="5" fill="{color}"{stroke}/>')
            i += 1
    rows = -(-total // per_row)
    y = rows * (cell + gap) + 34
    for n, color, label in groups:
        stroke = f' stroke="{c.muted}" stroke-width="1.5"' if color == c.empty else ""
        body.append(f'<rect x="0" y="{y - 18}" width="22" height="22" rx="4" fill="{color}"{stroke}/>')
        body.append(_text(34, y, f"{n}", 22, c.ink, 800, anchor="start"))
        body.append(_text(70, y, label, 21, c.ink if color == c.focus else c.muted, 700 if color == c.focus else 400,
                          anchor="start"))
        y += 34
    return _svg(per_row * (cell + gap), y - 10, "".join(body),
                f"{total} turnovers: {own} stolen by his own defender, {second} by a second defender, "
                f"{no_steal} with no steal.")


# ------------------------------------------------------------------ win odds

def odds_bar(team: str, opp: str, chance: float, margin: float, c: Ink = PAGE, team_logo: str | None = None,
             opp_logo: str | None = None) -> str:
    """A tug-of-war bar: the Clippers' share of the win chance against the opponent's (with logos if given)."""
    w, x0, bar_w = 900, 40, 820
    split = x0 + bar_w * chance
    fav = team if chance >= 0.5 else opp
    body = [
        f'<rect x="{x0}" y="70" width="{bar_w}" height="56" rx="12" fill="{c.gray}"/>',
        f'<rect x="{x0}" y="70" width="{max(bar_w * chance, 24):.1f}" height="56" rx="12" fill="{c.good}"/>',
        f'<rect x="{max(split - 12, x0):.1f}" y="70" width="12" height="56" fill="{c.good}"/>' if 0.03 < chance < 0.97
        else "",
        f'<line x1="{x0 + bar_w / 2}" y1="60" x2="{x0 + bar_w / 2}" y2="136" stroke="{c.ink}" stroke-width="2" '
        'stroke-dasharray="5 5" opacity="0.6"/>',
        (f'<image href="{escape(team_logo)}" x="{x0}" y="6" width="54" height="54"/>' if team_logo else ""),
        (f'<image href="{escape(opp_logo)}" x="{x0 + bar_w - 54}" y="6" width="54" height="54"/>' if opp_logo else ""),
        _text(x0 + (64 if team_logo else 0), 46, team, 30, c.ink, 800, anchor="start", spacing=2),
        _text(x0 + bar_w - (64 if opp_logo else 0), 46, opp, 30, c.ink, 800, anchor="end", spacing=2),
        _text(x0 + 18, 109, f"{chance:.0%}", 32, "#0e0e0d" if chance > 0.12 else c.ink, 800, anchor="start"),
        _text(x0 + bar_w - 18, 109, f"{1 - chance:.0%}", 32, "#0e0e0d" if chance < 0.88 else c.ink, 800,
              anchor="end"),
        _text(x0 + bar_w / 2, 166, f"Expected margin: {fav} by {abs(margin):.1f}" if abs(margin) >= 0.5
              else "Expected margin: a toss-up", 22, c.muted),
    ]
    return _svg(w, 180, "".join(body),
                f"Win chance: {team} {chance:.0%}, {opp} {1 - chance:.0%}. Expected margin {margin:+.1f}.")


# ------------------------------------------------------------------ mini pictures for the stat tiles

TONES = {"good": "good", "bad": "bad", "warn": "focus", "info": "gray"}


def mini(viz: tuple, tone: str = "info", c: Ink = DARK) -> str:
    """A small picture of one number, drawn inside a stat tile (240 x 40).

    - ``("dots", k, n)``: n dots, the first k filled ("4 of 6 leads blown");
    - ``("pair", a, b, label_a, label_b)``: two bars, before and after ("58% -> 52%");
    - ``("share", p, mark)``: a filled bar from 0 to 1, with a tick at ``mark`` (a threshold);
    - ``("rank", r, n)``: n dots from best (right) to worst (left), the r-th one highlighted;
    - ``("diverge", values, labels)``: bars either side of zero, blue up / orange down.
    """
    color = getattr(c, TONES.get(tone, "gray"))
    kind, *a = viz
    w, h = 240, 46
    out: list[str] = []
    if kind == "dots":
        k, n = int(a[0]), int(a[1])
        r = max(3.0, min(8.0, (w - 4) / max(n, 1) / 2 - 2))
        step = min((w - 2 * r) / max(n - 1, 1), 2 * r + 7) if n > 1 else 0
        out += [f'<circle cx="{r + 1 + i * step:.1f}" cy="20" r="{r:.1f}" '
                + (f'fill="{color}"/>' if i < k else f'fill="none" stroke="{c.muted}" stroke-width="1.5"/>')
                for i in range(n)]
    elif kind == "pair":
        va, vb, la, lb = float(a[0]), float(a[1]), str(a[2]), str(a[3])
        top = max(abs(va), abs(vb), 1e-9)
        for y, v, label, fill in ((5, va, la, c.gray), (26, vb, lb, color)):
            out.append(_text(0, y + 12, label, 13, c.muted, anchor="start"))
            out.append(f'<rect x="66" y="{y}" width="{max(3.0, 170 * abs(v) / top):.1f}" height="14" rx="3" '
                       f'fill="{fill}"/>')
    elif kind == "share":
        p = float(a[0])
        mark = float(a[1]) if len(a) > 1 and a[1] is not None else None
        out.append(f'<rect x="0" y="14" width="{w}" height="12" rx="6" fill="{c.empty}"/>')
        out.append(f'<rect x="0" y="14" width="{max(6.0, w * min(max(p, 0), 1)):.1f}" height="12" rx="6" '
                   f'fill="{color}"/>')
        if mark is not None:
            x = w * mark
            out.append(f'<line x1="{x:.1f}" y1="6" x2="{x:.1f}" y2="34" stroke="{c.ink}" stroke-width="2"/>')
    elif kind == "rank":
        r_, n = int(a[0]), int(a[1])
        step = (w - 16) / max(n - 1, 1)
        for i in range(1, n + 1):
            x = w - 8 - (i - 1) * step  # 1st on the right
            out.append(f'<circle cx="{x:.1f}" cy="16" r="{7 if i == r_ else 3}" '
                       f'fill="{color if i == r_ else c.empty}"/>')
        out.append(_text(w - 2, 44, "best", 12, c.muted, anchor="end"))
        out.append(_text(2, 44, "worst", 12, c.muted, anchor="start"))
    elif kind == "diverge":
        values, labels = list(a[0]), list(a[1])
        lim = max([abs(v) for v in values] + [1e-9])
        slot = w / len(values)
        out.append(f'<line x1="0" y1="20" x2="{w}" y2="20" stroke="{c.muted}" stroke-width="1"/>')
        for i, (v, label) in enumerate(zip(values, labels)):
            x = i * slot + slot * 0.3
            hgt = max(2.0, 17 * abs(v) / lim)
            y = 20 - hgt if v >= 0 else 20
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot * 0.4:.1f}" height="{hgt:.1f}" rx="2" '
                       f'fill="{c.good if v > 0 else c.bad if v < 0 else c.gray}"/>')
            out.append(_text(x + slot * 0.2, 45 if v >= 0 else 8, label, 11, c.muted))
    else:
        return ""
    return (f'<svg class="mini" viewBox="0 0 {w} {h}" preserveAspectRatio="xMinYMid meet" aria-hidden="true" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(out)}</svg>')


# ------------------------------------------------------------------ the scoreboard

def scoreboard(team: str, opp: str, team_q: list[int], opp_q: list[int], won: bool, c: Ink = DARK,
               team_logo: str | None = None, opp_logo: str | None = None) -> str:
    """An arena-style scoreboard: points in each quarter (and overtime), the second half, and the final."""
    periods = [f"Q{i}" if i <= 4 else ("OT" if i == 5 else f"{i - 4}OT") for i in range(1, len(team_q) + 1)]
    cols = periods + ["2ND HALF", "FINAL"]
    t2, o2 = sum(team_q[2:4]), sum(opp_q[2:4])
    rows = [(team, team_q + [t2, sum(team_q)]), (opp, opp_q + [o2, sum(opp_q)])]
    x0, cw = 170, 82
    width = x0 + cw * len(cols) + 16
    body = [f'<rect x="0" y="0" width="{width}" height="196" rx="14" fill="#0e0e0d" stroke="{c.courtline}" '
            'stroke-width="2"/>']
    for j, col in enumerate(cols):
        body.append(_text(x0 + cw * j + cw / 2, 40, col, 15, c.muted, 700, spacing=1.5))
    for i, (name, vals) in enumerate(rows):
        y = 92 + i * 62
        logo = team_logo if i == 0 else opp_logo
        if logo:
            body.append(f'<image href="{escape(logo)}" x="20" y="{y - 34}" width="44" height="44"/>')
        body.append(_text(74 if logo else 26, y, name, 30, c.ink, 800, anchor="start", spacing=2))
        for j, v in enumerate(vals):
            other = rows[1 - i][1][j]
            col = cols[j]
            won_col = v > other
            fill = c.ink if won_col else c.muted
            if i == 0 and col in ("2ND HALF", "FINAL"):
                fill = c.good if v > other else c.bad if v < other else c.ink
            body.append(_text(x0 + cw * j + cw / 2, y + 2, str(v), 36 if col == "FINAL" else 30, fill,
                              800 if won_col or col in ("2ND HALF", "FINAL") else 500))
    for sep in (x0 + cw * 2, x0 + cw * len(periods)):  # halftime, then the totals
        body.append(f'<line x1="{sep}" y1="18" x2="{sep}" y2="178" stroke="{c.courtline}" stroke-width="2"/>')
    body.append(_text(x0 + cw * 2, 190, "HALFTIME", 11, c.courtline, 700, spacing=1.5))
    body.append(f'<line x1="20" y1="122" x2="{width - 20}" y2="122" stroke="{c.empty}" stroke-width="1.5"/>')
    return _svg(width, 200, "".join(body),
                f"Scoreboard: {team} {sum(team_q)}, {opp} {sum(opp_q)}. Second half {t2}-{o2}. "
                f"{'Won' if won else 'Lost'}.")


# ------------------------------------------------------------------ 1 in N, as basketballs

def _ball(cx: float, cy: float, r: float, fill: str, line: str) -> str:
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"/>'
            f'<path d="M{cx - r:.1f} {cy:.1f} H{cx + r:.1f} M{cx:.1f} {cy - r:.1f} V{cy + r:.1f}" '
            f'stroke="{line}" stroke-width="1.2" fill="none"/>')


def ball_ladder(rows: list[tuple[str, float, float]], c: Ink = PAGE, max_balls: int = 25) -> str:
    """One row per player: N basketballs, one orange, for '1 turnover every N plays' after halftime.

    ``rows``: (name, plays per turnover before halftime, plays per turnover after halftime).
    """
    r, step, x0, row_h = 9.5, 23, 190, 44
    body = [_text(x0, 22, "After halftime: one turnover every … plays", 15, c.muted, 700, anchor="start"),
            _text(x0 + step * max_balls + 40, 22, "Before halftime", 15, c.muted, 700, anchor="start")]
    for i, (name, before, after) in enumerate(rows):
        y = 58 + i * row_h
        n = max(1, round(after))
        body.append(_text(x0 - 16, y + 6, name, 17, c.ink, 700, anchor="end"))
        for k in range(min(n, max_balls)):
            body.append(_ball(x0 + r + k * step, y, r, c.bad if k == 0 else c.empty, "#1a1a19" if k == 0 else c.gray))
        label = f"{n}" + ("+" if n > max_balls else "")
        body.append(_text(x0 + min(n, max_balls) * step + 8, y + 6, label, 17, c.bad if n <= 5 else c.ink, 800,
                          anchor="start"))
        b = f"every {before:.0f}" if before == before and before > 0 else "no turnovers"
        body.append(_text(x0 + step * max_balls + 40, y + 6, b, 16, c.muted, anchor="start"))
    width = x0 + step * max_balls + 180
    return _svg(width, 40 + row_h * len(rows), "".join(body),
                "How often each player's second-half plays end in a turnover: "
                + "; ".join(f"{name} one in {after:.0f}" for name, _, after in rows))
