"""Hand-drawn SVG graphics for the dashboard, in the style of the clippers-home-court project.

Each function returns an HTML string (a ``<div class="gfx">`` wrapping an inline SVG) that the
dashboard renders with ``st.markdown(..., unsafe_allow_html=True)``. They are pictures of a
number, not charts: a court split into halves, The Wall behind one basket, players drawn to
scale, and a 10x10 grid of missed shots. Every number comes in as an argument, so the graphics
update with the data, and every picture has an ``aria-label`` that says what it shows.
"""

from __future__ import annotations

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
    good: str = "#3987e5"
    bad: str = "#e66767"
    focus: str = "#d95926"
    gray: str = "#6f6e69"
    empty: str = "#33322f"


DARK = Ink()


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

def court_halves(h1: float, h2: float, c: Ink = DARK) -> str:
    """The court split at half court: the average score margin in first halves and in second halves."""
    tint = lambda v: c.good if v > 0 else c.bad if v < 0 else None  # noqa: E731
    body = [f'<g transform="translate(20,20)">{_court(c, tint(h1), tint(h2))}</g>']
    for x, label, v in ((210, "1ST HALF", h1), (590, "2ND HALF", h2)):
        color = c.good if v > 0 else c.bad if v < 0 else c.ink
        body += [_text(x, 112, label, 22, c.muted, 700, spacing=3),
                 _text(x, 192, _signed(v), 76, color, 800),
                 _text(x, 232, "points a game", 22, c.muted)]
    body.append(_text(400, 356, "Average Clippers score margin in each half this season", 22, c.muted))
    return _svg(800, 370, "".join(body),
                f"Basketball court split in two: first halves {_signed(h1)} points a game, "
                f"second halves {_signed(h2)}.")


# ------------------------------------------------------------------ The Wall

def wall_court(ft_before: float, ft_after: float, c: Ink = DARK) -> str:
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

def _person(cx: float, ground: float, height_in: float, weight_lb: float, color: str, px_per_in: float = 3.2) -> str:
    """A simple standing figure: height to scale, shoulder width scaled by weight."""
    h = height_in * px_per_in
    top = ground - h
    head_r = h * 0.07
    shoulders = h * 0.25 * (weight_lb / 200) ** 0.9
    torso_top, torso_bot = top + head_r * 2 + 4, top + h * 0.56
    leg_w = shoulders * 0.36
    return (f'<circle cx="{cx:.1f}" cy="{top + head_r:.1f}" r="{head_r:.1f}" fill="{color}"/>'
            f'<rect x="{cx - shoulders / 2:.1f}" y="{torso_top:.1f}" width="{shoulders:.1f}" '
            f'height="{torso_bot - torso_top:.1f}" rx="{shoulders * 0.28:.1f}" fill="{color}"/>'
            f'<rect x="{cx - shoulders * 0.42:.1f}" y="{torso_bot - 2:.1f}" width="{leg_w:.1f}" '
            f'height="{ground - torso_bot + 2:.1f}" rx="{leg_w / 2:.1f}" fill="{color}"/>'
            f'<rect x="{cx + shoulders * 0.42 - leg_w:.1f}" y="{torso_bot - 2:.1f}" width="{leg_w:.1f}" '
            f'height="{ground - torso_bot + 2:.1f}" rx="{leg_w / 2:.1f}" fill="{color}"/>')


def height_matchup(name: str, height_in: float, weight_lb: float, every_tall: float, every_short: float,
                   prior_every_tall: float, prior_every_short: float, tall_in: float = 78,
                   c: Ink = DARK) -> str:
    """The player next to a bigger and a shorter defender, with how often he turns it over against each."""
    ground = 300
    body = []
    panels = ((210, f"Guarded by someone {_ft_in(tall_in)} or taller", tall_in + 2, every_tall, prior_every_tall,
               c.bad),
              (610, "Guarded by someone shorter", tall_in - 4, every_short, prior_every_short, c.good))
    for cx, title, d_height, every, prior, color in panels:
        body += [_person(cx - 50, ground, height_in, weight_lb, c.focus),
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
                 fg_allowed: float, team_fg: float, c: Ink = DARK) -> str:
    """The player next to the average player he guards, drawn wider by weight."""
    ground = 290
    gap = opp_weight_lb - weight_lb
    body = [_person(250, ground, height_in, weight_lb, c.focus),
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


# ------------------------------------------------------------------ 100 missed shots

def _waffle(x0: float, y0: float, n: int, color: str, c: Ink, cell: int = 22, gap: int = 4) -> str:
    out = []
    for i in range(100):
        r, col = divmod(i, 10)
        fill = color if i < n else c.empty
        out.append(f'<rect x="{x0 + col * (cell + gap)}" y="{y0 + r * (cell + gap)}" width="{cell}" '
                   f'height="{cell}" rx="4" fill="{fill}"/>')
    return "".join(out)


def missed_shots(before: float, after: float, last_season: float, c: Ink = DARK) -> str:
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

def turnover_strip(own: int, second: int, no_steal: int, c: Ink = DARK) -> str:
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
