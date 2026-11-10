"""Clippers Courtside dashboard.

Run with ``courtside dashboard`` (or ``streamlit run dashboard/app.py``).
Reads ``data/processed/`` and ``reports/analysis.json`` only: no API calls.
Written for two audiences at once: plain-English headlines and explanations
up front, the basketball numbers one hover or expander away.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from courtside.analysis import gamereport as R
from courtside.analysis import gameplan as GP
from courtside.analysis import preview as P
from courtside.ingest import schedule as sched
from courtside.analysis.hypotheses import CONTRIBUTES, DRIVES, ordinal
from courtside.config import get_settings
from courtside.viz import dashboard_data as D
from courtside.viz import graphics as G
from courtside.viz import story as S
from courtside.viz import whatnow as W

# One meaning per color, validated on the dark surface (dataviz validator: blue/orange pass every
# colorblind check; every mark is 3:1+ against the page):
#   blue = better / good · orange = worse / bad, the second half, and the focus · gray = context.
H1C, H2C = "#86847e", "#d95926"  # 1st half (context, 3.4:1 on the page) / 2nd half (the story)
GOOD, BAD, GRAY = "#3987e5", "#d95926", "#898781"
FOCUS = "#d95926"
INK_MARK = "#3b3f4c"
LIGHT = "#86847e"  # 3.4:1 on the light page
MUTED_BAR = "#7f7d77"  # 3.8:1 on the light page
BG = "#ffffff"  # the white panel every chart sits on (marker outlines match it)
PLOTLY_CONFIG = {"displayModeBar": False}
TONE_BADGE = {"yes": ("orange", "The answer"), "partly": ("violet", "Part of it"), "no": ("gray", "Not really")}

st.set_page_config(page_title="Clippers Courtside", page_icon=D.logo_url(get_settings().team_id), layout="wide",
                   initial_sidebar_state="collapsed")
st.logo(D.logo_url(get_settings().team_id), size="large")
st.markdown("""
<style>
.kicker {text-transform: uppercase; letter-spacing: .09em; font-size: .78rem; opacity: .65; margin-bottom: .2rem}
.hero {font-size: clamp(1.9rem, 4vw, 3rem); font-weight: 800; line-height: 1.08; margin: 0 0 .6rem 0}
.lede {font-size: 1.15rem; opacity: .85; max-width: 52rem; margin-bottom: 1.2rem}
.story {font-size: 1.12rem; line-height: 1.55}
.big-q {font-size: 1.05rem; font-weight: 700; margin-bottom: .3rem}
.rank {font-size: 2.4rem; font-weight: 800; line-height: 1; text-align: center; opacity: .8}
div[data-testid="stMetricValue"] {font-size: 1.9rem}
.jump {display: flex; flex-wrap: wrap; gap: .45rem; margin: .4rem 0 1.4rem 0}
.jump a {text-decoration: none; font-size: .9rem; padding: .3rem .8rem; border-radius: 999px;
         border: 1px solid rgba(137,135,129,.45); color: inherit}
.jump a:hover {border-color: #3987e5; color: #3987e5}
.tldr {border-left: 4px solid #d95926; padding: .2rem 0 .2rem 1rem; margin: .6rem 0 .4rem 0}
.tldr li {margin-bottom: .35rem; font-size: 1.05rem}
.vcards {display: grid; grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr)); gap: .8rem; margin: .4rem 0 1rem 0}
[data-testid="stMainBlockContainer"] {padding-top: 60px}  /* every page opens with a band, right under the menu */
/* the style-only blocks take no room (their styles still apply) */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {display: none}
.vcard {display: flex; flex-direction: column; gap: .15rem; text-decoration: none !important; color: #f0efec !important;
        background: linear-gradient(160deg, #1f3364 0%, #14213f 62%); box-shadow: 0 6px 18px rgba(19,33,63,.16); border: 1px solid #2c2c29; border-radius: 14px;
        padding: .9rem 1rem .85rem 1rem; transition: transform .12s, border-color .12s}
a.vcard:hover {transform: translateY(-2px); border-color: #4a5a85}
.vcard :is(div, span, p) {color: #f0efec}
.nextup, .pcard, .scard .sc-top {color: #f0efec}
.v-title {font-weight: 650; font-size: .9rem; line-height: 1.25; opacity: .8}
.v-dot {display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; margin-right: .4rem;
        vertical-align: .08rem}
.v-big {font-size: 1.95rem; font-weight: 800; line-height: 1.1; margin: .15rem 0 .1rem 0; letter-spacing: -.01em}
.v-mini svg {width: 100%; max-width: 280px; height: 46px; display: block; margin: .15rem 0 .25rem 0}
.v-small {font-size: .84rem; opacity: .7; line-height: 1.32}
.v-img {width: 74px; height: 54px; object-fit: cover; object-position: top; border-radius: 10px;
         background: #2c2c29; margin-bottom: .35rem}
.vcard:has(.v-img) .v-big {font-size: 1.6rem}
[data-testid="stHeaderLogo"] {height: 2.4rem; max-height: 2.4rem}
.vcards.intro {grid-template-columns: repeat(auto-fill, minmax(200px, 1fr))}
#games-list {scroll-margin-top: 4rem}
[data-testid="stLayoutWrapper"]:has(> .st-key-gamebar) {position: sticky; top: 3.6rem; z-index: 60}
.st-key-gamebar {background: #f6f5f1; padding: .35rem 0 .55rem 0; border-bottom: 1px solid #dedbd2}
.lineup {display: flex; align-items: center; gap: .9rem; padding: .55rem .2rem; border-bottom: 1px solid #e3e0d8}
.lineup .faces {display: flex}
.lineup .faces img {width: 52px; height: 52px; border-radius: 50%; object-fit: cover; object-position: top;
                    background: #e3e0d8; border: 2px solid #f6f5f1; margin-left: -10px}
.lineup .faces img:first-child {margin-left: 0}
.v-faces {display: flex; margin-bottom: .4rem}
.v-faces img {width: 46px; height: 46px; border-radius: 50%; object-fit: cover; object-position: top;
              background: #2c2c29; border: 2px solid #1e1e1c; margin-left: -12px}
.v-faces img:first-child {margin-left: 0}
.lineup .who {flex: 1; font-size: .92rem; opacity: .85}
.lineup .pm {font-size: 1.35rem; font-weight: 800; min-width: 4.5rem; text-align: right}
.lineup .min {font-size: .82rem; opacity: .6; min-width: 3.5rem}
[class*="st-key-gcard_"] {padding: .55rem .6rem !important; gap: .25rem}
.st-key-gcard_sel {border-color: #d95926 !important; background: #fbefe8}
.gc {padding-bottom: .9rem}
/* ---- matchup cards: Clippers navy and the opponent's color, split on a diagonal */
.gm {position: relative; height: 62px; border-radius: 10px; overflow: hidden; margin: -.1rem 0 1rem 0;
     background: linear-gradient(115deg, #1d428a 0 48%, #f0efec 48% 50%, var(--opp) 50% 100%)}
.gm img {position: absolute; top: 50%; transform: translateY(-50%); width: 40px; height: 40px; object-fit: contain;
         filter: drop-shadow(0 1px 3px rgba(0,0,0,.45))}
.gm .lac {left: 11%} .gm .opp {right: 11%}
/* ---- color bands, after the team site: a light band for each page's header, a navy band for its key cards */
/* bands fill the full width with border-image (paints outside the box without widening the page), and
   fade into the page color at their edges so sections ease into each other */
[class*="st-key-band_"] {position: relative; padding: 1.8rem 0 1.9rem 0; margin: 0}
/* bands that follow each other touch (no strip of page color between them) */
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-band_"]) + [data-testid="stLayoutWrapper"]:has(> [class*="st-key-band_"]) {
  margin-top: -16px}
[class*="st-key-band_light"] {border-image: linear-gradient(180deg, #ffffff 0%, #ffffff 72%, #f6f5f1 100%)
                               fill 0 // 0 100vmax}
[class*="st-key-band_navy"] {border-image: linear-gradient(180deg, #f6f5f1 0, #e6ebf5 56px, #e6ebf5 calc(100% - 56px),
                                                       #f6f5f1 100%) fill 0 // 0 100vmax}
h1, .hero {color: #13213f}
[class*="st-key-band_light"] .kicker {opacity: .7}
[class*="st-key-band_light"] .page-sub, [class*="st-key-band_light"] .lede {opacity: .85}

/* ---- Next up: the next game as the page's banner, over a faint drawn court */
.nextup {position: relative; overflow: hidden; border-radius: 18px; padding: 1.3rem 1.6rem 1.2rem 1.6rem;
         margin: .4rem 0 1.2rem 0; border: 1px solid #2c3550;
         background: radial-gradient(70% 140% at 100% 50%, color-mix(in srgb, var(--opp) 42%, transparent) 0%,
                                     transparent 70%),
                     radial-gradient(120% 140% at 0% 0%, #1d2a4a 0%, #141a2b 45%, #141413 100%)}
.nextup::before {content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background: var(--opp)}
.nextup .court {position: absolute; right: -40px; top: -10px; width: 560px; opacity: .10; pointer-events: none}
.nu-kicker {text-transform: uppercase; letter-spacing: .09em; font-size: .78rem; opacity: .75}
.nu-row {display: flex; align-items: center; gap: 1.1rem; margin: .5rem 0 .35rem 0; position: relative}
.nu-row img {width: 74px; height: 74px; object-fit: contain}
.nu-at {font-size: 1.6rem; opacity: .6; font-weight: 700}
.nu-name {font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 2.3rem; line-height: 1}
.nu-chance {margin-left: auto; text-align: right}
.nu-chance b {font-family: 'Barlow Condensed', sans-serif; font-size: 3.2rem; line-height: 1; display: block}
.nu-chance span {font-size: .85rem; opacity: .75}
.nu-key {position: relative; font-size: 1.02rem; opacity: .92; max-width: 46rem}
.nu-key b {color: #f0efec}
/* ---- player cards: trading-card style, one row that scrolls sideways */
.pcards {display: flex; gap: .8rem; overflow-x: auto; padding: .2rem .1rem .8rem .1rem; margin-bottom: .6rem}
.pcard {flex: 0 0 172px; border-radius: 14px; overflow: hidden; border: 1px solid #2c2c29; background: #1b1b19}
.pc-top {position: relative; height: 128px; background: linear-gradient(160deg, #1d428a 0%, #26335a 55%, #c8102e 140%)}
.pc-top img {position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); height: 118px}
.pc-num {position: absolute; z-index: 1; top: .4rem; left: .6rem; font-family: 'Barlow Condensed', sans-serif;
         font-weight: 800; font-size: 1.7rem; line-height: 1; color: #f0efec; text-shadow: 0 1px 6px rgba(0,0,0,.6)}
.pc-pos {position: absolute; top: .55rem; right: .6rem; font-size: .72rem; letter-spacing: .08em; opacity: .8}
.pc-body {padding: .55rem .7rem .7rem .7rem}
.pc-name {font-weight: 700; font-size: .95rem; line-height: 1.15}
.pc-stats {display: flex; gap: .9rem; margin-top: .35rem}
.pc-stats b {font-family: 'Barlow Condensed', sans-serif; font-size: 1.35rem; display: block; line-height: 1}
.pc-stats span {font-size: .72rem; opacity: .7; line-height: 1.15; display: block}
/* ---- opposing scorers: photo on their team's color, defenders listed underneath */
.scards {display: grid; grid-template-columns: repeat(var(--cols, 3), minmax(0, 1fr)); gap: .9rem; margin: .4rem 0 1rem 0}
.scard {border-radius: 14px; overflow: hidden; background: #ffffff; border: 1px solid #e3e0d8;
        box-shadow: 0 6px 18px rgba(19,33,63,.08); animation: rise .45s ease-out both}
.sc-top {position: relative; height: 118px; background: linear-gradient(150deg, var(--tc) 0%, var(--td) 100%)}
.sc-top img.face {position: absolute; bottom: 0; right: .6rem; height: 112px}
.sc-top img.logo {position: absolute; top: .6rem; left: .7rem; width: 34px; height: 34px; object-fit: contain}
.sc-name {position: absolute; left: .8rem; bottom: .55rem; font-weight: 800; font-size: 1.05rem; line-height: 1.1;
          text-shadow: 0 1px 6px rgba(0,0,0,.45); max-width: 55%}
.sc-pts {font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 2rem; padding: .5rem .9rem 0 .9rem}
.sc-pts span {font-family: inherit; font-size: .95rem; font-weight: 600; opacity: .6; margin-left: .25rem}
.sc-row {display: flex; align-items: center; gap: .55rem; padding: .35rem .9rem; border-top: 1px solid #f0ede6}
.sc-row img {width: 30px; height: 30px; border-radius: 50%; object-fit: cover; object-position: top; background: #e3e0d8}
.sc-row .who {flex: 1; font-size: .88rem}
.sc-row .pts {font-weight: 800}
.sc-foot {font-size: .78rem; opacity: .65; padding: .4rem .9rem .7rem .9rem}
/* ---- returning players: the trading card, with what they'd bring as colored chips */
.rcards {display: grid; grid-template-columns: repeat(var(--cols, 3), minmax(0, 1fr)); gap: .9rem; margin: .6rem 0 1rem 0}
.rcards .pcard {flex: none}
.rcards .pc-top {height: 140px}
.rcards .pc-top img {height: 132px}
.pc-status {font-size: .78rem; opacity: .75; margin-top: .15rem}
.chips {display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem}
.chip {font-size: .74rem; font-weight: 700; padding: .14rem .55rem; border-radius: 999px; color: #fff}
/* ---- one player: a wide card with his numbers */
.hero-card {display: grid; grid-template-columns: 220px 1fr; border-radius: 16px; overflow: hidden; margin: .4rem 0 1rem 0;
            background: #ffffff; border: 1px solid #e3e0d8; box-shadow: 0 6px 18px rgba(19,33,63,.08)}
.hc-photo {position: relative; background: linear-gradient(160deg, #1d428a 0%, #26335a 55%, #c8102e 150%); min-height: 170px}
.hc-photo img {position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); height: 160px}
.hc-photo .pc-num {font-size: 2rem}
.hc-body {padding: 1rem 1.3rem}
.hc-name {font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 2rem; line-height: 1; color: #13213f}
.hc-stats {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin-top: .8rem}
.hc-stats b {font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; display: block; line-height: 1}
.hc-stats span {font-size: .8rem; opacity: .7; line-height: 1.2; display: block; margin-top: .2rem}
.hc-up {color: #d95926} .hc-down {color: #3987e5}
/* ---- white panels on the off-white page: every chart and drawing sits on its own card */
/* the panel goes on the chart's wrapper, so the chart sizes itself to the inside of the panel */
[data-testid="stElementContainer"]:has(.stPlotlyChart) {background: #ffffff; border-radius: 16px;
  padding: .7rem .9rem .35rem .9rem; box-shadow: 0 4px 16px rgba(19,33,63,.07)}
.gfx {background: #ffffff; border-radius: 16px; padding: 1rem 1.2rem; box-shadow: 0 4px 16px rgba(19,33,63,.07);
      margin-bottom: .6rem}
.gfx.bare {background: none; box-shadow: none; padding: 0}
/* ---- side panels: the numbers that go with a drawing, beside it */
.cmp {background: #ffffff; border-radius: 16px; padding: 1rem 1.2rem .6rem 1.2rem;
      box-shadow: 0 4px 16px rgba(19,33,63,.07)}
.cmp h4 {font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 1.35rem; margin: 0 0 .25rem 0;
         color: #13213f; padding: 0}
.cmp-row {padding: .6rem 0 .45rem 0; border-top: 1px solid #f0ede6}
.cmp-top {display: flex; justify-content: space-between; align-items: baseline; gap: .6rem}
.cmp-label {font-weight: 650; font-size: .92rem}
.cmp-vals {font-family: 'Barlow Condensed', sans-serif; font-size: 1.35rem; font-weight: 800; white-space: nowrap}
.cmp-vals .b {color: #7f7d77; font-weight: 600}
.cmp-sub {font-size: .78rem; opacity: .7; margin-top: .1rem}
.trk {position: relative; height: 14px; margin: .35rem 0 .1rem 0}
.trk::before {content: ""; position: absolute; left: 0; right: 0; top: 6px; height: 2px; background: #ebe8e1;
              border-radius: 2px}
.trk .ln {position: absolute; top: 6px; height: 2px; opacity: .6}
.trk .dot {position: absolute; top: 1px; width: 12px; height: 12px; border-radius: 50%; transform: translateX(-50%)}
.who-row {display: flex; align-items: center; gap: .65rem; padding: .5rem 0; border-top: 1px solid #f0ede6}
.who-row img {width: 40px; height: 40px; border-radius: 50%; object-fit: cover; object-position: top;
              background: #e3e0d8}
.who-row .nm {flex: 1; font-size: .92rem; line-height: 1.2}
.who-row .nm span {font-size: .78rem; opacity: .65}
.who-row .n {font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 1.5rem; color: #d95926}
/* ---- who gets sloppier: one row per player, before (gray) to after (colored), two columns */
.slop {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem 1rem; margin: .4rem 0 1rem 0}
.slop-row {display: grid; grid-template-columns: 52px 1fr auto; gap: .2rem .8rem; align-items: center;
           background: #ffffff; border-radius: 14px; padding: .7rem .9rem; box-shadow: 0 4px 16px rgba(19,33,63,.07)}
.slop-row img {grid-row: span 2; width: 52px; height: 52px; border-radius: 50%; object-fit: cover;
               object-position: top; background: #e3e0d8}
.slop-row .nm {font-weight: 700}
.slop-row .vals {font-family: 'Barlow Condensed', sans-serif; font-size: 1.3rem; font-weight: 800; text-align: right}
.slop-row .vals .b {color: #7f7d77; font-weight: 600}
.slop-row .trk {grid-column: 2 / 4; margin: 0}
.chip.sm {font-size: .7rem; padding: .08rem .45rem; margin-left: .4rem; vertical-align: .1rem}
/* ---- On this page: section pills that stay under the menu and light up as you scroll */
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-toc_"]) {position: sticky; top: 60px; z-index: 55}
[class*="st-key-toc_"]:has(nav.toc) {background: rgba(246,245,241,.95); backdrop-filter: blur(6px);
                                     padding: .45rem 0 .5rem 0; border-bottom: 1px solid #e3e0d8}
nav.toc {display: flex; flex-wrap: wrap; align-items: center; gap: .4rem}
nav.toc .lbl {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; opacity: .6; margin-right: .3rem}
nav.toc a {text-decoration: none; font-size: .85rem; font-weight: 600; padding: .25rem .75rem; border-radius: 999px;
           color: #13213f; background: #ffffff; border: 1px solid #e3e0d8; transition: all .15s}
nav.toc a:hover {border-color: #1d428a}
nav.toc a.on {background: #13213f; color: #ffffff; border-color: #13213f}
.st-key-gamebar nav.toc {margin-top: .45rem}
h2, h3 {scroll-margin-top: 150px}
/* ---- the data is behind: a game finished but the morning refresh hasn't brought it in */
.behind {margin: .6rem 0 0 0; padding: .55rem .9rem; border-radius: 10px; background: #fff3e8; color: #8a3a12;
         border: 1px solid #f3c9a8; font-size: .9rem}
/* ---- small motion: cards rise in, mini-charts draw left to right, cards lift on hover */
@keyframes rise {from {opacity: 0; transform: translateY(8px)} to {opacity: 1; transform: none}}
@keyframes draw {from {clip-path: inset(0 100% 0 0)} to {clip-path: inset(0 0 0 0)}}
.vcard, .pcard, .nextup {animation: rise .45s ease-out both}
.vcard:nth-child(2), .pcard:nth-child(2) {animation-delay: .05s}
.vcard:nth-child(3), .pcard:nth-child(3) {animation-delay: .10s}
.vcard:nth-child(4), .pcard:nth-child(4) {animation-delay: .15s}
.vcard:nth-child(n+5), .pcard:nth-child(n+5) {animation-delay: .20s}
.v-mini svg {animation: draw .8s ease-out .2s both}
.pcard {transition: transform .12s, border-color .12s}
.pcard:hover {transform: translateY(-3px); border-color: #4a4a45}
@media (prefers-reduced-motion: reduce) {.vcard, .pcard, .nextup, .v-mini svg {animation: none}}
.gc-date {font-size: .78rem; opacity: .7}
.gc-opp {font-weight: 750; font-size: 1.02rem}
.gc-status {font-weight: 700; font-size: .9rem}
.ledger-sum {font-size: 1.25rem; margin: .2rem 0 .8rem 0}
.ledger {background: #ffffff; border: 1px solid #e3e0d8; border-radius: 14px; padding: .8rem 1rem}
.ledger .lhead {font-weight: 750; margin-bottom: .45rem}
.ledger-row {display: grid; grid-template-columns: 1.6fr 1fr auto; gap: .7rem; align-items: center;
             padding: .38rem 0; border-top: 1px solid #eeebe4}
.ledger .lname {font-size: .95rem}
.ledger .lbar {height: 10px; background: #ebe8e1; border-radius: 6px; overflow: hidden}
.ledger .lbar span {display: block; height: 100%; border-radius: 6px}
.ledger .lval {font-weight: 800; font-size: 1.1rem; min-width: 3.4rem; text-align: right}
.ledger .lempty {opacity: .6; font-size: .9rem}
.kval {font-weight: 800; font-size: 1.05rem; color: #3987e5; margin-top: .3rem}
.kval span {font-weight: 500; font-size: .82rem; color: inherit; opacity: .75}
.kwhy {font-size: .9rem; opacity: .85; line-height: 1.4}
.lean {display: flex; gap: .55rem; align-items: center; margin: .2rem 0}
.lean img {width: 40px; height: 40px; border-radius: 50%; object-fit: cover; object-position: top;
           background: #2c2c29}
.lean span {font-size: .8rem; opacity: .7}
.odds-note {text-align: center; font-size: 1.1rem; margin: -.2rem 0 .8rem 0}
.takeaway {font-size: 1.18rem; margin: .2rem 0 .8rem 0}
.page-sub {opacity: .75; font-size: 1.05rem; margin-top: -.6rem; margin-bottom: .8rem}
</style>
""", unsafe_allow_html=True)

settings = get_settings()


def data_stamp() -> tuple[float, int]:
    """Newest modified time and file count of the data the app reads; changes whenever a refresh lands."""
    files = [p for p in settings.processed_dir.rglob("*") if p.is_file()]
    files.append(settings.reports_dir / "analysis.json")
    return max((p.stat().st_mtime for p in files if p.exists()), default=0.0), len(files)


@st.cache_data(max_entries=1)
def load(stamp: tuple[float, int]) -> D.DashboardData:
    """Cached per ``stamp``, so a running server picks up new games without a restart."""
    return D.load(settings.processed_dir, settings.reports_dir / "analysis.json")


def show(fig: go.Figure, height: int, key: str | None = None, select: bool = False, **layout):
    """Render a plotly figure; with ``select=True`` clicks on points rerun the app and are returned."""
    settings_ = dict(height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="closest",
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",  # the white panel shows
                     legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, font=dict(size=13)),
                     hoverlabel=dict(align="left", font=dict(size=13)), font=dict(size=13),
                     barcornerradius=4, clickmode="event+select" if select else None)
    fig.update_layout(**(settings_ | layout))
    face_px = getattr(fig, "_face_px", None)
    if face_px:
        plot_h = height - 36 - 8 - face_px - 30  # plot area height, roughly, under the default margins
        for im in fig.layout.images:
            if im.yref == "paper":
                im.sizey = face_px / max(plot_h, 60)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(zeroline=False)
    return st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key,
                           on_select="rerun" if select else "ignore", selection_mode="points")


def picture(svg: str, max_width: str | None = None) -> None:
    """Render one of the SVG graphics from courtside.viz.graphics."""
    style = f' style="max-width:{max_width};margin-left:auto;margin-right:auto"' if max_width else ""
    st.markdown(svg.replace('<div class="gfx">', f'<div class="gfx"{style}>', 1), unsafe_allow_html=True)


def clicked(event, field: str, memo: str):
    """The clicked point's ``field`` if this click is new since last run (so widgets can still be used)."""
    pts = event.selection.points if event and event.selection else []
    value = pts[0].get(field) if pts else None
    if value is None or st.session_state.get(memo) == value:
        return None
    st.session_state[memo] = value
    return value


def kicker() -> str:
    """The small line above every page title: team, season, and how fresh the data is."""
    last = pd.to_datetime(data.games.loc[data.games["season"] == season_now, "game_date"]).max()
    return f'<div class="kicker">LA Clippers · {season_now} season · through {last:%b} {last.day}</div>'


def page_top(title: str, subtitle: str, show_kicker: bool = True, band: bool = True) -> None:
    box = st.container(key="band_light_" + re.sub(r"\W+", "_", title.lower())) if band else st.container()
    with box:
        if show_kicker:
            st.markdown(kicker(), unsafe_allow_html=True)
        left, right = st.columns([4, 1], vertical_alignment="bottom")
        left.title(title)
        with right:
            glossary_button()
        st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def next_step(page, blurb: str) -> None:
    """The way on to the next page of the story, at the bottom of a page."""
    st.divider()
    left, right = st.columns([3, 1], vertical_alignment="center")
    left.markdown(f'<div class="takeaway">{blurb}</div>', unsafe_allow_html=True)
    right.page_link(page, label=f"Next: {page.title} →", width="stretch")


def track(before: float, after: float, lo: float, hi: float, after_color: str) -> str:
    """A thin before -> after track: gray dot at ``before``, colored dot at ``after``, a line between."""
    pos = lambda v: max(0.0, min(100.0, 100 * (v - lo) / (hi - lo) if hi > lo else 0.0))  # noqa: E731
    a, b = pos(before), pos(after)
    return (f'<div class="trk"><span class="ln" style="left:{min(a, b):.1f}%;width:{abs(b - a):.1f}%;'
            f'background:{after_color}"></span><span class="dot" style="left:{a:.1f}%;background:{MUTED_BAR}">'
            f'</span><span class="dot" style="left:{b:.1f}%;background:{after_color}"></span></div>')


def compare_panel(title: str, rows: list[tuple]) -> str:
    """A white side panel of before -> after rows.

    ``rows``: (label, before text, after text, before value, after value, higher is better, lo, hi, small print).
    """
    out = [f'<div class="cmp"><h4>{title}</h4>']
    for label, bt, at, bv, av, up, lo, hi, sub in rows:
        better = (av > bv) if up else (av < bv)
        color = GOOD if better and abs(av - bv) > 1e-9 else BAD if abs(av - bv) > 1e-9 else MUTED_BAR
        out.append(f'<div class="cmp-row"><div class="cmp-top"><span class="cmp-label">{label}</span>'
                   f'<span class="cmp-vals"><span class="b">{_minus(bt)}</span> → '
                   f'<span style="color:{color}">{_minus(at)}</span></span></div>'
                   + track(bv, av, lo, hi, color) + (f'<div class="cmp-sub">{sub}</div>' if sub else "") + "</div>")
    return "".join(out) + "</div>"


def signed(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


try:
    data = load(data_stamp())
except FileNotFoundError as e:
    st.error(f"Processed data not found ({e.filename}). Run `courtside build` and `courtside analyze` first.")
    st.stop()

meta = data.analysis.get("meta", {})
season_now = meta.get("season", data.seasons[0])
season_prev = meta.get("comparison_season", data.seasons[-1])
focus = settings.focus_player
focus_last = focus.split(" ", 1)[-1]
PLAYER_IDS = D.player_ids(data)
TEAM_LAST = D.last_names(data.player_box, season_now, settings.team_abbreviation)

# Headlines and big numbers in a condensed sports face; body text stays in the theme's sans serif.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&display=swap');
h1, h2, h3, .hero, .v-big, .rank, .gc-opp, .ledger-sum b, div[data-testid="stMetricValue"] {
  font-family: 'Barlow Condensed', 'Source Sans Pro', sans-serif !important; letter-spacing: .005em}
h1, .hero {font-weight: 800 !important}
.hero {font-size: clamp(2.2rem, 4.6vw, 3.4rem)}
.v-big {font-size: 2.2rem}
[data-testid="stHeadingWithActionElements"] h1::after, .hero::after {
  content: ""; display: block; width: 64px; height: 4px; margin-top: .45rem; border-radius: 2px;
  background: linear-gradient(90deg, #c8102e 0 50%, #1d428a 50% 100%)}
</style>
""", unsafe_allow_html=True)


def face(name: str | None) -> str | None:
    """Small headshot URL for a player's full name (None if it isn't a player we know)."""
    pid = PLAYER_IDS.get(name) if name else None
    return D.headshot_url(pid, small=True) if pid is not None else None


def axis_faces(fig: go.Figure, sources: list[str | None], row_px: float, axis: str = "y") -> None:
    """A face next to each category label, in category order (the order the chart's rows were added).

    ``sources`` are image URLs (``face(name)``, or a team logo), one per category; None leaves that row bare.
    ``row_px`` is the height of one row (horizontal charts) or the face height (vertical ones).
    """
    h = row_px * (0.86 if axis == "y" else 1)
    for i, src in enumerate(sources):
        if not src:
            continue
        if axis == "y":
            fig.add_layout_image(source=src, xref="paper", yref="y", x=0, y=i, xanchor="right", yanchor="middle",
                                 sizex=0.5, sizey=0.86, sizing="contain", layer="above")
        else:
            fig.add_layout_image(source=src, xref="x", yref="paper", x=i, y=0, xanchor="center", yanchor="top",
                                 sizex=0.9, sizey=1, sizing="contain", layer="above")
    if axis == "y":
        fig.update_yaxes(ticklabelstandoff=int(h * 1.37) + 8)
    else:
        fig.update_xaxes(ticklabelstandoff=int(h) + 6)
        fig._face_px = h  # read by show() to size the face strip under the plot


def with_photos(df: pd.DataFrame, name_col: str = "player_name") -> pd.DataFrame:
    """The table with a photo column in front, for st.dataframe (use PHOTO_COL in its column_config)."""
    return df.assign(photo=[face(n) for n in df[name_col]])[["photo"] + list(df.columns)]


PHOTO_COL = {"photo": st.column_config.ImageColumn("", width="small")}



def glossary_button() -> None:
    with st.popover("What do these words mean?", icon=":material/menu_book:"):
        for term, text in S.GLOSSARY.items():
            st.markdown(f"**{term}**  \n{text}")



# ================================================================ the answer
STATUS_BADGE = {S.NEW: "orange", S.SAME: "violet", S.SIMILAR: "gray", S.BETTER: "blue"}
VERDICT_MARK = {"Worse than last season": "▼ Worse than last season", "Better than last season": "▲ Better than last season",
                "About the same": "= About the same"}


def page_answer() -> None:
    hl = S.headline(data.analysis, data.comparison, data.games, season_now) if not data.comparison.empty else None
    att_now, att_prev = data.attribution.get(season_now), data.attribution.get(season_prev)
    res = {r["id"]: r for r in data.analysis.get("results", [])}
    sh = D.season_halves(data.game_halves)
    reasons = (S.ranked_reasons(att_now, att_prev, sh.loc[season_now])
               if att_now is not None and att_prev is not None else [])
    top = reasons[0].factor if reasons else None
    last_game = pd.to_datetime(data.game_leads.loc[data.game_leads["season"] == season_now, "game_date"]).max()
    title = ("They're fine for a half. Then the shots stop falling." if top == "shooting"
             else "They're fine for a half. Then it falls apart.")
    with st.container(key="band_light_answer"):
        st.markdown(f'<div class="kicker">LA Clippers · {season_now} season · through {last_game:%b} '
                    f'{last_game.day}</div><div class="hero">{title}</div>', unsafe_allow_html=True)
        if hl:
            garland_main = "H3" in res and res["H3"]["key"]["share_of_swing"] >= DRIVES
            st.markdown(
                f'<div class="lede">{hl["wins"]}-{hl["losses"]}, and {hl["blown"]} of {hl["leads"]} early leads '
                f"blown. Is it {focus}'s turnovers? "
                + ("Mostly, yes." if garland_main else "Partly, but not mainly.")
                + " Click any card to see the chart behind it.</div>", unsafe_allow_html=True)
            glossary_button()
    if hl:
        with st.container(key="band_navy_answer"):
            h_ = D.season_halves(data.game_halves).loc[season_now]
            n_g = max(int(h_.loc["H1", "games"]), 1)
            court_col, num_col = st.columns([1.55, 1], gap="medium", vertical_alignment="center")
            with court_col:
                picture(G.court_halves(hl["h1_per_game"], hl["h2_per_game"]))
            num_col.markdown(compare_panel("Half by half", [
                ("Score margin, a game", signed(hl["h1_per_game"]), signed(hl["h2_per_game"]),
                 hl["h1_per_game"], hl["h2_per_game"], True, -10, 10, "first half → second half"),
                ("Shooting accuracy (eFG%)", f'{h_.loc["H1", "efg_pct"]:.0%}', f'{h_.loc["H2", "efg_pct"]:.0%}',
                 h_.loc["H1", "efg_pct"], h_.loc["H2", "efg_pct"], True, 0.40, 0.65, ""),
                ("Three-point shooting", f'{h_.loc["H1", "fg3_pct"]:.0%}', f'{h_.loc["H2", "fg3_pct"]:.0%}',
                 h_.loc["H1", "fg3_pct"], h_.loc["H2", "fg3_pct"], True, 0.25, 0.50, ""),
                ("Turnovers, a game", f'{h_.loc["H1", "tov_all"] / n_g:.1f}', f'{h_.loc["H2", "tov_all"] / n_g:.1f}',
                 h_.loc["H1", "tov_all"] / n_g, h_.loc["H2", "tov_all"] / n_g, False, 4, 14,
                 f'opponents score {h_.loc["H1", "opp_pts_off_tov"] / n_g:.1f} → '
                 f'{h_.loc["H2", "opp_pts_off_tov"] / n_g:.1f} a game off them'),
            ]), unsafe_allow_html=True)
            # The court above already shows the half-by-half margin, and section 3 lists what's new.
            answer_cards([c_ for c_ in S.answer_page_cards(hl, reasons, res, season_prev)
                          if not c_.title.startswith(("Fine before", "New problems"))])

    hurts = [r for r in reasons if r.per_game < 0]
    helps = [r for r in reasons if r.per_game >= 0]

    # ---------------------------------------------------------- the 60-second version
    if hl and reasons:
        with st.expander("Read the 60-second version as text"):
            st.markdown("\n".join(f"- {line}" for line in S.sixty_seconds(reasons, res, hl, season_prev)),
                        unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        worst = S.worst_collapse(data.game_leads, season_now)
        if worst is not None and b1.button(f"Watch the worst collapse ({worst['label']})", icon=":material/play_arrow:",
                                            type="primary", width="stretch"):
            st.session_state["jump_game"] = worst["game_id"]
            st.switch_page(PAGE_REPLAY)
        if b2.button("What needs to happen now →", icon=":material/checklist:", width="stretch"):
            st.switch_page(PAGE_WHATNOW)
        if b3.button("Compare with all 30 teams →", icon=":material/leaderboard:", width="stretch"):
            st.switch_page(PAGE_LEAGUE)
    nav_ = toc_slot("answer")

    # ---------------------------------------------------------- 1. ranked reasons (costs and helps)
    total_help = sum(r.per_game for r in helps)
    section("1. Why they blow leads, ranked", "reasons",
            "Each part of the game, after halftime: <b>orange = this season</b>, gray = last season. Left of zero "
            "costs them points; right of zero helps." if reasons else "")
    if reasons:
        ordered = hurts + sorted(helps, key=lambda r: -r.per_game)
        names = [f"{i}. {r.name}" for i, r in enumerate(ordered, 1)]
        fig = go.Figure()
        for label, vals, color in ((f"This season ({season_now})", [r.per_game for r in ordered], BAD),
                                   (f"Last season ({season_prev})", [r.prev_per_game for r in ordered], MUTED_BAR)):
            fig.add_trace(go.Bar(
                y=names, x=vals, orientation="h", name=label, marker_color=color,
                # the top row names its series right on the bar (the legend stays too)
                text=[f"{signed(v)} a game" + (f" · {label.split(' (')[0].lower()}" if i == 0 else "")
                      for i, v in enumerate(vals)], textposition="outside", cliponaxis=False,
                customdata=[r.detail for r in ordered],
                hovertemplate="<b>%{y}</b><br>" + label + ": %{x:+.1f} points a game after halftime"
                              "<br>%{customdata}<extra></extra>"))
        lo = min(min(r.per_game, r.prev_per_game) for r in ordered)
        hi = max(max(r.per_game, r.prev_per_game) for r in ordered)
        fig.add_vline(x=0, line=dict(color=INK_MARK, width=1))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Points a game after halftime  (← costs them · helps them →)",
                         range=[min(lo, 0) * 1.45 - 0.8, max(hi, 0) * 1.45 + 1.8], zeroline=False)
        show(fig, 110 + 62 * len(ordered))
        if helps and hl:
            st.caption(f"The ones right of zero hide part of the drop: without them, second halves would be "
                       f"{signed(hl['h2_per_game'] - total_help)} a game instead of {signed(hl['h2_per_game'])}.")
        with st.expander("Details for each reason"):
            for i, r in enumerate(ordered, 1):
                verb = f"costs **{-r.per_game:.1f} points a game**" if r.per_game < 0 else \
                    f"gains **{r.per_game:.1f} points a game**"
                st.markdown(f"**{i}. {r.name}** {verb} after halftime. {r.detail} "
                            f"Last season: {signed(r.prev_per_game)} a game.")
                st.badge(r.status, color=STATUS_BADGE[r.status])
            if att_now is not None:
                residual = att_now.set_index("factor").loc["residual", "per_game"]
                st.markdown(f"A further {signed(residual)} a game can't be placed in any one part.")

    # ---------------------------------------------------------- where the shots stop falling
    SHOTS_SAID = {"At the rim": "Shots at the rim", "Short range": "Short-range shots",
                  "Mid-range": "Mid-range shots", "Three": "Threes"}
    if len(data.shots) and (data.shots["season"] == season_now).any():
        z = D.shot_zones(data.shots, season_now)
        ch = (z["mean_h2"] - z["mean_h1"]).sort_values()
        worst, steady = ch.index[0], ch.abs().idxmin()
        st.subheader("Where the shots stop falling", anchor="shots")
        TOC.append((TOC_LABELS["shots"], "shots"))
        st.markdown(
            f'<div class="takeaway"><b>{SHOTS_SAID[worst]}</b> go from '
            f'{z.loc[worst, "mean_h1"]:.0%} before halftime to <b>{z.loc[worst, "mean_h2"]:.0%}</b> after'
            + (f"; {SHOTS_SAID[steady].lower()} hold about steady ({z.loc[steady, 'mean_h1']:.0%} → "
               f"{z.loc[steady, 'mean_h2']:.0%})." if steady != worst else ".") + "</div>",
            unsafe_allow_html=True)
        cur = data.shots[data.shots["season"] == season_now]
        pts = lambda h: list(zip(cur.loc[cur["half"] == h, "loc_x"], cur.loc[cur["half"] == h, "loc_y"],  # noqa: E731
                                 cur.loc[cur["half"] == h, "made"]))
        zones = [(zn, r.mean_h1, r.mean_h2, int(r.size_h1), int(r.size_h2)) for zn, r in z.iterrows()]
        n_g = max(cur["game_id"].nunique(), 1)
        map_col, zone_col = st.columns([2.35, 1], gap="medium", vertical_alignment="center")
        with map_col:
            picture(G.shot_map(pts("H1"), pts("H2"), zones))
        zone_col.markdown(compare_panel("Zone by zone", [
            (zn, f"{r.mean_h1:.0%}", f"{r.mean_h2:.0%}", r.mean_h1, r.mean_h2, True, 0.2, 0.75,
             f"{r.size_h1 / n_g:.1f} → {r.size_h2 / n_g:.1f} shots a game") for zn, r in z.iterrows()]),
            unsafe_allow_html=True)
        with st.expander("Details"):
            st.markdown(
                "- Every Clippers field-goal attempt this season from the NBA's shot charts; overtime left out. "
                "The counts match the box scores half by half.\n"
                "- Zones: *at the rim* is the restricted area, *short range* the rest of the paint, *three* "
                "includes the corners.\n- "
                + "; ".join(f"{zn}: {int(r.size_h1)} shots before halftime, {int(r.size_h2)} after"
                            for zn, r in z.iterrows()) + ".")

    # ---------------------------------------------------------- 3. theories
    section("2. The popular theories, checked", "theories")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader(f"Is it {focus_last}?")
        if "H3" in res:
            k3 = res["H3"]["key"]
            share = k3["share_of_swing"]
            fig = go.Figure()
            for x0, x1, label, shade in ((0, CONTRIBUTES, "minor", 0.08), (CONTRIBUTES, DRIVES, "part of it", 0.18),
                                         (DRIVES, 1.0, "the main cause", 0.32)):
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.5, y1=0.5, line_width=0,
                              fillcolor=f"rgba(217,89,38,{shade})", layer="below")
                fig.add_annotation(x=(x0 + x1) / 2, y=0.72, text=label, showarrow=False, font=dict(size=12))
            fig.add_trace(go.Bar(x=[min(share, 1)], y=[0], orientation="h", width=0.34, marker_color=FOCUS,
                                 hovertemplate=f"{focus_last}'s extra turnovers explain {share:.0%} of the "
                                               f"second-half drop<extra></extra>"))
            fig.add_annotation(x=min(share, 1), y=-0.62, text=f"<b>{share:.0%}</b>", showarrow=False,
                               font=dict(size=20))
            fig.update_xaxes(range=[0, 1], tickformat=".0%", tickvals=[0, CONTRIBUTES, DRIVES, 1])
            fig.update_yaxes(visible=False, range=[-0.95, 0.95])
            show(fig, 190, showlegend=False)
            st.markdown(f"His extra turnovers explain about **{share:.0%}** of the drop. "
                        + (f"That's real, but short of the **{DRIVES:.0%}** we set in advance for calling something "
                           "*the* cause." if share < DRIVES else "That's enough to call him the main cause."))
    with right:
        cards = S.answer_cards(data.analysis)
        answer_cards(S.theory_cards(cards), link=False, big_size="1.3rem")
        with st.expander("The full answer to each theory"):
            for card in cards:
                st.markdown(f"**{card.question}** {card.answer}. {card.text}")

    # ---------------------------------------------------------- The Wall
    ft = D.opp_ft_by_half(data.game_halves)
    wall = S.wall_check(ft)
    if wall is not None:
        st.subheader("Is it The Wall?", anchor="wall")
        dome = ft[(ft["where"] == "home") & (ft["season"] >= S.INTUIT_DOME_FIRST_SEASON)]
        by = dome.groupby("split")[["ftm", "fta"]].sum()
        st.markdown(f'<div class="takeaway"><b>{wall.big}.</b> {wall.small}</div>', unsafe_allow_html=True)
        picture(G.wall_court(by.loc["H1", "ftm"] / by.loc["H1", "fta"], by.loc["H2", "ftm"] / by.loc["H2", "fta"]),
                max_width="900px")
        st.link_button("See the full Intuit Dome study", settings.wall_study_url, icon=":material/open_in_new:")
        with st.expander("Each season, at home and on the road"):
            order = sorted(ft["season"].unique(), reverse=True)
            x = [f"{s_}<br>{'At Intuit Dome' if w_ == 'home' else 'On the road'}" for s_ in order
                 for w_ in ("home", "road")]
            fig = go.Figure()
            for split, color, label in (("H1", H1C, "Before halftime"), ("H2", H2C, "After halftime")):
                rows = [ft[(ft["season"] == s_) & (ft["where"] == w_) & (ft["split"] == split)]
                        for s_ in order for w_ in ("home", "road")]
                vals = [float(r_["ft_pct"].iloc[0]) if len(r_) else None for r_ in rows]
                att = [int(r_["fta"].iloc[0]) if len(r_) else 0 for r_ in rows]
                fig.add_trace(go.Bar(x=x, y=vals, name=label, marker_color=color, customdata=att,
                                     text=[f"{v:.0%}" if v is not None else "" for v in vals],
                                     textposition="outside",
                                     hovertemplate="%{x}<br>" + label + ": %{y:.1%} on %{customdata} free throws"
                                                   "<extra></extra>"))
            fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.12], title="Opponents' free throws made")
            show(fig, 320)
        with st.expander("Details"):
            st.markdown(
                "- Teams switch baskets at halftime, so visitors face The Wall for only one half. The box score "
                "doesn't record which end a team shot at, so this checks whether opponents miss more after "
                "halftime at home at all. "
                + ("They do, by more than chance would explain." if wall.big == "Maybe" else "They don't.") + "\n"
                "- Includes late free throws from intentional fouls.\n"
                "- The home-court study looks at every arena since 2013-14: visitors' free-throw shooting at "
                "Intuit Dome isn't consistently worse than at other arenas.")

    # ---------------------------------------------------------- 3. since last season
    section_n = 3
    if not data.comparison.empty and season_prev in data.attribution:
        heading(f"{section_n}. What changed since last season", "changes")
        comp = data.comparison.set_index("season")
        cur, prev = comp.loc[season_now], comp.loc[season_prev]
        cur_leads = int(cur["blown_leads"] + cur["held_leads"])
        prev_leads = int(prev["blown_leads"] + prev["held_leads"])
        new = [r.name for r in reasons if r.status == S.NEW]
        same = [r.name for r in reasons if r.status == S.SAME]
        better = [r.name for r in reasons if r.status == S.BETTER]
        answer_cards([
            W.Verdict("Early leads blown", f"{int(cur['blown_leads'])} of {cur_leads}",
                      f"Last season: {int(prev['blown_leads'])} of {prev_leads} "
                      f"({prev['blown_leads'] / max(prev_leads, 1):.0%}).", "bad", "changes",
                      ("pair", prev["blown_leads"] / max(prev_leads, 1), cur["blown_leads"] / max(cur_leads, 1),
                       "last year", "now")),
            W.Verdict("Second halves, a game", signed(cur["margin_h2_per_game"]),
                      f"Last season: {signed(prev['margin_h2_per_game'])} a game.", "bad"
                      if cur["margin_h2_per_game"] < prev["margin_h2_per_game"] else "good", "changes",
                      ("diverge", (prev["margin_h2_per_game"], cur["margin_h2_per_game"]), ("last year", "now"))),
        ] + [W.Verdict(title, " · ".join(names), sub, tone, "changes")
             for title, names, sub, tone in (
                 ("New this season", new, "Worse than last season by a point a game or more.", "bad"),
                 ("Same as last season", same, "Hurt then, hurts now.", "warn"),
                 ("Better than last season", better, "Hurts less than it did.", "good")) if names],
            link=False, big_size="1.15rem")
        with st.expander("Every stat, this season vs last (table)"):
            rows = S.scorecard(sh, season_now, season_prev)
            table = pd.DataFrame([{
                "": r.stat, f"{season_prev} 1st half": r.prev_h1, f"{season_prev} 2nd half": r.prev_h2,
                f"{season_now} 1st half": r.cur_h1, f"{season_now} 2nd half": r.cur_h2,
                "This season's 2nd-half drop-off": VERDICT_MARK[r.verdict]} for r in rows])
            st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table))
            st.caption("The last column compares how much each stat changes from 1st to 2nd half this season with "
                       "the same change last season. Opponent free throws include late fouls by a team already "
                       "behind.")
        section_n += 1

    # ---------------------------------------------------------- 5. what if
    if hl and reasons:
        heading(f"{section_n}. What if they fixed it?", "whatif")
        st.write("Switch a problem back to last season's level and see what their second halves would look like.")
        fixable = [r for r in hurts if r.prev_per_game - r.per_game >= 0.1]
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            fixed = {r.factor for r in fixable
                     if st.toggle(f"{r.name} back to last season ({signed(r.prev_per_game)} instead of "
                                  f"{signed(r.per_game)} per game)", key=f"fix_{r.factor}")}
            g_pts = res["H3"]["key"]["excess_pts"] / max(hl["games"], 1) if "H3" in res else 0.0
            fix_g = st.toggle(f"{focus_last}'s turnovers back to his last-season rate (about {signed(g_pts)} per game)",
                              key="fix_garland", disabled="turnovers" in fixed,
                              help="Already included if you fix all turnovers.")
        new_h2 = S.what_if(hl["h2_per_game"], reasons, fixed, g_pts, fix_g)
        with right:
            st.metric("Second halves, per game", signed(new_h2), border=True,
                      delta=f"{new_h2 - hl['h2_per_game']:+.1f} vs today" if new_h2 != hl["h2_per_game"] else None)
            fig = go.Figure(go.Bar(
                y=["Today", "With your fixes", "1st halves (for reference)"],
                x=[hl["h2_per_game"], new_h2, hl["h1_per_game"]], orientation="h",
                marker_color=[BAD if hl["h2_per_game"] < 0 else GOOD, BAD if new_h2 < 0 else GOOD, GRAY],
                text=[signed(v) for v in (hl["h2_per_game"], new_h2, hl["h1_per_game"])], textposition="outside",
                hovertemplate="%{y}: %{x:+.1f} per game<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=GRAY, width=1))
            lim = max(abs(hl["h2_per_game"]), abs(new_h2), abs(hl["h1_per_game"]), 2) * 1.35
            fig.update_xaxes(range=[-lim, lim], title="Score margin per game")
            fig.update_yaxes(autorange="reversed")
            show(fig, 220, showlegend=False)
        st.caption("Bookkeeping, not a prediction: real games don't change one thing at a time.")
        section_n += 1

    # ---------------------------------------------------------- 6. every stat
    heading(f"{section_n}. Every stat, before and after halftime", "stats")
    with st.expander("Open the full stat sheet", icon=":material/table:"):
        sheet = D.full_stat_sheet(data.ranks, season_now)
        st.dataframe(sheet, hide_index=True, width="stretch", height=38 + 35 * len(sheet))
    st.caption("League ranks: 1st = best of 30 teams. The change rank compares each team's second half with its "
               "own first half. These values come from the league's own team stats, so every team is measured the "
               "same way; they can differ slightly from the box-score totals above. Data: stats.nba.com via the open-source nba_api package; not affiliated with the "
               "NBA or the Clippers. Full method: reports/findings.md.")
    fill_toc(nav_)
    next_step(PAGE_PLAYERS, "Who is losing the ball after halftime, player by player.")

# =================================================================== games

TONE_COLOR = {"good": GOOD, "bad": BAD, "warn": FOCUS, "info": GRAY}


def _minus(text: str) -> str:
    """A hyphen before a digit becomes a true minus sign, so every number on the page looks the same."""
    return re.sub(r"(?<![\w])-(?=\d)", "−", text)


def answer_cards(cards: list[W.Verdict], link: bool = True, big_size: str | None = None) -> None:
    """Short-headline cards with one big number each; with ``link`` a click jumps to the card's section."""
    cards = [W.Verdict(c.title, _minus(c.big), _minus(c.small), c.tone, c.anchor, c.viz, c.image) for c in cards]

    def faces_html(c: W.Verdict) -> str:
        if c.image:
            return f'<img class="v-img" src="{c.image}" alt="" loading="lazy">'
        ids = D.names_in(f"{c.title} {c.big}", PLAYER_IDS, TEAM_LAST)[:5]
        return ('<div class="v-faces">' + "".join(f'<img src="{D.headshot_url(i, small=True)}" alt="" '
                                                 'loading="lazy">' for i in ids) + "</div>") if ids else ""

    big_style = f' style="font-size:{big_size}"' if big_size else ""
    html = "".join(
        (f'<a class="vcard" href="#{c.anchor}"' if link else '<div class="vcard"')
        + ">"
        + faces_html(c)
        + (f'<div class="v-title">{c.title}</div>' if c.viz or c.image else
           f'<div class="v-title"><span class="v-dot" style="background:{TONE_COLOR.get(c.tone, GRAY)}"></span>'
           f'{c.title}</div>')
        + f'<div class="v-big"{big_style}>{c.big}</div>'
        + (f'<div class="v-mini">{G.mini(c.viz, c.tone)}</div>' if c.viz else "")
        + f'<div class="v-small">{c.small}</div>' + ("</a>" if link else "</div>")
        for c in cards)
    n = len(cards)
    cols = n if n <= 4 else min(4, -(-n // 2))
    st.markdown(f'<div class="vcards" style="--cols:{cols}">{html}</div>', unsafe_allow_html=True)


TOC: list[tuple[str, str]] = []  # (short label, anchor) for this run's page, filled as sections are drawn
TOC_LABELS = {
    "reasons": "Ranked reasons", "shots": "Where shots stop falling", "theories": "Theories", "changes": "Since last season",
    "whatif": "What if?", "stats": "Every stat", "minutes": "Minutes", "defense": "Defense & size",
    "returning": "Returning players", "garland": "Ball handling", "more": "More evidence", "runs": "How it played out", "decided": "What decided it",
    "players": "Who played well", "matchups": "Who guarded whom", "units": "Lineups", "who": "Losing the ball",
    "sloppier": "Sloppier?", "one": "One player", "plan": "Game plan", "watch": "Their scorers",
    "tape": "Tale of the tape", "teams": "All 30 teams", "ranks": "Every rank",
}


def heading(title: str, anchor: str, toc: bool = True) -> None:
    """A section heading, listed in the page's On this page bar."""
    st.header(title, divider="gray", anchor=anchor)
    if toc:
        TOC.append((TOC_LABELS.get(anchor, title), anchor))


def toc_slot(name: str):
    """An empty slot near the top of a long page; ``fill_toc`` puts the section pills in it at the end."""
    return st.container(key=f"toc_{name}")


def fill_toc(slot) -> None:
    """Section pills for every heading drawn on this page (3 or more), the one on screen highlighted."""
    if len(TOC) < 3:
        return
    pills = "".join(f'<a href="#{a}" data-a="{a}">{label}</a>' for label, a in TOC)
    slot.html(
        f'<nav class="toc"><span class="lbl">On this page</span>{pills}</nav>'
        """<script>
        (function () {
          // Streamlit redraws elements, so look everything up live on each scroll rather than holding on to it.
          const mark = () => {
            const links = [...document.querySelectorAll('nav.toc a')];
            let current = null;
            for (const l of links) {
              const h = document.getElementById(l.dataset.a);
              if (h && h.getBoundingClientRect().top < 190) current = l.dataset.a;
            }
            links.forEach(l => l.classList.toggle('on', l.dataset.a === current));
          };
          if (!window.__tocBound) {
            let queued = false;
            document.addEventListener('scroll', () => {
              if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; mark(); }); }
            }, true);
            // pill clicks: land the heading just below the bar (Streamlit's own jump ignores scroll-margin)
            document.addEventListener('click', (e) => {
              const a = e.target.closest('nav.toc a');
              if (!a) return;
              const h = document.getElementById(a.dataset.a);
              const box = document.querySelector('[data-testid="stMain"]');
              if (!h || !box) return;
              e.preventDefault(); e.stopPropagation();
              box.scrollBy({top: h.getBoundingClientRect().top - 150, behavior: 'smooth'});
            }, true);
            window.__tocBound = true;
          }
          window.__tocMark = mark;
          setTimeout(mark, 300);
        })();
        </script>""", unsafe_allow_javascript=True)


def behind_notice() -> None:
    """Say so when a finished game isn't in the data yet (the morning refresh hasn't run)."""
    miss = D.games_missing(data.schedule, data.games, settings.team_id, pd.Timestamp.now(tz="UTC"))
    if len(miss):
        last = pd.to_datetime(data.games["game_date"]).max()
        st.markdown(f'<div class="behind"><b>Catching up:</b> {len(miss)} finished game'
                    f'{"s are" if len(miss) > 1 else " is"} not in yet. Everything here runs through '
                    f'{last:%b} {last.day}; the site refreshes each morning.</div>', unsafe_allow_html=True)


def section(title: str, anchor: str, takeaway: str = "", badge: tuple[str, str] | None = None,
            toc: bool = True) -> None:
    heading(title, anchor, toc)
    if badge:
        st.badge(badge[0], color=badge[1])
    if takeaway:
        st.markdown(f'<div class="takeaway">{takeaway}</div>', unsafe_allow_html=True)


def legend_only(fig: go.Figure, items: list[tuple[str, str, str]]) -> None:
    """Legend entries that don't depend on the color of a trace's first point."""
    for name, color, symbol in items:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(color=color, size=11, symbol=symbol)))


def page_whatnow() -> None:
    n_games = int(meta.get("games", 0))
    n = max(n_games, 1)
    page_top("What needs to happen now",
             f"What the numbers point to after {n_games} games. Click any card to see the chart behind it. "
             "Evidence, not a coaching plan.")
    nav_ = toc_slot("whatnow")
    res = {r["id"]: r for r in data.analysis.get("results", [])}
    lg = data.ranks[(data.ranks["season"] == season_now) & (data.ranks["split"] == "H2")
                    & (data.ranks["metric"] == "efg_pct")]["league_mean"]
    league_h2 = float(lg.iloc[0]) if len(lg) else float("nan")
    team = settings.team_abbreviation
    sd = W.shooting_drop(data.player_box, season_now, team, league_h2)
    sf = W.starting_five(data.player_box, data.lineups, season_now, team)
    bl = W.ball_load(data.player_box, data.lineups, res["H3"]["key"], season_now, team, focus, n) if "H3" in res else None
    back = (W.returning(data.roster, data.player_box, data.games, data.history, season_now, team,
                        settings.watch_players) if len(data.roster) else settings.watch_players)
    cands = W.candidates(data.history, data.player_box, season_now, team, back, league_h2)
    rot = W.rotation(data.player_box, data.onoff, data.bio, season_now, team) if len(data.onoff) else None
    tracked = len(data.matchups) and len(data.bio)
    dfn = W.defenders(data.matchups, data.hustle, data.bio, season_now, team) if tracked else None
    gl = W.glass(data.game_halves, season_now, season_prev)
    ids = data.player_box.loc[data.player_box["player_name"] == focus, "player_id"]
    pr = (W.pressure(data.matchups, data.turnovers, data.bio, season_now, season_prev, team, settings.team_id,
                     int(ids.iloc[0])) if tracked and len(ids) and "stealer_id" in data.turnovers else None)
    chem = W.chemistry(data.player_box, data.lineups, season_now, season_prev, team)

    with st.container(key="band_navy_whatnow"):
        answer_cards(W.verdicts(sf, bl, cands, sd, focus, rot, dfn, gl, pr, n_games))
    with st.expander("Read it as text instead"):
        st.markdown("\n".join(f"- {line}" for line in W.summary_lines(sf, bl, cands, sd, focus, rot, dfn, gl, pr)),
                    unsafe_allow_html=True)

    # ------------------------------------------------------------ minutes
    if rot is not None and len(rot):
        r = rot[rot["h2_min_on"] > 0].sort_values("h2_mpg").copy()
        losing = int((r["h2_pm_pg"] < 0).sum())
        earned = list(r.loc[r["earned"], "player_name"].map(lambda x: x.split(" ", 1)[-1]))
        section("Who plays after halftime, and how it goes", "minutes",
                f"The Clippers lose the second half with <b>{losing} of {len(r)}</b> regulars on the floor."
                + (f" {' and '.join(earned)} {'is' if len(earned) == 1 else 'are'} the exception." if earned else ""))
        tag = np.where(r["struggling"], BAD, np.where(r["earned"], GOOD, MUTED_BAR))
        too_few = ~r["struggling"] & ~r["earned"] & (r["h2_min_on"] < W.EARNED_MIN)
        labels = [f"{x.split(' ', 1)[-1]}{' (R)' if rk else ''} · {m:.0f} min"
                  for x, rk, m in zip(r["player_name"], r["rookie"], r["h2_mpg"])]
        fig = go.Figure(go.Bar(
            y=labels, x=r["h2_pm_pg"], orientation="h", showlegend=False,
            marker=dict(color=tag, pattern=dict(shape=np.where(too_few, "/", ""), fgcolor=INK_MARK, solidity=0.25)),
            text=[signed(v) for v in r["h2_pm_pg"]], textposition="outside",
            customdata=np.column_stack([r["player_name"], r["h2_mpg"], r["q4_mpg"], r["gp"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.1f} second-half min a game "
                          "(%{customdata[2]:.1f} in the 4th)<br>Clippers %{x:+.1f} a game with him on after "
                          "halftime<extra></extra>"))
        legend_only(fig, [("Fewer late minutes", BAD, "square"), ("Earned more minutes", GOOD, "square"),
                          ("Keep as is", MUTED_BAR, "square")])
        if too_few.any():
            fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Too few minutes to tell",
                                 marker=dict(color=MUTED_BAR, pattern=dict(shape="/", fgcolor=INK_MARK,
                                                                           solidity=0.25))))
        lim = float(r["h2_pm_pg"].abs().max()) * 1.35
        fig.add_vline(x=0, line=dict(color=GRAY, width=1))
        fig.update_xaxes(range=[-lim, lim], title="Points a game with him on the floor after halftime")
        axis_faces(fig, [face(x) for x in r["player_name"]], 34)
        show(fig, 110 + 34 * len(r))
        st.caption("Sorted by second-half minutes a game (most at the top). (R) = rookie. Blue = the team does much "
                   "better with him on than off, even if the bar is small; orange = the team falls well behind with him "
                   "on. Open Details for the exact rules.")
        with st.expander("Details and the full minutes table"):
            st.markdown(f"- **Fewer late minutes** = the team is {abs(W.STRUGGLE_NET):.0f}+ points per 100 trips "
                        f"behind with him on after halftime, over {W.STRUGGLE_MIN}+ minutes.\n"
                        f"- **Earned more minutes** = the team is {W.EARNED_ON_OFF:.0f}+ points per 100 trips better "
                        f"with him on than off, over {W.EARNED_MIN}+ minutes, while he plays under {W.EARNED_MPG} a game.\n"
                        "- Small samples swing on a few possessions, and on-court numbers depend on who else is "
                        "out there.")
            st.dataframe(
                with_photos(rot[["player_name", "rookie", "gp", "mpg", "h2_mpg", "q4_mpg", "h2_pm_pg", "h2_on_off",
                                 "h2_reb36"]]),
                hide_index=True, width="stretch",
                column_config=PHOTO_COL | {
                    "player_name": "Player", "rookie": st.column_config.CheckboxColumn("Rookie"),
                    "gp": "Games", "mpg": st.column_config.NumberColumn("Min/game", format="%.1f"),
                    "h2_mpg": st.column_config.NumberColumn("2nd-half min/game", format="%.1f"),
                    "q4_mpg": st.column_config.NumberColumn("4th-qtr min/game", format="%.1f"),
                    "h2_pm_pg": st.column_config.NumberColumn("2nd-half +/- a game", format="%+.1f"),
                    "h2_on_off": st.column_config.NumberColumn("On minus off (per 100)", format="%+.1f"),
                    "h2_reb36": st.column_config.NumberColumn("2nd-half rebounds/36", format="%.1f"),
                })

    # ------------------------------------------------------------ Garland
    if bl is not None:
        ratio = bl.tov_pct_h2 / bl.prior_tov_pct_h2 if bl.prior_tov_pct_h2 else float("nan")
        cause = ""
        if pr is not None:
            cause = (" The cause looks like traps." if pr.trapped else
                     " The cause looks like bigger defenders, not double teams." if pr.length_problem else "")
        section(f"Should {focus_last} handle the ball less?", "garland",
                f"<b>{bl.tov_pct_h2:.0%}</b> of his second-half plays end in a turnover, {ratio:.1f}x his usual "
                f"rate.{cause}",
                ("Yes, for now", "orange") if bl.should_handle_less else ("Not really", "blue"))
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            safest = bl.safest.head(4)
            names = [f"<b>{focus_last}</b>"] + [x.split(" ", 1)[-1] for x in safest["player_name"]]
            vals = [bl.tov_pct_h2] + list(safest["tov_pct"])
            fig = go.Figure(go.Bar(
                y=names, x=vals, orientation="h", marker_color=[FOCUS] + [MUTED_BAR] * len(safest),
                text=[f"1 in {round(1 / v)}" if v else "" for v in vals], textposition="outside",
                hovertemplate="%{y}: %{x:.1%} of second-half plays end in a turnover<extra></extra>"))
            fig.add_vline(x=bl.prior_tov_pct_h2, line=dict(color=INK_MARK, width=1, dash="dash"),
                          annotation_text=f"{focus_last} last season", annotation_position="top")
            fig.update_xaxes(tickformat=".0%", range=[0, max(vals) * 1.35],
                             title="Second-half plays that end in a turnover")
            fig.update_yaxes(autorange="reversed")
            axis_faces(fig, [face(focus)] + [face(x) for x in safest["player_name"]], 33)
            st.markdown("**How often his plays end in a turnover**")
            show(fig, 250, showlegend=False)
        with right:
            if pr is not None and pr.tov:
                st.markdown("**Is he being double-teamed into them?**")
                picture(G.turnover_strip(pr.steals - pr.help, pr.help, pr.tov - pr.steals))
                st.caption((f"{pr.help_share:.0%} are stolen by a second defender, the way a double team usually "
                            f"ends: more than his teammates ({pr.mates_help_share:.0%})." if pr.trapped else
                            f"Only {pr.help_share:.0%} are stolen by a second defender, the way a double team "
                            f"usually ends (teammates: {pr.mates_help_share:.0%}).")
                           + " Double teams aren't logged publicly; this is the closest proxy.")
        if pr is not None:
            if pr.length_problem:
                size_head = ("Bigger defenders are still a problem." if pr.size_gap_last_season
                             else "Bigger defenders are the new problem.")
            else:
                size_head = "Bigger defenders aren't a problem right now."
            st.markdown(f"**{size_head}** How often {focus_last} loses the ball, by the height of the player "
                        "guarding him:")
            me = data.bio[data.bio["player_name"] == focus].sort_values("season").tail(1)
            ht = float(me["height_in"].iloc[0]) if len(me) else 73.0
            wt = float(me["weight_lb"].iloc[0]) if len(me) else 190.0
            draw_col, who_col = st.columns([1.6, 1], gap="medium", vertical_alignment="center")
            with draw_col:
                picture(G.height_matchup(focus_last, ht, wt, 100 / pr.tov100_long, 100 / pr.tov100_short,
                                         100 / pr.prior_tov100_long, 100 / pr.prior_tov100_short, W.LONG_IN,
                                         photo=face(focus)))
            rows_ = "".join(
                f'<div class="who-row"><img src="{face(x.def_player) or ""}" alt=""><div class="nm"><b>'
                f'{x.def_player}</b><br><span>{x.def_team} · '
                f'{W._ft_in(x.height_in) if x.height_in == x.height_in else "?"}</span></div>'
                f'<div class="n">{x.tov:.0f}</div></div>' for x in pr.forcers.head(6).itertuples())
            who_col.markdown(f'<div class="cmp"><h4>Who forces them</h4><div class="cmp-sub" style="margin-bottom:'
                             f'.3rem">The defenders credited with the most of {focus_last}\'s turnovers this '
                             f"season (player tracking).</div>{rows_}</div>", unsafe_allow_html=True)
        with st.expander("Details"):
            st.markdown(
                f"- He handles {bl.play_share_h2:.0%} of the team's second-half plays and makes "
                f"{bl.tov_share_h2:.0%} of its turnovers. Getting back to his usual rate is worth about "
                f"{signed(bl.pts_per_game)} points a game.\n"
                f"- Lineups with him: {signed(bl.net_with)} per 100 trips ({bl.min_with:.0f} min); without him: "
                f"{signed(bl.net_without)} ({bl.min_without:.0f} min). Mixed up with who he plays alongside, so a "
                "flag, not proof.")
            if pr is not None:
                types = {"bad_pass": "bad passes", "lost_ball": "lost balls", "violation": "travels and violations",
                         "offensive_foul": "offensive fouls", "shot_clock": "shot-clock violations"}
                mix = ", ".join(f"{k_} {types.get(t_, t_)}" for t_, k_ in
                                sorted(pr.h2_types.items(), key=lambda kv: -kv[1]))
                f_ = pr.forcers.head(5)
                st.markdown(
                    f"- His teammates show no size gap ({pr.mates_tov100_long:.1f} vs {pr.mates_tov100_short:.1f} "
                    f"per 100), so it's specific to him.\n"
                    f"- After halftime: {mix}.\n"
                    "- Defenders credited with the most: " + "; ".join(
                        f"{x.def_player} ({x.def_team}, {W._ft_in(x.height_in) if x.height_in == x.height_in else '?'}) "
                        f"{x.tov:.0f}" for x in f_.itertuples()) + ".\n"
                    f"- Tracking covers {pr.games} of {n_games} games (it's published a few hours after each game).")

    # ------------------------------------------------------------ defense
    section("Defense, rebounding and size", "defense",
            f"After halftime opponents get back <b>{gl.opp_oreb_h2:.0%}</b> of their misses (before it "
            f"{gl.opp_oreb_h1:.0%})." + (
                " And some of the rotation is giving up a lot of size." if dfn is not None and
                dfn.table["gives_up_size"].any() else ""),
            ("Rebounding got worse", "orange") if gl.worse else ("Rebounding holds up", "blue"))
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown("**Of every 100 shots the other team misses, how many they grab back**")
        picture(G.missed_shots(gl.opp_oreb_h1, gl.opp_oreb_h2, gl.prior_opp_oreb_h2))
    with right:
        small = dfn.table[dfn.table["gives_up_size"]] if dfn is not None and len(dfn.table) else None
        if small is not None and len(small):
            q4 = rot.set_index("player_id")["q4_mpg"] if rot is not None else pd.Series(dtype=float)
            r_ = small.assign(q4=small["player_id"].map(q4).fillna(0)).sort_values("q4", ascending=False).iloc[0]
            who = r_["player_name"].split(" ", 1)[-1]
            st.markdown(f"**{who} gives up size{' as a rookie' if r_['rookie'] else ''}**")
            picture(G.size_matchup(who, r_["own_ht"], r_["own_wt"], r_["opp_ht"], r_["opp_wt"], r_["fg_pct"],
                                   dfn.team_fg, photo=face(r_["player_name"])))
    left = right = st.container()  # one column: the shooting chart, then the size table
    with left:
        if dfn is not None and len(dfn.table):
            t = dfn.table.sort_values("fg_pct", ascending=True)
            st.markdown("**How well opponents shoot when he's the defender**")
            fig = go.Figure(go.Bar(
                y=t["player_name"].str.split(" ", n=1).str[-1] + np.where(t["rookie"], " (R)", ""), x=t["fg_pct"],
                orientation="h", marker_color=[BAD if a_ else MUTED_BAR for a_ in t["attacked"]],
                text=[f"{v:.0%}" for v in t["fg_pct"]], textposition="outside",
                customdata=np.column_stack([t["player_name"], t["fga"]]),
                hovertemplate="%{customdata[0]}: opponents shoot %{x:.1%} on %{customdata[1]:.0f} shots"
                              "<extra></extra>"))
            fig.add_vline(x=dfn.team_fg, line=dict(color=INK_MARK, width=1, dash="dash"),
                          annotation_text=f"team {dfn.team_fg:.0%}", annotation_position="top")
            fig.update_xaxes(tickformat=".0%", range=[0, float(t["fg_pct"].max()) * 1.25])
            axis_faces(fig, [face(x) for x in t["player_name"]], 30)
            show(fig, 90 + 30 * len(t), showlegend=False)
            st.caption(f"Orange = opponents shoot {W.ATTACKED_FG:.0%}+ better on him than on the team. (R) = rookie.")
    with right:
        if dfn is not None and len(dfn.table):
            t = dfn.table.sort_values("wt_gap")
            with st.expander("How big are the players each one guards?"):
                fig = go.Figure()
                for r_ in t.itertuples():
                    fig.add_trace(go.Scatter(x=[r_.own_wt, r_.opp_wt], y=[r_.player_name.split(" ", 1)[-1]] * 2,
                                             mode="lines", hoverinfo="skip", showlegend=False,
                                             line=dict(color=FOCUS if r_.gives_up_size else LIGHT, width=4)))
                y = [x.split(" ", 1)[-1] for x in t["player_name"]]
                fig.add_trace(go.Scatter(x=t["own_wt"], y=y, mode="markers", name="His weight",
                                         marker=dict(color=H1C, size=12, line=dict(color=BG, width=1.5)),
                                         hovertemplate="%{y}: %{x:.0f} lb<extra>his weight</extra>"))
                fig.add_trace(go.Scatter(x=t["opp_wt"], y=y, mode="markers", name="Who he guards (average)",
                                         marker=dict(color=FOCUS, size=12, line=dict(color=BG, width=1.5)),
                                         hovertemplate="%{y} guards players averaging %{x:.0f} lb<extra></extra>"))
                fig.update_xaxes(title="Weight (lb)")
                axis_faces(fig, [face(x) for x in t["player_name"]], 30)
                show(fig, 90 + 30 * len(t))
                st.caption("Orange line = guards players 25+ lb heavier than himself.")
    if dfn is not None and len(dfn.table):
        with st.expander("Details and effort stats (contests, deflections, box-outs)"):
            for r_ in dfn.table[dfn.table["gives_up_size"]].itertuples():
                st.markdown(f"- **{r_.player_name}** ({W._ft_in(r_.own_ht)}, {r_.own_wt:.0f} lb"
                            f"{', rookie' if r_.rookie else ''}): guards players averaging {r_.opp_wt:.0f} lb; "
                            f"{r_.def_box_outs:.0f} defensive box-outs in {r_.hustle_min:.0f} minutes.")
            st.dataframe(
                with_photos(dfn.table[["player_name", "rookie", "hustle_min", "contested_shots_36", "deflections_36",
                                       "def_box_outs_36", "fg_pct", "shooting_fouls"]].sort_values(
                    "hustle_min", ascending=False)),
                hide_index=True, width="stretch",
                column_config=PHOTO_COL | {
                    "player_name": "Player", "rookie": st.column_config.CheckboxColumn("Rookie"),
                    "hustle_min": st.column_config.NumberColumn("Minutes tracked", format="%.0f"),
                    "contested_shots_36": st.column_config.NumberColumn("Contests/36", format="%.1f"),
                    "deflections_36": st.column_config.NumberColumn("Deflections/36", format="%.1f"),
                    "def_box_outs_36": st.column_config.NumberColumn("Def. box-outs/36", format="%.1f"),
                    "fg_pct": st.column_config.NumberColumn("Opp. FG%", format="percent"),
                    "shooting_fouls": st.column_config.NumberColumn("Shooting fouls", format="%.0f"),
                })
            st.caption(f"Player tracking, whole games, {dfn.games} games so far. Rebound shares: "
                       f"{gl.opp_2nd_h2:.1f} opponent second-chance points a game after halftime "
                       f"({gl.opp_2nd_h1:.1f} before it).")

    # ------------------------------------------------------------ returning players
    shooters = [c for c in cands if c.fg3a >= W.THREE_MIN_ATT]
    section("Will the returning players help?", "returning",
            ("The returning shooters hit " + ", ".join(f"<b>{c.fg3_pct:.0%}</b>" for c in shooters)
             + f" of their threes after halftime. The Clippers hit {sd.team_fg3_h2:.0%} now.") if shooters else "")
    left, right = st.columns(2, gap="large")
    with left:
        if shooters:
            xs = [c.name.split(" ", 1)[-1] for c in shooters] + ["Clippers now"]
            ys = [c.fg3_pct for c in shooters] + [sd.team_fg3_h2]
            fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[FOCUS] * len(shooters) + [MUTED_BAR],
                                   text=[f"{v:.0%}" for v in ys], textposition="outside",
                                   hovertemplate="%{x}: %{y:.1%} from three after halftime<extra></extra>"))
            fig.update_yaxes(tickformat=".0%", range=[0, max(ys) * 1.3])
            axis_faces(fig, [face(c.name) for c in shooters] + [D.logo_url(settings.team_id, dark=False)], 40, axis="x")
            st.markdown("**Three-point shooting after halftime**")
            show(fig, 300, showlegend=False)
    with right:
        bigs = [c for c in cands if "rebounding" in c.helps]
        if bigs and rot is not None:
            cur = rot[(rot["h2_mpg"] * rot["gp"] >= 40) & ~rot["player_name"].isin([c.name for c in bigs])]
            top = cur.sort_values("h2_reb36", ascending=False).head(1)
            xs = [c.name.split(" ", 1)[-1] for c in bigs] + [f"{x.split(' ', 1)[-1]} (best now)"
                                                            for x in top["player_name"]]
            ys = [c.reb36 for c in bigs] + list(top["h2_reb36"])
            fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[FOCUS] * len(bigs) + [MUTED_BAR] * len(top),
                                   text=[f"{v:.1f}" for v in ys], textposition="outside",
                                   hovertemplate="%{x}: %{y:.1f} rebounds per 36 minutes after halftime"
                                                 "<extra></extra>"))
            fig.update_yaxes(range=[0, max(ys) * 1.3])
            axis_faces(fig, [face(c.name) for c in bigs] + [face(x) for x in top["player_name"]], 40, axis="x")
            st.markdown("**Rebounds per 36 minutes after halftime**")
            show(fig, 300, showlegend=False)
    chip_color = {"shooting": "#2e8b57", "threes": "#1d7dd1", "ball security": "#6b4fbb", "rebounding": "#c8102e"}
    roster = data.roster.set_index("player_name") if len(data.roster) else pd.DataFrame(columns=["number", "position"])
    rcards = []
    for c in cands:
        info = roster.loc[c.name] if c.name in roster.index else None
        num = f"#{info['number']}" if info is not None and str(info["number"]).strip() else ""
        status = ("Out so far" if c.this_season_minutes == 0 else f"Back: {c.this_season_minutes:.0f} min so far")
        chips = "".join(f'<span class="chip" style="background:{chip_color.get(h, "#7f7d77")}">{h.capitalize()}</span>'
                        for h in c.helps) + ('<span class="chip" style="background:#7f7d77">Small sample</span>'
                                             if c.small_sample else "")
        rcards.append(
            f'<div class="pcard"><div class="pc-top"><span class="pc-num">{num}</span>'
            f'<span class="pc-pos">{info["position"] if info is not None else ""}</span>'
            f'<img src="{face(c.name) or ""}" alt="" loading="lazy"></div><div class="pc-body">'
            f'<div class="pc-name">{c.name}</div><div class="pc-status">{status}'
            + (f" · {c.mpg:.0f} min a game in {c.sample.split(',')[0]}" if c.mpg == c.mpg else "")
            + f'</div><div class="chips">{chips}</div></div></div>')
    st.markdown(f'<div class="rcards" style="--cols:{min(max(len(rcards), 1), 6)}">{"".join(rcards)}</div>',
                unsafe_allow_html=True)
    with st.expander("Details"):
        st.markdown(f"- **Who's here:** everyone on today's roster who has missed {W.MISSED_SHARE:.0%} or more of "
                    f"the games since joining the team and played {W.PRIOR_MPG:.0f}+ minutes a game in a recent "
                    "season. The list rebuilds itself after every refresh. The public stats don't say *why* a "
                    "player sat (injury, rest or coach's decision), so it includes all three.\n"
                    "- Numbers for players who haven't played much this season come from their most recent season "
                    f"with {W.SMALL_SAMPLE_MIN}+ second-half minutes, on whatever team they played for.")
        st.dataframe(with_photos(pd.DataFrame([{
            "Player": c.name, "Sample": c.sample, "2nd-half min": c.h2_minutes, "Shooting (eFG%)": c.efg,
            "3P%": c.fg3_pct, "3PA": c.fg3a, "Turnover rate": c.tov_pct, "Rebounds/36": c.reb36} for c in cands],
            columns=["Player", "Sample", "2nd-half min", "Shooting (eFG%)", "3P%", "3PA", "Turnover rate",
                     "Rebounds/36"]), "Player"),
            hide_index=True, width="stretch",
            column_config=PHOTO_COL | {"2nd-half min": st.column_config.NumberColumn(format="%.0f"),
                           "Shooting (eFG%)": st.column_config.NumberColumn(format="percent"),
                           "3P%": st.column_config.NumberColumn(format="percent"),
                           "3PA": st.column_config.NumberColumn(format="%.0f"),
                           "Turnover rate": st.column_config.NumberColumn(format="percent"),
                           "Rebounds/36": st.column_config.NumberColumn(format="%.1f")})

    heading("More evidence", "more")
    st.markdown('<div class="takeaway">The starting five, chemistry, and who stopped shooting: the cards at the top already give the answer for each.</div>', unsafe_allow_html=True)
    with st.expander("Open the starting five, chemistry and shooting", icon=":material/expand_more:"):
        # ------------------------------------------------------------ starting five
        if sf is not None:
            rest = sf.rest_h2_pm / n
            section("Does the starting five need to change?", "starters",
                    f"No. The starters hold even after halftime; the other lineups lose "
                    f"<b>{abs(rest):.1f} points a game</b>." if sf.not_the_problem and rest < 0 else
                    f"Starters: {signed(sf.h2_pm / n)} a game after halftime; everyone else {signed(rest)}.",
                    ("Keep it", "blue") if sf.not_the_problem else ("Maybe change it", "orange"), toc=False)
            with st.expander("Details"):
                st.markdown(f"- Starters: {', '.join(sf.names)}; {sf.games_together} starts together.\n"
                            f"- They've played {sf.h2_min:.0f} second-half minutes together; other lineups "
                            f"{sf.rest_h2_min:.0f}.\n"
                            f"- Per 100 trips down the floor: starters {signed(sf.h2_net)} after halftime "
                            f"({signed(sf.h1_net)} before it), everyone else {signed(sf.rest_h2_net)}.\n"
                            "- Lineup numbers swing a lot on a few possessions; treat this as a direction.")

        # ------------------------------------------------------------ chemistry
        p = chem.pairs.sort_values("pm")
        best = p.iloc[-1] if len(p) else None
        section("Chemistry: which duos work?", "chemistry",
                f"<b>{chem.continuity:.0%}</b> of minutes go to players who were here last season."
                + (f" Best duo after halftime: <b>{best['a'].split(' ', 1)[-1]} + {best['b'].split(' ', 1)[-1]}</b>."
                   if best is not None else ""), toc=False)
        left, right = st.columns([1, 2.2], gap="large")
        with left:
            st.markdown("**How much of the team is new**")
            answer_cards([W.Verdict("Minutes from players who were here last season", f"{chem.continuity:.0%}",
                                    f"New this season: {', '.join(n.split(' ', 1)[-1] for n in chem.new[:5]) or 'nobody'}.",
                                    "info", "chemistry", ("share", chem.continuity, None))], link=False)
        with right:
            if len(p):
                show_p = pd.concat([p.head(5), p.tail(5)]).drop_duplicates(subset=["a_id", "b_id"]).sort_values("pm")
                labels = [f"{a.split(' ', 1)[-1]} + {b.split(' ', 1)[-1]}{' ✦' if new else ''}"
                          for a, b, new in zip(show_p["a"], show_p["b"], show_p["new_pair"])]
                fig = go.Figure(go.Bar(
                    y=labels, x=show_p["pm"], orientation="h",
                    marker_color=[GOOD if v > 0 else BAD if v < 0 else MUTED_BAR for v in show_p["pm"]],
                    text=[f"{signed(v, 0)} in {m:.0f} min" for v, m in zip(show_p["pm"], show_p["min"])],
                    textposition="outside",
                    hovertemplate="%{y}: Clippers %{x:+.0f} after halftime with both on<extra></extra>"))
                lim = float(show_p["pm"].abs().max()) * 1.6
                fig.add_vline(x=0, line=dict(color=GRAY, width=1))
                fig.update_xaxes(range=[-lim, lim], title="Clippers' score with both on the floor after halftime")
                st.markdown("**The best and worst duos after halftime**")
                show(fig, 90 + 32 * len(show_p), showlegend=False)
                st.caption("✦ = never played together last season. Duos with 25+ second-half minutes together.")
        with st.expander("Details"):
            st.markdown(f"- **New this season:** {', '.join(chem.new) or 'nobody'}\n"
                        f"- **Back from last season:** {', '.join(chem.returning)}\n"
                        f"- {int(p['new_pair'].sum())} of {len(p)} regular duos never played together last season.\n"
                        "- Chemistry itself can't be measured. Duos overlap, so one player's bad nights show up in all "
                        "of his duos.")

        # ------------------------------------------------------------ who stopped shooting
        section("Who stopped shooting after halftime?", "shooting",
                f"<b>{sd.n_worse} of {sd.n_players}</b> regulars shoot worse after halftime.", toc=False)
        t = sd.table.sort_values("change")
        fig = go.Figure()
        for _, r_ in t.iterrows():
            fig.add_trace(go.Scatter(x=[r_["efg_h1"], r_["efg_h2"]], y=[r_["player_name"]] * 2, mode="lines",
                                     line=dict(color=BAD if r_["change"] < 0 else LIGHT, width=3), hoverinfo="skip",
                                     showlegend=False))
        for col, color, name in (("efg_h1", H1C, "Before halftime"), ("efg_h2", H2C, "After halftime")):
            fig.add_trace(go.Scatter(
                x=t[col], y=t["player_name"], mode="markers", name=name,
                marker=dict(color=color, size=13, line=dict(color=BG, width=1.5)), customdata=t[f"fga_{col[-2:]}"],
                hovertemplate="%{y}: %{x:.1%} on %{customdata:.0f} shots<extra>" + name + "</extra>"))
        fig.add_vline(x=league_h2, line=dict(color=INK_MARK, width=1, dash="dash"), annotation_text="league average",
                      annotation_position="top")
        fig.update_xaxes(tickformat=".0%", title="Shooting accuracy (eFG%, which counts threes as worth more)")
        axis_faces(fig, [face(x) for x in t["player_name"]], 40)
        show(fig, 90 + 40 * len(t))
        with st.expander("Details"):
            st.markdown("The biggest drops come from players who were unusually hot before halftime, so some of this "
                        "should even out. Players with 40+ minutes in each half.")

    # ------------------------------------------------------------ what would change this
    with st.expander("What would change this advice", icon=":material/update:"):
        watch = []
        if "H3" in res:
            k3 = res["H3"]["key"]
            watch.append(f"**{focus_last} stops being the main cause** if his extra turnovers fall from "
                         f"{k3['share_of_swing']:.0%} to under {DRIVES:.0%} of the second-half drop."
                         if k3["share_of_swing"] >= DRIVES else
                         f"**{focus_last} becomes the main cause** if his extra turnovers grow from "
                         f"{k3['share_of_swing']:.0%} to {DRIVES:.0%} of the second-half drop.")
        if "H1" in res:
            k1 = res["H1"]["key"]
            sure = k1["boot_lo"] > 0 or k1["boot_hi"] < 0
            watch.append(("**The collapse could still be bad luck** if the 90% range for the drop widens to "
                          "include zero again." if sure else
                          "**The collapse is confirmed (not bad luck)** once the 90% range for the drop stops "
                          "including zero.") + f" It's {signed(k1['boot_lo'])} to {signed(k1['boot_hi'])} today.")
        watch.append((f"**The shooting has recovered**: second-half shooting is above the league average "
                      f"({league_h2:.1%}). The problem is back if it dips under." if sd.team_h2 >= league_h2 else
                      f"**The shooting problem is fading** if second-half shooting climbs back above the league "
                      f"average ({league_h2:.1%}).") + f" It's {sd.team_h2:.1%} today.")
        if pr is not None:
            watch.append((f"**{focus_last} has adjusted to bigger defenders** once his turnovers against them "
                          f"drop under {W.LENGTH_RATIO}x his rate against shorter ones." if pr.length_problem else
                          f"**Bigger defenders become a problem for {focus_last}** if his turnovers against them "
                          f"climb to {W.LENGTH_RATIO}x his rate against shorter ones.")
                         + f" It's {pr.size_ratio:.1f}x today.")
        watch.append(("**The rebounding is fixed** when opponents get back under " if gl.worse else
                      "**The rebounding slips** if opponents get back more than ")
                     + f"{gl.prior_opp_oreb_h2 + W.GLASS_WORSE:.1%} of their misses after halftime. It's "
                     f"{gl.opp_oreb_h2:.1%} today.")
        st.markdown("\n".join(f"- {w}" for w in watch))
        st.caption("Everything on this page reruns with `courtside all` after each game. The rules behind each "
                   "verdict are listed at the top of src/courtside/viz/whatnow.py.")
    fill_toc(nav_)


def _step_game(key: str, labels: list[str], step: int) -> None:
    """Previous / next game buttons."""
    cur = st.session_state.get(key, labels[-1])
    i = labels.index(cur) if cur in labels else len(labels) - 1
    st.session_state[key] = labels[max(0, min(len(labels) - 1, i + step))]


def _set_game(key: str, label: str) -> None:
    st.session_state[key] = label


def _upcoming(season: str) -> tuple[pd.DataFrame, list, pd.DataFrame, float | None]:
    """Games left this season, their previews, the team table, and the season's second-half shooting cost."""
    if season != season_now or data.schedule.empty or data.league_periods.empty:
        return pd.DataFrame(), [], pd.DataFrame(), None
    teams = P.blended(P.team_table(data.league_periods, season_now), P.team_table(data.league_periods, season_prev))
    att = data.attribution.get(season_now)
    shooting = float(att.set_index("factor").loc["shooting", "per_game"]) if att is not None else None
    tid = settings.team_id
    up = data.schedule[((data.schedule["home_id"] == tid) | (data.schedule["away_id"] == tid))
                       & (data.schedule["status"] < 3)].sort_values("tipoff_utc").reset_index(drop=True)
    return up, [P.preview(g, teams, tid, shooting) for _, g in up.iterrows()], teams, shooting


def _hide_intro() -> None:
    st.session_state["intro_hidden"] = True


def intro() -> None:
    """The welcome at the top of the landing page, in its light band."""
    with st.container(key="band_light_intro"):
        _intro()


def _intro() -> None:
    """What this is, and one live tile per page."""
    g = data.games[data.games["season"] == season_now]
    wins, losses = int((g["result"] == "W").sum()), int((g["result"] == "L").sum())
    last = pd.to_datetime(g["game_date"]).max()
    lc = data.game_leads.loc[data.game_leads["season"] == season_now, "lead_class"]
    blown, leads = int((lc == "blown_lead").sum()), int(lc.isin(["blown_lead", "held_lead"]).sum())
    hl = (S.headline(data.analysis, data.comparison, data.games, season_now)
          if not data.comparison.empty and season_now in set(data.comparison["season"]) else None)
    h1, h2 = (hl["h1_per_game"], hl["h2_per_game"]) if hl else (np.nan, np.nan)
    ph = D.player_halves(data.player_stats, season_now)
    top = ph[ph["split"] == "H2"].sort_values("tov_share").iloc[-1] if len(ph) else None
    rk = data.ranks[(data.ranks["season"] == season_now) & data.ranks["is_team"]
                    & (data.ranks["metric"] == "net_rtg") & (data.ranks["split"] == "H2")]["rank"]

    tiles = [("games-list", "Games", f"{wins}-{losses}",
              "Pick any game below: a report for each one played, a game plan for each one coming up.",
              None, None, "info")]
    if leads:
        tiles.append(("answer", "Blown leads", f"{blown} of {leads}",
                      "early leads they let slip. Why the second half goes wrong.", ("dots", blown, leads), None,
                      "bad" if blown * 2 >= leads else "info"))
    if not np.isnan(h2):
        tiles.append(("what-now", "What now?", f"{signed(h2)} per game",
                      "after halftime. What the numbers say has to change.",
                      ("pair", h1, h2, "1st half", "2nd half"), None, "bad" if h2 < h1 else "good"))
    if top is not None:
        tiles.append(("players", "Players", top["player_name"],
                      f"makes {top['tov_share']:.0%} of the turnovers after halftime. Who else, and how.",
                      None, D.headshot_url(int(top["player_id"])), "warn"))
    if len(rk):
        r = int(rk.iloc[0])
        tiles.append(("league", "Vs. the league", f"{ordinal(r)} of 30",
                      "in the second half. How every team changes after halftime.", ("rank", r, 30), None,
                      "bad" if r > 20 else "good" if r <= 10 else "warn"))

    st.markdown(f'<div class="kicker">LA Clippers · {season_now} season · updated through {last:%b} {last.day}'
                '</div><div class="hero">Clippers Courtside</div>'
                '<div class="lede">A fan-made tracker of the Clippers\' season, built from the NBA\'s own play-by-play '
                'data. It began with one question (why do they keep losing leads after halftime?) and now follows '
                'every game. It updates the morning after each game. Start anywhere:</div>',
                unsafe_allow_html=True)
    html = "".join(
        f'<a class="vcard" href="{"#" + slug if slug == "games-list" else slug}" target="_self">'
        + (f'<img class="v-img" src="{img}" alt="" loading="lazy">' if img else "")
        + f'<div class="v-title"><span class="v-dot" style="background:{TONE_COLOR.get(tone, GRAY)}"></span>'
          f'{title}</div><div class="v-big">{_minus(big)}</div>'
        + (f'<div class="v-mini">{G.mini(viz, tone)}</div>' if viz else "")
        + f'<div class="v-small">{small}</div></a>'
        for slug, title, big, small, viz, img, tone in tiles)
    st.markdown(f'<div class="vcards intro">{html}</div>', unsafe_allow_html=True)
    left, right = st.columns([5, 1], vertical_alignment="center")
    with left:
        with st.expander("How it works"):
            st.markdown(
                "- **Data:** every play of every Clippers game, pulled from stats.nba.com with the open-source "
                "[nba_api](https://github.com/swar/nba_api) package, compared with last season and the rest of "
                "the league.\n"
                "- **Updates:** each morning, a refresh pulls the newest games and reruns everything. Every page "
                "and every number moves with it.\n"
                "- **Odds** for upcoming games come from a simple team-strength model: no injuries, rest or "
                "betting lines.\n"
                "- Not affiliated with the NBA or the Clippers. Words you don't know? Try the "
                "**What do these words mean?** button.")
    right.button("Hide intro", icon=":material/close:", on_click=_hide_intro, width="stretch", type="tertiary")


COURT_LINES = ('<svg class="court" viewBox="0 0 760 300" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" '
               'fill="none" stroke="#ffffff" stroke-width="4"><rect x="2" y="2" width="756" height="296" rx="6"/>'
               '<line x1="380" y1="2" x2="380" y2="298"/><circle cx="380" cy="150" r="45"/>'
               '<rect x="2" y="95" width="150" height="110"/><rect x="608" y="95" width="150" height="110"/>'
               '<circle cx="152" cy="150" r="45"/><circle cx="608" cy="150" r="45"/>'
               '<path d="M2 30 L50 30 A140 140 0 0 1 50 270 L2 270"/>'
               '<path d="M758 30 L710 30 A140 140 0 0 0 710 270 L758 270"/></svg>')


def _future_label(t: pd.Timestamp, pv, j: int) -> str:
    """How an upcoming game reads in the game picker (the banner's button opens it by this label)."""
    return (f"{t:%b} {t.day} · {'vs' if pv.home else '@'} {pv.opponent} · "
            f"{'next game, ' if j == 0 else ''}{pv.chance:.0%} to win")


def next_up(box, up: pd.DataFrame, previews: list, teams: pd.DataFrame, shooting: float | None) -> None:
    """The next game as a banner: both logos, when and where, the win chance, and the top key to it."""
    if not previews:
        return
    pv, g = previews[0], up.iloc[0]
    t = g["tipoff_utc"].tz_convert("America/Los_Angeles")
    opp_id = int(g["away_id"] if pv.home else g["home_id"])
    players = GP.player_table(data.player_box, season_now, settings.team_abbreviation)
    misc = data.team_misc[data.team_misc["season"] == season_now] if len(data.team_misc) else pd.DataFrame(
        columns=["team_id", "opp_pts_paint"])
    keys = GP.game_plan(pv, teams, misc, players, shooting)
    key = (f'<div class="nu-key">The key: <b>{keys[0].title}</b>, worth about {signed(keys[0].points)} a game. '
           f'{keys[0].why}</div>') if keys else ""
    when = f"{t:%a %b} {t.day} · {t:%I:%M %p}".replace(" 0", " ") + f" PT · {pv.arena}"
    with box:
        st.markdown(
            f'<div class="nextup" style="--opp:{D.team_color(opp_id)}">{COURT_LINES}'
            f'<div class="nu-kicker">Next up · {when}</div>'
            f'<div class="nu-row"><img src="{D.logo_url(settings.team_id)}" alt="Clippers">'
            f'<span class="nu-at">{"vs" if pv.home else "@"}</span><img src="{D.logo_url(opp_id)}" alt="">'
            f'<span class="nu-name">{pv.opponent_name}</span>'
            f'<div class="nu-chance"><b>{pv.chance:.0%}</b><span>chance to win</span></div></div>{key}</div>',
            unsafe_allow_html=True)
        st.button("Open the game plan", icon=":material/sports_basketball:", key="open_next_up",
                  on_click=_set_game, args=(f"pick_{season_now}", _future_label(t, pv, 0)))


def page_replay() -> None:
    if not st.session_state.get("intro_hidden"):
        intro()
        st.markdown('<div id="games-list"></div>', unsafe_allow_html=True)
    banner = st.container()
    hidden = bool(st.session_state.get("intro_hidden"))
    page_top("Games", "Every game this season: the report for games played, the game plan for games coming up. "
             "Updates after each game.", show_kicker=hidden, band=hidden)
    leads = data.game_leads.set_index("game_id")
    g_all = data.games.set_index("game_id")
    season = st.session_state.get("game_season", data.seasons[0])
    if season not in data.seasons:
        season = data.seasons[0]
    options = D.game_options(data.game_leads, season).iloc[::-1].reset_index(drop=True)  # oldest first
    strip = leads.loc[options["game_id"]].reset_index()
    gi = g_all.reindex(strip["game_id"])
    entries = [{"kind": "past", "gid": g, "date": pd.Timestamp(d), "home": bool(h), "opp": o, "opp_id": int(oid),
                "status": f"{r} {int(tp)}-{int(op)}", "won": r == "W",
                "label": f"{pd.Timestamp(d):%b} {pd.Timestamp(d).day} · {'vs' if h else '@'} {o} · {r} {int(tp)}-{int(op)}"}
               for g, d, h, o, oid, r, tp, op in zip(strip["game_id"], gi["game_date"], gi["is_home"],
                                                     gi["opponent_abbreviation"], gi["opponent_id"], gi["result"],
                                                     gi["team_pts"], gi["opp_pts"])]
    up, previews, teams, shooting = _upcoming(season)
    for j, (pv, (_, g)) in enumerate(zip(previews, up.iterrows())):
        t = g["tipoff_utc"].tz_convert("America/Los_Angeles")
        entries.append({"kind": "future", "idx": j, "date": t, "home": pv.home, "opp": pv.opponent,
                        "opp_id": int(g["away_id"] if pv.home else g["home_id"]), "status": f"{pv.chance:.0%} to win",
                        "won": None, "label": _future_label(t, pv, j)})
    if season == season_now:
        next_up(banner, up, previews, teams, shooting)
    labels = [e["label"] for e in entries]
    n_past = len(strip)
    xs = pd.to_datetime(strip["game_date"]).dt.strftime("%b %d").str.replace(" 0", " ") + " " + strip[
        "opponent_abbreviation"]
    by_x = dict(zip(xs, labels[:n_past]))
    pick_key = f"pick_{season}"
    st.session_state.setdefault(pick_key, labels[n_past - 1] if n_past else labels[0])  # the latest result
    jump = st.session_state.pop("jump_game", None)
    if jump in set(strip["game_id"]):
        st.session_state[pick_key] = labels[list(strip["game_id"]).index(jump)]
    x = clicked(st.session_state.get(f"strip_{season}"), "x", f"strip_click_{season}")  # a click on the strip
    if x in by_x:
        st.session_state[pick_key] = by_x[x]
    if st.session_state[pick_key] not in labels:
        st.session_state[pick_key] = labels[n_past - 1] if n_past else labels[0]

    # ---- the game picker: pinned to the top of the page while scrolling
    with st.container(key="gamebar"):
        c_season, c_prev, c_pick, c_next, c_logo = st.columns([1.2, 0.45, 4, 0.45, 0.5], vertical_alignment="bottom",
                                                                gap="small")
        c_season.selectbox("Season", data.seasons, key="game_season")
        c_prev.button("◀", key="prev_game", on_click=_step_game, args=(pick_key, labels, -1), help="Previous game",
                      width="stretch")
        pick = c_pick.selectbox("Game", labels, key=pick_key)
        c_next.button("▶", key="next_game", on_click=_step_game, args=(pick_key, labels, 1), help="Next game",
                      width="stretch")
        idx = labels.index(pick)
        entry = entries[idx]
        c_logo.image(D.logo_url(entry["opp_id"], dark=False), width=46)
        nav_ = st.container()  # the On this page pills, filled once the report or plan is drawn

    # ---- the schedule around this game: played games and the ones coming up
    lo = max(0, min(idx - 3, len(entries) - 7))
    window = entries[lo:lo + 7]
    cols = st.columns(len(window), gap="small")
    for k, (col, e) in enumerate(zip(cols, window), start=lo):
        sel = k == idx
        with col.container(border=True, key=f"gcard_sel" if sel else f"gcard_{k}"):
            oc = D.team_color(e["opp_id"])
            st.markdown(f'<div class="gm" style="--opp:{oc}"><img class="lac" src="{D.logo_url(settings.team_id)}" '
                        f'alt="Clippers"><img class="opp" src="{D.logo_url(e["opp_id"], dark=not D.is_light(oc))}" '
                        f'alt="{e["opp"]}"></div>', unsafe_allow_html=True)
            color = GOOD if e["won"] else BAD if e["won"] is False else INK_MARK
            st.markdown(f'<div class="gc"><div class="gc-date">{e["date"]:%a %b} {e["date"].day}</div>'
                        f'<div class="gc-opp">{"vs" if e["home"] else "@"} {e["opp"]}</div>'
                        f'<div class="gc-status" style="color:{color}">{e["status"]}</div></div>',
                        unsafe_allow_html=True)
            st.button("Showing" if sel else ("Report" if e["kind"] == "past" else "Game plan"), key=f"open_{k}",
                      on_click=_set_game, args=(pick_key, e["label"]), disabled=sel, width="stretch")

    if entry["kind"] == "future":
        _game_plan(up.iloc[entry["idx"]], previews[entry["idx"]], teams, shooting)
        fill_toc(nav_)
        return

    gid = entry["gid"]
    row = leads.loc[gid]
    where = "Home, Intuit Dome" if bool(g_all.loc[gid, "is_home"]) else "Away"
    st.markdown(f'<div class="kicker">{pd.Timestamp(row["game_date"]):%A, %B} {pd.Timestamp(row["game_date"]).day} · '
                f'{where}</div>', unsafe_allow_html=True)

    with st.expander("Season at a glance: every game from halftime to the final (click one to open it)"):
        fig = go.Figure()
        for xx, h, f, blown in zip(xs, strip["halftime_margin"], strip["final_margin"],
                                   strip["lead_class"] == "blown_lead"):
            fig.add_trace(go.Scatter(x=[xx, xx], y=[h, f], mode="lines",
                                     line=dict(color=BAD if blown else LIGHT, width=3), hoverinfo="skip",
                                     showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=strip["halftime_margin"], mode="markers", name="At halftime",
                                 marker=dict(color=H1C, size=11, line=dict(color=BG, width=1.5)),
                                 hovertemplate="%{x}<br>At halftime: %{y:+d}<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=strip["final_margin"], mode="markers", name="Final",
                                 marker=dict(color=H2C, size=11, symbol="diamond", line=dict(color=BG, width=1.5)),
                                 hovertemplate="%{x}<br>Final: %{y:+d}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        if len(strip) > 20:
            fig.update_xaxes(showticklabels=False)
        show(fig, 230, key=f"strip_{season}", select=True, yaxis_title="Clippers margin")
        st.caption("Halftime (●) to final (◆). Orange = a blown early lead.")

    halves = data.game_halves[(data.game_halves["game_id"] == gid)
                              & data.game_halves["split"].astype(str).isin(["H1", "H2"])]
    att = D.game_attribution(data.game_halves, gid)
    mq = D.game_margin(data.margins, gid).groupby("period")[["team_score", "opp_score"]].max()
    team_q = mq["team_score"].diff().fillna(mq["team_score"]).astype(int).tolist()
    opp_q = mq["opp_score"].diff().fillna(mq["opp_score"]).astype(int).tolist()
    opp = row["opponent_abbreviation"]
    names = data.player_box.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    lines = R.player_lines(data.player_box, gid, "LAC")
    opp_lines = R.player_lines(data.player_box, gid, opp)
    factors = R.vs_normal(data.game_halves, gid)
    seg = data.segments[data.segments["game_id"] == gid] if len(data.segments) else pd.DataFrame()
    game_units = R.units(seg, names)
    plays = data.plays[data.plays["game_id"] == gid] if len(data.plays) else pd.DataFrame()
    run_against, run_for = R.biggest_runs(data.margins, gid)
    if len(plays) and len(seg):
        run_against = R.describe_run(run_against, plays, seg, names) if run_against else None
        run_for = R.describe_run(run_for, plays, seg, names) if run_for else None
    game_def = R.defenders_for_game(data.matchups, gid, "LAC") if len(data.matchups) else pd.DataFrame()
    sbox = data.player_box[(data.player_box["season"] == season) & (data.player_box["team_abbreviation"] == "LAC")
                           & (data.player_box["split"].astype(str) == "FULL")].groupby("player_name")[
        ["minutes", "fga", "fta", "tov", "oreb", "dreb", "fgm", "fg3m"]].sum().reset_index()
    sd_ = W.defenders(data.matchups, data.hustle, data.bio, season, "LAC") if len(data.matchups) and len(
        data.bio) else None
    season_def = sd_.table if sd_ is not None else None
    lvs = R.levers(factors, lines, sbox, game_def, season_def, game_units) if len(factors) else []

    # ---- the result
    opp_id_ = int(g_all.loc[gid, "opponent_id"])
    # the scoreboard is its own white card, so it skips the panel padding and runs the full width
    st.markdown(G.scoreboard("LAC", opp, team_q, opp_q, row["final_margin"] > 0,
                             team_logo=D.logo_url(settings.team_id, dark=False),
                             opp_logo=D.logo_url(opp_id_, dark=False), team_color="#1d428a",
                             opp_color=D.team_color(opp_id_)).replace('<div class="gfx">',
                                                                      '<div class="gfx" style="padding:0">', 1),
                unsafe_allow_html=True)
    with st.container(border=True):
        tag = S.LEAD_TAG.get(row["lead_class"], "")
        if tag:
            st.badge(tag, color="orange" if row["lead_class"] == "blown_lead" else "blue")
        st.markdown(f'<div class="story">{S.game_story(row, halves, att)}</div>', unsafe_allow_html=True)
        if len(factors):
            st.markdown(f'<div class="takeaway"><b>{R.verdict(factors, lvs)}</b></div>', unsafe_allow_html=True)

    # ---- the tiles: best, worst, the runs, turnovers
    best, worst = R.best_and_worst(lines)
    tiles = []
    for who, title, tone in ((best, "Best Clipper", "good"), (worst, "Toughest night", "bad")):
        if who is not None:
            tiles.append(W.Verdict(
                f"{title}: {who['player_name']}", f"{who['game_score']:.1f}",
                f"Game Score. {int(who['pts'])} pts, {int(who['reb'])} reb, {int(who['ast'])} ast, "
                f"{int(who['tov'])} TO in {who['minutes']:.0f} min; {signed(who['plus_minus'], 0)} on the floor.",
                tone, "players"))
    for run, title, tone in ((run_against, f"Biggest {opp} run", "bad"), (run_for, "Biggest Clippers run", "good")):
        if run is not None:
            who = ", ".join(n.split(" ", 1)[-1] for n in run.unit) if run.unit else "?"
            tiles.append(W.Verdict(title, run.label, f"{R.run_window(run)}. On the floor: {who}.", tone, "runs",
                                   ("diverge", (run.team_pts, -run.opp_pts), ("LAC", opp))))
    tiles.append(S.game_cards(row, halves, data.turnovers, focus_last)[-1])
    answer_cards(tiles)

    # ---- the replay, with the runs shaded
    heading("How it played out", "runs")
    m = D.game_margin(data.margins, gid)
    end = max(48.0, float(m["minute"].max()))
    blown = row["lead_class"] == "blown_lead"
    line_color = BAD if blown else H1C
    show_tov = st.toggle("Show Clippers turnovers on the chart", value=True,
                         help=f"Orange diamonds are {focus}'s turnovers; gray circles are everyone else's.")
    fig = go.Figure()
    for run, color in ((run_against, "rgba(217,89,38,0.16)"), (run_for, "rgba(57,135,229,0.16)")):
        if run is not None:
            fig.add_vrect(x0=run.start / 60, x1=run.end / 60, fillcolor=color, line_width=0, layer="below",
                          annotation_text=("LAC " if not run.against else f"{opp} ") + run.label,
                          annotation_position="top left", annotation_font_size=12)
    fig.add_vline(x=24, line=dict(color=GRAY, width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1))
    custom = np.column_stack([m["team_score"], m["opp_score"], m["period"]])
    fig.add_trace(go.Scatter(x=m["minute"], y=m["margin"], mode="lines", line=dict(color=LIGHT, width=1.5, shape="hv"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=m["minute"], y=m["margin"], mode="lines", line=dict(color=line_color, width=2.5, shape="hv"),
        customdata=custom, name="Clippers margin",
        hovertemplate="Minute %{x:.1f}<br>LAC %{customdata[0]} – " + opp + " %{customdata[1]}"
                      "<br>Margin %{y:+d}<extra></extra>"))
    if show_tov:
        tm = S.turnover_markers(data.turnovers, data.margins, gid, focus_last)
        for is_focus, name, symbol, color, size in ((False, "Other turnovers", "circle", GRAY, 9),
                                                    (True, f"{focus_last} turnovers", "diamond", FOCUS, 12)):
            sub = tm[tm["is_focus"] == is_focus]
            fig.add_trace(go.Scatter(
                x=sub["minute"], y=sub["margin"], mode="markers", name=name,
                marker=dict(symbol=symbol, color=color, size=size, line=dict(color=BG, width=1.2)),
                customdata=np.column_stack([sub["who"], sub["sub_type"], sub["kind"], sub["pts_off"]]),
                hovertemplate="<b>%{customdata[0]}</b>: %{customdata[1]}<br>%{customdata[2]}"
                              "<br>Opponent scored %{customdata[3]} off it<extra></extra>"))
    stops = S.replay_frames(m, step_seconds=60)
    frames = []
    for t in stops:
        cut = m[m["minute"] <= t]
        last = cut.iloc[-1] if len(cut) else m.iloc[0]
        frames.append(go.Frame(
            name=f"{t:.1f}", traces=[1],
            data=[go.Scatter(x=cut["minute"], y=cut["margin"], customdata=custom[: len(cut)])],
            layout=dict(title=dict(text=f"Minute {t:.0f} · LAC {int(last['team_score'])} – "
                                        f"{opp} {int(last['opp_score'])}"))))
    fig.frames = frames
    fig.update_layout(
        title=dict(text=f"Final · LAC {int(m['team_score'].iloc[-1])} – {opp} {int(m['opp_score'].iloc[-1])}",
                   x=0, font=dict(size=15)),
        updatemenus=[dict(type="buttons", showactive=False, x=1, xanchor="right", y=1.18, direction="left", buttons=[
            dict(label="▶ Replay", method="animate",
                 args=[None, dict(frame=dict(duration=25, redraw=True), transition=dict(duration=0),
                                  fromcurrent=False, mode="immediate")]),
            dict(label="❚❚", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ])],
    )
    n_ot = int(row["overtime_periods"])
    fig.update_xaxes(range=[0, end + 0.8], tickvals=[0, 12, 24, 36, 48] + [50.5 + 5 * i for i in range(n_ot)],
                     ticktext=["Start", "Q2", "Half", "Q4", "End of Q4"]
                     + [("OT" if i == 0 else f"{i + 1}OT") for i in range(n_ot)])
    lim = max(10, float(m["margin"].abs().max()) + 4)
    show(fig, 430, yaxis=dict(title="Clippers ahead ↑ / behind ↓", range=[-lim, lim]))
    st.caption("Shaded: the biggest run each way. Press ▶ Replay to watch it unfold; hover a turnover for who, "
               "what and what it cost.")
    with st.expander("Inside the runs"):
        for run in (run_against, run_for):
            if run is None:
                continue
            side = f"{opp} run" if run.against else "Clippers run"
            scorers = ", ".join(f"{n} {v}" for n, v in run.opp_scorers) or "nobody"
            st.markdown(f"**{side} {run.label}** ({R.run_window(run)}). On the floor for LAC: "
                        f"{', '.join(run.unit) or '?'}. Clippers turnovers: {', '.join(run.turnovers) or 'none'}; "
                        f"missed shots: {run.misses}; {opp} offensive rebounds: {run.opp_orebs}. "
                        f"{opp} points: {scorers}.")

    # ---- what decided it, and what needed to happen
    if len(factors):
        won = row["final_margin"] > 0
        heading("What decided it" if won else "What needed to happen", "decided")
        st.markdown(f'<div class="takeaway">Compared with their average game this season '
                    f'({signed(factors.attrs["normal_margin"])} a game), here\'s where this one was better or worse.'
                    "</div>", unsafe_allow_html=True)
        fz = factors[factors["factor"] != "residual"]
        better = fz[fz["points"] >= 0.5].sort_values("points", ascending=False)
        worse = fz[fz["points"] <= -0.5].sort_values("points")
        top = float(fz["points"].abs().max() or 1)

        def ledger(rows: pd.DataFrame, color: str, title: str, empty: str) -> str:
            items = "".join(
                f'<div class="ledger-row"><div class="lname">{n}</div><div class="lbar"><span style="width:'
                f'{100 * abs(v) / top:.0f}%;background:{color}"></span></div>'
                f'<div class="lval" style="color:{color}">{signed(v)}</div></div>'
                for n, v in zip(rows["name"], rows["points"]))
            return (f'<div class="ledger"><div class="lhead" style="color:{color}">{title}</div>'
                    + (items or f'<div class="lempty">{empty}</div>') + "</div>")

        normal, final = factors.attrs["normal_margin"], factors.attrs["margin"]
        st.markdown(f'<div class="ledger-sum">A normal night: <b>{signed(normal)}</b> &nbsp;→&nbsp; this game: '
                    f'<b style="color:{GOOD if final > 0 else BAD}">{signed(final, 0)}</b></div>',
                    unsafe_allow_html=True)
        left_, right_ = st.columns(2, gap="large")
        left_.markdown(ledger(better, GOOD, "▲ Better than a normal night", "Nothing stood out."),
                       unsafe_allow_html=True)
        right_.markdown(ledger(worse, BAD, "▼ Worse than a normal night", "Nothing stood out."),
                        unsafe_allow_html=True)
        st.caption("Points each part of the game was worth compared with their average game this season. Better and "
                   "worse add up to the difference between the two margins (plus a little the bookkeeping can't place).")
        if not won and lvs:
            st.markdown("**The fixes, biggest first**")
            answer_cards([W.Verdict(lv.title, f"{lv.points:.0f} pts", "; ".join(lv.suggestions) or "—",
                                    "bad", "decided") for lv in lvs], link=False, big_size="1.5rem")

    # ---- who played
    heading("Who played well", "players")
    if len(lines):
        st.markdown('<div class="takeaway">Every Clipper who played, best night first. The bar is his Game Score '
                    '(a one-number box-score summary); the tick marks 10, a good night.</div>', unsafe_allow_html=True)
        answer_cards([W.Verdict(
            f"{r_.player_name}{' · starter' if r_.starter else ''}", f"{int(r_.pts)} pts",
            f"{int(r_.reb)} reb · {int(r_.ast)} ast · {int(r_.tov)} TO · {int(r_.fgm)}/{int(r_.fga)} FG in "
            f"{r_.minutes:.0f} min · {signed(r_.plus_minus, 0)} on the floor. Game Score {r_.game_score:.1f}.",
            "good" if r_.game_score >= 10 else "bad" if r_.game_score < 3 else "info", "players",
            ("share", max(r_.game_score, 0) / 25, 10 / 25), D.headshot_url(r_.player_id))
            for r_ in lines.itertuples()], link=False, big_size="1.5rem")

    # ---- who guarded whom
    mu = R.matchups_for_game(data.matchups, gid, "LAC", opp_lines) if len(data.matchups) else pd.DataFrame()
    heading("Who guarded whom", "matchups")
    if mu.empty:
        st.info("Player tracking for this game isn't published yet (it usually appears a few hours after the game).")
    else:
        st.markdown(f'<div class="takeaway">{opp}\'s top scorers, and what they scored against each Clippers '
                    "defender.</div>", unsafe_allow_html=True)
        opp_id = int(g_all.loc[gid, "opponent_id"])
        tc = D.team_color(opp_id)
        cards = []
        for (scorer, total), grp in mu.groupby(["scorer", "scorer_pts"], sort=False):
            pid = opp_lines.loc[opp_lines["player_name"] == scorer, "player_id"]
            other = int(total - grp["pts"].sum())
            rows = "".join(
                f'<div class="sc-row"><img src="{face(d) or ""}" alt=""><div class="who">Guarded by '
                f'<b>{d.split(" ", 1)[-1]}</b><br><span style="opacity:.65">{int(m_)}/{int(a_)} shooting, '
                f'{pp:.0f} possessions</span></div><div class="pts" style="color:{BAD if p_ >= 10 else INK_MARK}">'
                f'{int(p_)}</div></div>'
                for d, p_, m_, a_, pp in zip(grp["defender"], grp["pts"], grp["fgm"], grp["fga"], grp["poss"]))
            cards.append(
                f'<div class="scard"><div class="sc-top" style="--tc:{tc};--td:{D.shade(tc)}">'
                f'<img class="logo" src="{D.logo_url(opp_id, dark=not D.is_light(tc))}" alt="">'
                + (f'<img class="face" src="{D.headshot_url(int(pid.iloc[0]))}" alt="">' if len(pid) else "")
                + f'<div class="sc-name">{scorer}</div></div><div class="sc-pts">{int(total)}<span>pts</span></div>'
                + rows
                + (f'<div class="sc-foot">The other {other}: other defenders, transition or the line.</div>'
                   if other > 0 else '<div class="sc-foot"></div>') + "</div>")
        st.markdown(f'<div class="scards" style="--cols:{min(len(cards), 3)}">{"".join(cards)}</div>',
                    unsafe_allow_html=True)
        st.caption("Points scored while each Clipper was his main defender (player tracking). Fast breaks and "
                   "scrambles with no clear defender aren't counted.")

    # ---- five-man units
    if len(game_units):
        heading("Which lineups worked", "units")
        ids_by_name = {v: k for k, v in names.items()}
        gu = game_units[game_units["seconds"] >= 120].sort_values("plus_minus", ascending=False)
        show_units = pd.concat([gu.head(3), gu.tail(3)]).drop_duplicates(subset=["team"])
        rows_html = []
        for u in show_units.itertuples():
            faces = "".join(f'<img src="{D.headshot_url(ids_by_name[n])}" alt="{n}" title="{n}">' for n in u.unit
                            if n in ids_by_name)
            color = GOOD if u.plus_minus > 0 else BAD if u.plus_minus < 0 else GRAY
            rows_html.append(
                f'<div class="lineup"><div class="faces">{faces}</div>'
                f'<div class="who">{", ".join(n.split(" ", 1)[-1] for n in u.unit)}</div>'
                f'<div class="min">{u.minutes:.0f} min</div>'
                f'<div class="pm" style="color:{color}">{signed(u.plus_minus, 0)}</div></div>')
        st.markdown('<div class="takeaway">The best and worst five-man units (2+ minutes together): points scored '
                    "minus allowed while they shared the floor.</div>" + "".join(rows_html), unsafe_allow_html=True)
        with st.expander("Every unit, as a chart"):
            gu2 = gu.sort_values("plus_minus")
            fig = go.Figure(go.Bar(
                y=[", ".join(n.split(" ", 1)[-1] for n in u_) for u_ in gu2["unit"]], x=gu2["plus_minus"],
                orientation="h", marker_color=[GOOD if v > 0 else BAD if v < 0 else MUTED_BAR for v in gu2["plus_minus"]],
                text=[f"{signed(v, 0)} in {mm:.0f} min" for v, mm in zip(gu2["plus_minus"], gu2["minutes"])],
                textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:+d}<extra></extra>"))
            lim = float(gu2["plus_minus"].abs().max() or 1) * 1.6
            fig.add_vline(x=0, line=dict(color=INK_MARK, width=1))
            fig.update_xaxes(range=[-lim, lim], title="Points scored minus allowed with that five on the floor")
            show(fig, 90 + 32 * len(gu2), showlegend=False)

    # ---- the details
    with st.expander("What changed after halftime in this game"):
        a = att.set_index("factor")
        fig = go.Figure(go.Bar(
            x=a["points"], y=[S.FACTOR_PLAIN[f][0] for f in a.index], orientation="h",
            marker_color=[GRAY if f == "residual" else GOOD if v >= 0 else BAD for f, v in a["points"].items()],
            customdata=[S.FACTOR_PLAIN[f][1] for f in a.index],
            hovertemplate="<b>%{y}</b>: %{x:+.1f} pts<br>%{customdata}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.add_vline(x=0, line=dict(color=GRAY, width=1))
        show(fig, 300, xaxis_title=f"Points vs the first half (total {signed(a['margin_change'].iloc[0], 0)})")
    with st.expander("Box score"):
        cols = ["player_name", "starter", "minutes", "pts", "reb", "ast", "stl", "blk", "tov", "pf", "fgm", "fga",
                "fg3m", "fg3a", "ftm", "fta", "plus_minus", "game_score"]
        st.dataframe(with_photos(lines[cols].round(1)), hide_index=True, width="stretch", column_config=PHOTO_COL)
        if len(game_def):
            st.markdown("**Each Clippers defender's night (tracking)**")
            st.dataframe(with_photos(game_def[["def_player", "main_assignment", "poss", "pts", "fgm", "fga"]].round(1),
                                     "def_player"), hide_index=True, width="stretch", column_config=PHOTO_COL)
    with st.expander("Before vs after halftime: every stat (table)"):
        st.dataframe(D.game_half_table(data.game_halves, gid), hide_index=True, width="stretch",
                     column_config={"stat": st.column_config.TextColumn("", width="medium")})
    with st.expander("Every Clippers turnover in this game"):
        st.dataframe(with_photos(D.game_turnovers(data.turnovers, gid), "player"), hide_index=True, width="stretch",
                     column_config=PHOTO_COL)
    fill_toc(nav_)

# ================================================================= players

def player_cards(ph: pd.DataFrame, season: str) -> None:
    """The rotation as trading cards: photo, jersey number, and two numbers each, busiest first."""
    h2 = ph[ph["split"] == "H2"].sort_values("play_share", ascending=False)
    roster = (data.roster.set_index("player_id") if len(data.roster) and season == season_now
              else pd.DataFrame(columns=["number", "position"]))
    cards = []
    for r in h2.itertuples():
        info = roster.loc[r.player_id] if r.player_id in roster.index else None
        num = f"#{info['number']}" if info is not None and str(info["number"]).strip() else ""
        pos = info["position"] if info is not None else ""
        every = f"1 in {round(1 / r.tov_pct)}" if r.tov_pct else "none"
        cards.append(
            f'<div class="pcard" title="{r.player_name}"><div class="pc-top"><span class="pc-num">{num}</span>'
            f'<span class="pc-pos">{pos}</span><img src="{D.headshot_url(int(r.player_id))}" alt="" loading="lazy">'
            f'</div><div class="pc-body"><div class="pc-name">{r.player_name}</div><div class="pc-stats">'
            f'<div><b>{r.play_share:.0%}</b><span>of the plays after halftime</span></div>'
            f'<div><b>{every}</b><span>plays ends in a turnover</span></div></div></div></div>')
    st.markdown(f'<div class="pcards">{"".join(cards)}</div>', unsafe_allow_html=True)


def page_players() -> None:
    page_top("Players", "Who loses the ball after halftime. Click a card or a player to dig in.")
    nav_ = toc_slot("players")
    band = st.container(key="band_navy_players")  # the season picker and the key cards share the navy band
    season_p = band.columns([1, 5])[0].selectbox("Season", data.seasons, index=0, key="player_season")
    ph = D.player_halves(data.player_stats, season_p)
    if ph.empty:
        band.info("No players with enough minutes in both halves yet.")
    else:
        with band:
            answer_cards(S.player_cards(ph, focus))
            player_cards(ph, season_p)
        h2 = ph[ph["split"] == "H2"].sort_values("tov_share")
        section("Who's losing the ball after halftime?", "who",
                "The further above the line, the more he gives it away for how much he has it.")
        names = sorted(ph["player_name"].unique())
        st.session_state.setdefault("player_pick", focus if focus in names else names[0])
        if st.session_state["player_pick"] not in names:
            st.session_state["player_pick"] = names[0]
        xmax, ymax = float(h2["play_share"].max()) * 1.15, float(h2["tov_share"].max()) * 1.15
        lim = min(xmax, ymax)
        named = set(h2.nlargest(4, "tov_share")["player_name"]) | set(h2.nlargest(2, "play_share")["player_name"])
        fig = go.Figure()
        # above the diagonal = more of the turnovers than of the plays
        fig.add_shape(type="path", path=f"M 0 0 L {lim} {lim} L {lim} {ymax} L 0 {ymax} Z",
                      fillcolor="rgba(217,89,38,0.08)",
                      line_width=0, layer="below")
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(color=INK_MARK, width=1.5, dash="dot"))
        fig.add_annotation(x=lim * 0.97, y=lim * 0.97, text="his fair share", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=12, color=INK_MARK), textangle=-38)
        fig.add_annotation(x=xmax * 0.03, y=ymax * 0.96, text="<b>More than his share of turnovers</b>",
                           showarrow=False, xanchor="left", font=dict(size=13, color=BAD))
        fig.add_trace(go.Scatter(
            x=h2["play_share"], y=h2["tov_share"], mode="markers+text", customdata=h2["player_name"],
            text=[n_.split(" ", 1)[-1] if n_ in named else "" for n_ in h2["player_name"]],
            textposition="bottom center",
            textfont=dict(size=12, color=INK_MARK), marker=dict(size=40, opacity=0), showlegend=False,
            hovertemplate="<b>%{customdata}</b><br>%{x:.0%} of the plays after halftime<br>%{y:.0%} of the "
                          "turnovers<extra></extra>"))
        for x_, y_, n_ in zip(h2["play_share"], h2["tov_share"], h2["player_name"]):
            if face(n_):
                fig.add_layout_image(source=face(n_), xref="x", yref="y", x=x_, y=y_, xanchor="center",
                                     yanchor="middle", sizex=xmax * 0.07, sizey=ymax * 0.11, sizing="contain",
                                     layer="above")
        fig.update_xaxes(range=[0, xmax], tickformat=".0%", title="Share of the team's plays after halftime")
        fig.update_yaxes(range=[0, ymax], tickformat=".0%", title="Share of the team's turnovers after halftime")
        event = show(fig, 520, key=f"who_{season_p}", select=True)
        name = clicked(event, "customdata", f"who_click_{season_p}")
        if name in names:
            st.session_state["player_pick"] = name
        st.caption("Each face is a player, placed by how much of the offense he runs (across) and how many of the "
                   "turnovers he makes (up). Above the dotted line = more than his share. Click a face to see his "
                   "games.")

        section("Who gets sloppier after halftime?", "sloppier",
                "How often one of his plays ends in a turnover: before halftime (gray dot) and after it. "
                "Sloppiest after halftime first.")
        wl = ph.pivot(index="player_name", columns="split", values="tov_pct").sort_values("H2", ascending=False)
        every = lambda v: f"1 in {round(1 / v)}" if v and v > 0 else "none"  # noqa: E731
        hi_ = float(wl.to_numpy().max()) * 1.08
        rows_ = []
        for n_, r_ in wl.iterrows():
            d_ = r_["H2"] - r_["H1"]
            color, verdict = ((BAD, "Sloppier") if d_ >= 0.02 else (GOOD, "Safer") if d_ <= -0.02
                              else (MUTED_BAR, "About the same"))
            rows_.append(
                f'<div class="slop-row"><img src="{face(n_) or ""}" alt=""><div class="nm">{n_.split(" ", 1)[-1]}'
                f'<span class="chip sm" style="background:{color}">{verdict}</span></div>'
                f'<div class="vals"><span class="b">{every(r_["H1"])}</span> → '
                f'<span style="color:{color}">{every(r_["H2"])}</span></div>'
                + track(r_["H1"], r_["H2"], 0, hi_, color) + "</div>")
        st.markdown(f'<div class="slop">{"".join(rows_)}</div>', unsafe_allow_html=True)
        st.caption("Plays = shots, free-throw trips and turnovers. Sloppier or safer = a change of 2+ percentage "
                   "points in how often his plays end in a turnover. Players with 40+ minutes in each half.")

        section("One player, game by game", "one")
        player = st.selectbox("Player", names, key="player_pick")
        t = ph[ph["player_name"] == player].set_index("split")
        pid_ = int(t["player_id"].iloc[0])
        info = data.roster.set_index("player_id").loc[pid_] if len(data.roster) and pid_ in set(
            data.roster["player_id"]) and season_p == season_now else None
        num = f"#{info['number']}" if info is not None and str(info["number"]).strip() else ""
        d = 100 * (t.loc["H2", "tov_pct"] - t.loc["H1", "tov_pct"])
        st.markdown(
            f'<div class="hero-card"><div class="hc-photo"><span class="pc-num">{num}</span>'
            f'<img src="{D.headshot_url(pid_)}" alt=""></div><div class="hc-body">'
            f'<div class="kicker">{info["position"] + " · " if info is not None else ""}after halftime, {season_p}</div>'
            f'<div class="hc-name">{player}</div><div class="hc-stats">'
            f'<div><b>{t.loc["H1", "tov_pct"]:.0%}</b><span>of his plays end in a turnover before halftime</span></div>'
            f'<div><b class="{"hc-up" if d > 0 else "hc-down"}">{t.loc["H2", "tov_pct"]:.0%}</b><span>after halftime '
            f'({"+" if d > 0 else "−"}{abs(d):.1f} pts)</span></div>'
            f'<div><b>{t.loc["H2", "tov_share"]:.0%}</b><span>of the team\'s turnovers after halftime, on '
            f'{t.loc["H2", "play_share"]:.0%} of its plays</span></div>'
            f'<div><b>{t.loc["H2", "live_share"]:.0%}</b><span>of them steals (live balls that lead to fast breaks)'
            f'</span></div></div></div></div>', unsafe_allow_html=True)
        g = D.player_games(data.player_stats, data.game_leads, season_p, player)
        gx = pd.to_datetime(g["game_date"]).dt.strftime("%b %d").str.replace(" 0", " ") + " " + g[
            "opponent_abbreviation"]
        fig = go.Figure([
            go.Bar(x=gx, y=g["tov_h1"], name="1st half", marker_color=H1C,
                   hovertemplate="%{x}: %{y} turnovers<extra>1st half</extra>"),
            go.Bar(x=gx, y=g["tov_h2"], name="2nd half", marker_color=H2C,
                   hovertemplate="%{x}: %{y} turnovers<extra>2nd half</extra>"),
        ])
        fig.update_layout(barmode="group", bargap=0.3)
        show(fig, 300, yaxis_title="Turnovers")
        with st.expander("Game-by-game table"):
            st.dataframe(g.drop(columns=["game_id"]), hide_index=True, width="stretch")
    fill_toc(nav_)
    next_step(PAGE_LEAGUE, "Is this normal? How every team changes after halftime.")

# ================================================================== league

def page_league() -> None:
    page_top("Vs. the league", "Does every team get worse after halftime? No. Most teams barely change.")
    stat_names = {"net_rtg": "Overall (net rating)", "efg_pct": "Shooting accuracy",
                  "opp_efg_pct": "Opponents' shooting", "tov_pct": "Turnovers",
                  "opp_ft_rate": "Fouling (opponent free throws)", "opp_oreb_pct": "Second chances given up"}
    band = st.container(key="band_navy_league")  # the season picker and the key cards share the navy band
    season_l = band.columns([1, 5])[0].selectbox("Season", data.seasons, index=0, key="league_season")
    cr = D.clippers_ranks(data.ranks, season_l)
    with band:
        answer_cards(S.league_cards(cr))
    section("How every team changes after halftime", "teams",
            "Pick a stat. Each logo is a team: <b>left = worse after halftime, right = better</b>; higher = "
            "better in first halves. Clippers ringed in orange.")
    metric = st.pills("Stat", list(stat_names), default="net_rtg", format_func=stat_names.get, key="league_metric")
    metric = metric or "net_rtg"
    flip = D.better_direction(metric) == "lower is better"
    sign = -1 if flip else 1  # so right and up always mean better
    ch = D.league_view(data.ranks, season_l, metric, "change").set_index("team_name")
    h1 = D.league_view(data.ranks, season_l, metric, "H1").set_index("team_name")
    ids = data.ranks[data.ranks["season"] == season_l].drop_duplicates("team_name").set_index("team_name")["team_id"]
    q = ch.join(h1[["value", "league_mean"]].add_suffix("_h1")).assign(team_id=ids)
    q["x"] = sign * q["value"]
    q["y"] = sign * (q["value_h1"] - q["league_mean_h1"])
    pct = metric not in ("net_rtg", "tov_pct")
    scale = 100 if pct else 1
    fmt = (lambda v: f"{signed(v * 100)} pts") if pct else (lambda v: signed(v))
    q[["x", "y"]] *= scale
    xr = float(q["x"].abs().max()) * 1.18 or 1
    yr = float(q["y"].abs().max()) * 1.22 or 1
    fig = go.Figure()
    corner = dict(showarrow=False, font=dict(size=13, color=INK_MARK), opacity=0.75)
    for x_, y_, xa, ya, text in ((-xr, yr, "left", "top", "Strong starts, fade late"),
                                 (xr, yr, "right", "top", "Strong, and get stronger"),
                                 (-xr, -yr, "left", "bottom", "Slow starts, get worse"),
                                 (xr, -yr, "right", "bottom", "Slow starts, finish strong")):
        fig.add_annotation(x=x_, y=y_, xanchor=xa, yanchor=ya, text=f"<b>{text}</b>", **corner)
    fig.add_shape(type="rect", x0=-xr, x1=0, y0=-yr, y1=yr, fillcolor="rgba(217,89,38,0.06)", line_width=0,
                  layer="below")
    fig.add_shape(type="rect", x0=0, x1=xr, y0=-yr, y1=yr, fillcolor="rgba(57,135,229,0.06)", line_width=0,
                  layer="below")
    fig.add_vline(x=0, line=dict(color=GRAY, width=1))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1, dash="dot"))
    lac = q[q["is_team"]]
    fig.add_trace(go.Scatter(x=lac["x"], y=lac["y"], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=46, color="rgba(0,0,0,0)", line=dict(color=FOCUS, width=3))))
    fig.add_trace(go.Scatter(
        x=q["x"], y=q["y"], mode="markers", showlegend=False, marker=dict(size=30, opacity=0),
        customdata=np.column_stack([q.index, [fmt(v) for v in q["value"]], q["rank"],
                                    [fmt(v) for v in q["value_h1"]], [fmt(v) for v in q["league_mean_h1"]]]),
        hovertemplate="<b>%{customdata[0]}</b><br>Change after halftime: %{customdata[1]} "
                      "(%{customdata[2]} of 30)<br>First halves: %{customdata[3]} (league %{customdata[4]})"
                      "<extra></extra>"))
    for name, r_ in q.iterrows():
        fig.add_layout_image(source=D.logo_url(int(r_["team_id"]), dark=False), xref="x", yref="y", x=r_["x"], y=r_["y"],
                             xanchor="center", yanchor="middle", sizex=xr * 0.11, sizey=yr * 0.11,
                             sizing="contain", layer="above")
    fig.update_xaxes(range=[-xr, xr], showticklabels=False, showgrid=False,
                     title="← worse after halftime            better after halftime →")
    fig.update_yaxes(range=[-yr, yr], showticklabels=False, showgrid=False,
                     title="← worse first halves      better first halves →")
    row = q[q["is_team"]].iloc[0]
    left, right = st.columns([2, 1], gap="large")
    with left:
        show(fig, 620)
        st.caption("Center lines: no change at halftime (solid), league-average first half (dotted). Hover a logo "
                   "for the numbers.")
    with right:
        st.metric(f"Clippers: {stat_names[metric]}", f"{ordinal(int(row['rank']))} of 30", border=True,
                  help="Rank of the change from first half to second half. 1st = improves the most.")
        st.metric("Clippers' change", fmt(row["value"]), border=True,
                  help=f"League average change: {fmt(row['league_mean'])}.")
        with st.expander("Every team, ranked"):
            ranked = q.sort_values("rank").reset_index()
            st.dataframe(pd.DataFrame({"logo": [D.logo_url(int(t), dark=False) for t in ranked["team_id"]],
                                       "Team": ranked["team_name"], "Rank": ranked["rank"],
                                       "Change": [fmt(v) for v in ranked["value"]]}),
                         hide_index=True, width="stretch",
                         column_config={"logo": st.column_config.ImageColumn("", width="small")})

    section("Where the Clippers rank on everything", "ranks",
            "Gray dot = their league rank in first halves, orange = second halves. Right = better.")
    stats = [x for x in cr.index if x != "Pace (possessions)"]
    fig = go.Figure()
    for name in stats:
        a, b = cr.loc[name, cr.columns[0]], cr.loc[name, cr.columns[1]]
        fig.add_trace(go.Scatter(x=[a, b], y=[name, name], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=BAD if b > a + 2 else GOOD if b < a - 2 else LIGHT, width=3)))
    for col, color, label in ((cr.columns[0], H1C, "1st half"), (cr.columns[1], H2C, "2nd half")):
        # name the two dots on the top row, right above them (the legend stays too)
        fig.add_annotation(x=cr.loc[stats[0], col], y=stats[0], text=label, showarrow=False, yshift=17,
                           font=dict(size=12, color=INK_MARK))
        fig.add_trace(go.Scatter(
            x=cr.loc[stats, col], y=stats, mode="markers", name=label,
            marker=dict(color=color, size=13, line=dict(color=BG, width=1.5)),
            text=[ordinal(int(v)) for v in cr.loc[stats, col]],
            hovertemplate="%{y}: %{text} of 30<extra>" + label + "</extra>"))
    fig.update_xaxes(range=[30.5, 0.5], tickvals=[30, 20, 10, 1], ticktext=["30th", "20th", "10th", "1st"],
                     title="League rank (1st = best)")
    fig.update_yaxes(autorange="reversed")
    show(fig, 90 + 38 * len(stats))
    st.caption("Orange line = they drop 3+ places after halftime; blue = they climb.")
    with st.expander("The ranks as a table"):
        st.dataframe(cr.map(ordinal), width="stretch", height=38 + 35 * len(cr))
    next_step(PAGE_WHATNOW, "So what has to change? What the numbers point to.")


# ================================================================== next game

TAPE = [  # (label, column, higher is better)
    ("Overall (net rating)", "net", True), ("Scoring (per 100)", "ortg", True), ("Defense (allowed per 100)", "drtg", False),
    ("Shooting accuracy", "efg", True), ("Opponents' shooting", "opp_efg", False),
    ("Ball security", "tov_pct", False), ("Forcing turnovers", "opp_tov_pct", True),
    ("Offensive rebounding", "oreb_pct", True), ("Defensive rebounding", "opp_oreb_pct", False),
    ("Getting to the line", "ft_rate", True),
]


def _faces(lean: list[tuple[str, int, str]]) -> str:
    """Small round headshots with name and the stat behind each pick."""
    return "".join(
        f'<div class="lean"><img src="{D.headshot_url(pid)}" alt=""><div><b>{name.split(" ", 1)[-1]}</b>'
        f'<br><span>{stat}</span></div></div>' for name, pid, stat in lean)


def _game_plan(game: pd.Series, pv, teams: pd.DataFrame, shooting: float | None) -> None:
    """An upcoming game: the odds, a checklist of keys that updates them, matchups, and the tale of the tape."""
    tid = settings.team_id
    tip = game["tipoff_utc"].tz_convert("America/Los_Angeles")
    opp_id = int(game["away_id"] if pv.home else game["home_id"])
    st.markdown(f'<div class="kicker">{tip:%a %b} {tip.day} · {tip:%I:%M %p}'.replace(" 0", " ") + " PT · "
                f'{pv.arena} · {"home" if pv.home else "away"}</div>'
                f'<div class="hero">Clippers {"vs" if pv.home else "at"} {pv.opponent_name}</div>',
                unsafe_allow_html=True)
    lac, opp = pv.lac, pv.opp
    players = GP.player_table(data.player_box, season_now, "LAC")
    misc = data.team_misc[data.team_misc["season"] == season_now] if len(data.team_misc) else pd.DataFrame(
        columns=["team_id", "opp_pts_paint"])
    keys = GP.game_plan(pv, teams, misc, players, shooting)

    # ---- the odds, live with the keys picked below
    odds = st.empty()
    h2h = P.head_to_head(data.games, season_now, pv.opponent)
    tiles = [
        W.Verdict("Clippers this season", signed(lac["net"]),
                  f"Points per 100 possessions, scored minus allowed: {P._ord(lac['net_rank'])} of 30.",
                  "bad" if lac["net"] < 0 else "good", "plan", ("rank", int(lac["net_rank"]), 30), D.logo_url(tid)),
        W.Verdict(f"{pv.opponent} this season", signed(opp["net"]), f"{P._ord(opp['net_rank'])} of 30.",
                  "bad" if opp["net"] > lac["net"] else "good", "plan", ("rank", int(opp["net_rank"]), 30),
                  D.logo_url(opp_id)),
    ]
    if len(h2h):
        wins = int((h2h["result"] == "W").sum())
        tiles.append(W.Verdict("Earlier this season", f"{wins}-{len(h2h) - wins}",
                               "; ".join(f"{r.result} {int(r.team_pts)}-{int(r.opp_pts)}" for r in h2h.itertuples()),
                               "info", "plan"))

    section(f"How to beat the {pv.opponent_name.split(' ')[-1]}", "plan",
            "The keys to this game from both teams' season numbers. Switch them on and off: the odds above "
            "update.")
    chosen = []
    if keys:
        for j, k in enumerate(keys):
            with st.container(border=True, key=f"key_{k.kind}"):
                c1, c2, c3 = st.columns([1.15, 2.4, 1.45], vertical_alignment="center", gap="medium")
                on = c1.toggle(k.title, value=False, key=f"plan_{game['game_id']}_{k.kind}")
                c1.markdown(f'<div class="kval">+{k.points:.1f} pts a game<br><span>{k.chance:.0%} to win with '
                            f"this alone</span></div>", unsafe_allow_html=True)
                c2.markdown(f'<div class="kwhy">{k.why}</div>', unsafe_allow_html=True)
                c3.markdown(_faces(k.lean_on), unsafe_allow_html=True)
                if on:
                    chosen.append(k)
        st.caption("Keys overlap a little (a turnover avoided can also be a shot that misses), so several together "
                   "are a best case.")
    else:
        st.write("Nothing in either team's numbers points to a specific key: it's about playing their usual game.")
    pts = sum(k.points for k in chosen)
    chance = P.win_chance(pv.margin + pts)
    with odds.container():
        picture(G.odds_bar("LAC", pv.opponent, chance, pv.margin + pts, team_logo=D.logo_url(tid, dark=False),
                           opp_logo=D.logo_url(opp_id, dark=False)), max_width="900px")
        st.markdown(f'<div class="odds-note">Today <b>{pv.chance:.0%}</b>'
                    + (f' → with {len(chosen)} key{"s" if len(chosen) != 1 else ""} below <b>{chance:.0%}</b>'
                       if chosen else " · switch on keys below to see what they're worth") + "</div>",
                    unsafe_allow_html=True)
        answer_cards(tiles, link=False)

    # ---- matchups
    defenders = W.defenders(data.matchups, data.hustle, data.bio, season_now, "LAC") if len(data.matchups) else None
    watch = P.who_to_watch(data.player_season, data.bio, season_now, pv.opponent,
                           defenders.table if defenders is not None else None)
    longs = P.long_rotation(data.player_season, data.bio, season_now, pv.opponent)
    if len(watch):
        section("Matchups: who guards their scorers", "watch",
                f"{pv.opponent}'s top scorers this season, and the Clipper best suited to guard each (closest in size, "
                "fewest points allowed per possession from player tracking).")
        ids = data.player_season.set_index("player_name")["player_id"].to_dict()
        answer_cards([W.Verdict(
            r.player_name, f"{r.pts:.1f} ppg",
            (f"{G._ft_in(r.height_in)}, {r.weight_lb:.0f} lb. " if pd.notna(r.height_in) and pd.notna(r.weight_lb)
             else "") + f"{r.reb:.1f} reb, {r.ast:.1f} ast in {r.min:.0f} min. "
            + (f"Guard him with: {r.guard_with.rstrip('.')}." if r.guard_with else ""), "warn", "watch",
            image=D.headshot_url(ids[r.player_name]) if r.player_name in ids else None)
            for r in watch.itertuples()], link=False)
        if len(longs) >= P.LONG_ROTATION:
            st.warning(f"{pv.opponent} plays {len(longs)} players between 6'6\" and 6'10\" in its top eight "
                       f"({', '.join(longs)}). {focus_last} turns it over about twice as often against defenders "
                       "that size: get him the ball on the move or off screens.", icon=":material/height:")
        elif longs:
            st.caption(f"{pv.opponent}'s 6'6\"-6'10\" rotation players: {', '.join(longs)}.")

    # ---- tale of the tape
    section("Tale of the tape", "tape", "League rank on each part of the game (1st = best). Further right is better.")
    fig = go.Figure()
    labels = [x[0] for x in TAPE]
    for name, row_, color in (("LAC", lac, H1C), (pv.opponent, opp, FOCUS)):
        fig.add_trace(go.Scatter(
            x=[row_[f"{c}_rank"] for _, c, _ in TAPE], y=labels, mode="markers+text", name=name,
            text=[P._ord(row_[f"{c}_rank"]) for _, c, _ in TAPE], textposition="top center",
            marker=dict(color=color, size=14, line=dict(color=BG, width=1.5)),
            hovertemplate="%{y}: %{text}<extra>" + name + "</extra>"))
    for lab, c, _ in TAPE:
        fig.add_shape(type="line", x0=lac[f"{c}_rank"], x1=opp[f"{c}_rank"], y0=lab, y1=lab,
                      line=dict(color=LIGHT, width=3), layer="below")
    fig.update_xaxes(range=[31, 0], tickvals=[30, 20, 10, 1], ticktext=["30th", "20th", "10th", "1st"],
                     title="League rank this season")
    fig.update_yaxes(autorange="reversed")
    show(fig, 90 + 40 * len(TAPE))

    with st.expander("How the odds and the game plan are worked out"):
        st.markdown(
            f"- **Team strength** is points scored minus allowed per 100 possessions this season, blended with last "
            f"season's while the sample is small: (games x this season + {P.PRIOR_GAMES} x last season) / "
            f"(games + {P.PRIOR_GAMES}).\n"
            f"- **Expected margin** = the difference in strength x expected possessions / 100, plus "
            f"{P.HOME_POINTS:.0f} points for the home team.\n"
            f"- **Win chance** treats the final margin as landing within about {P.MARGIN_SD:.1f} points of that, the "
            "usual spread of an NBA game.\n"
            "- **Keys** only appear when the numbers point to them; each rule (and how its points are counted) is "
            "written at the top of src/courtside/analysis/gameplan.py.\n"
            "- It doesn't know about injuries, rest or travel, and isn't betting advice. Models this simple pick "
            "roughly two in three NBA games correctly.")
        st.dataframe(teams[["team_name", "gp", "net", "prior_net", "blend_net", "pace"]].sort_values(
            "blend_net", ascending=False).round(1), hide_index=True, width="stretch")


PAGE_ANSWER = st.Page(page_answer, title="Blown leads", icon=":material/lightbulb:", url_path="answer")
PAGE_WHATNOW = st.Page(page_whatnow, title="What now?", icon=":material/checklist:", url_path="what-now")
PAGE_REPLAY = st.Page(page_replay, title="Games", url_path="home",
                      default=True)
PAGE_PLAYERS = st.Page(page_players, title="Players", icon=":material/groups:", url_path="players")
PAGE_LEAGUE = st.Page(page_league, title="Vs. the league", icon=":material/leaderboard:", url_path="league")
# Story order: the tracker, the answer, the evidence (players, the league), then what to do about it.
PAGES = {"replay": PAGE_REPLAY, "answer": PAGE_ANSWER, "players": PAGE_PLAYERS, "league": PAGE_LEAGUE,
         "whatnow": PAGE_WHATNOW}
# The default page lives at "/"; this hidden twin makes a shared /games link work too.
PAGE_GAMES_LINK = st.Page(page_replay, title="Games", url_path="games", visibility="hidden")
nav = st.navigation(list(PAGES.values()) + [PAGE_GAMES_LINK], position="top")
behind_notice()
# Tests run a specific page by setting this in session state (AppTest can't change the URL).
forced = st.session_state.get("_dashboard_page")
if forced in PAGES:
    {"answer": page_answer, "whatnow": page_whatnow, "replay": page_replay, "players": page_players,
     "league": page_league}[forced]()
else:
    nav.run()
