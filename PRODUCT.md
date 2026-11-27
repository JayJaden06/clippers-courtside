# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Two audiences, served equally (confirmed by the owner):

- **Clippers fans following the 2026-27 season.** They come after a game to see what happened and why, or before one to see the odds and who's out. They know basketball, not statistics.
- **Recruiters and hiring managers** looking at the project from the owner's resume or LinkedIn. They skim in a few minutes and want to see rigorous, well-built work: real data, sound methods, a live product that updates itself.

A page has to work for both at once: the fan gets the answer, the recruiter sees how it was reached.

## Product Purpose

Clippers Courtside is an interactive data story and live season tracker for the LA Clippers. It started from one question (why do the Clippers keep losing leads after halftime?) and now follows every game.

A visit succeeds when the visitor (all four confirmed):

- **understands the why:** what the numbers say about the second-half collapses, backed by evidence rather than hot takes;
- **knows the next game:** the odds, who's out, and the keys to the matchup;
- **can relive any game:** what decided it, who played well, who was out;
- **is impressed by the work:** the methods, the testing and the self-updating pipeline are visible enough to start a conversation.

## Positioning

- **The numbers write the words.** Every verdict, headline and card sentence is generated from the data, so the story stays true as the season changes. Nothing on the site is typed by hand.
- **Theories are tested, not asserted.** Popular explanations (Garland's turnovers, fouling, The Wall) are checked against rules for "this is the cause" that were set before looking at the results.
- **It keeps itself current.** The same analysis reruns after every game, and the odds account for the NBA's official injury report.

## Operating Context

- Live at https://clippers-courtside.streamlit.app (Streamlit Community Cloud), redeployed on every push to `main`.
- Data from stats.nba.com via nba_api, and the NBA's injury report PDFs from official.nba.com. stats.nba.com blocks cloud servers, so the refresh runs on the owner's Windows PC: after sign-in, unlock or wake, and daily at 8:30am. It pushes only when something new arrived.
- Five pages, in story order: Games (each game's report or game plan), Blown leads (the answer), Players, Vs. the league, What now?
- The owner reviews and merges every change as its own pull request.

## Capabilities and Constraints

- **Built in Streamlit (Python).** Pages are drawn by Python on each interaction. Custom look comes from one injected stylesheet (`src/courtside/web/style.css`), HTML in markdown, and small scripts through `st.html`. There's no front-end framework, and the page redraws (and replays any load-in) on every page switch or game pick.
- **Data timing:** player tracking for a game appears a few hours after it ends. The injury report for a game is posted the day before. Nothing is live during a game.
- **Small samples early in the season.** Findings are leads to keep checking, and the site says so.
- **Current scope (not confirmed as binding):** designed for desktop and laptop; phones aren't a priority today.
- **Undecided:** whether the site should ever adapt for phones.

## Brand Commitments

- Name: **Clippers Courtside**.
- A fan project, not affiliated with or endorsed by the NBA or the LA Clippers. Data, team logos and player headshots belong to the NBA and its teams and are credited as such.
- **Honest uncertainty (binding):** say when a sample is small or a theory falls short of the bar; never overclaim.
- **Data-generated text (binding):** no hand-typed verdicts or player claims.

## Evidence on Hand

- The live site and its committed data (`data/processed/`), regenerated daily.
- `README.md`: method, findings, repo map. `reports/findings.md`: the current written findings.
- 271 automated tests, including a headless run of every page; CI on every push.
- Screenshots in `docs/images/`.
- **Absent, do not fabricate:** user counts, testimonials, press, or prediction-accuracy claims. Grading the odds against results is listed as future work, not done.

## Product Principles

1. **Answer first, method one step away.** The fan gets the plain answer at a glance; the recruiter can open how it was reached without it cluttering the fan's view.
2. **The numbers write the words.** If the data can't support a sentence, the sentence doesn't appear.
3. **Say how sure we are.** Small samples, failed theories and missing data are stated, not hidden.
4. **Always current.** Every page reflects the latest game without manual edits, and says plainly when data is late.

## Accessibility & Inclusion

No formal standard has been set. Current practice: the blue/orange/gray palette is checked for colorblind readers and 3:1 contrast on the light background, and motion respects the system's reduced-motion setting. The owner has not made these binding.
