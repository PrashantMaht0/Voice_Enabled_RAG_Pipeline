---
source: extracted-from-code
extracted_from: app.py (gr.themes.Soft() plus custom_css), the local V-RAG dashboard
tokens: demo/src/style.css (:root custom properties, light and dark)
---

# Live demo design direction (spec 0009)

The demo should feel like the same product as your local dashboard, moved into a browser tab.

## Character
- Calm, technical, trustworthy. A light grey canvas with white, softly rounded panels (Gradio Soft), an indigo
  primary action, and two dark islands borrowed from the dashboard: slate metric cards (`#1e293b`, value
  `#38bdf8`, label `#94a3b8`) and a GitHub dark monospace trace (`#0d1117` on `#58a6ff`).
- Numbers and citations are the hero: every answer shows where it came from, as chips naming document and page.

## Composition
- Header band: the `V-RAG` wordmark, the tagline ("Ask real documents by text or voice. Answers are checked against
  them and cited."), and a status line on the right (model state, "runs on your GPU").
- A model card under the header until the models are loaded: the model picker with download sizes, the Load button,
  and one progress row per part (answer model, search models, Python core).
- Tabs, as on the dashboard: Ask, Documents, Trace, Results.
  - Ask: a wide chat column (suggested question chips above an empty chat, the conversation, the question box with
    Ask and Stop), and a side column with the last answer's sources.
  - Documents: one card per sample document (title, publisher, license, source link, pages).
  - Trace: the recent questions, and the selected one's steps in the dark monospace panel with timings.
  - Results: the slate metric cards and the measured table.
- Footer: privacy line (what runs where) and links to the repository and licenses.

## Components
- Panels: white, 12px radius, 1px `--border`, a soft shadow. Primary button indigo, secondary buttons outlined.
- Chips: rounded, small, for suggested questions (clickable) and citations (document · page · kind).
- Progress rows: label, bar, percent and the current file.
- States: every panel has an empty, loading and error state written as real copy, never blank.

## Build mandate
- Light and dark themes (prefers-color-scheme), phone width stacks columns, controls at least 44px tall there.
- Semantic HTML, visible focus, labels on every control, live region for streamed answers.
- All document text is set with textContent, never innerHTML.
