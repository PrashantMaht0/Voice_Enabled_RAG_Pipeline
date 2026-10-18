---
source: extracted-from-code
extracted_from: app.py (gr.themes.Soft() plus custom_css), the local V-RAG dashboard
tokens: demo/src/style.css (:root custom properties, light and dark)
---

# Live demo design direction

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
- Tabs, exactly the dashboard's three: Ask, Documents, Observability (changed 2026-10-16).
  - Ask: a wide chat column (suggested question chips, the conversation, the question box with Ask, Stop and
    Clear chat); a side column with the five slate metric cards (Confidence, Grounded, Retrieval, Total, First
    audio) and the retrieved sources; below both, full width and always open, the pipeline trace in the dark
    monospace panel, so the pipeline is visible on the same page.
  - Documents: on the left the sample documents' licenses and sources (upload and remove join here with the PDF
    reader); on the right the corpus table (file, pages, chunks by kind, status).
  - Observability: the window selector and Refresh, the ten slate cards in two rows of five, the recent questions
    table with the selected trace beside it, and the stage table.
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
