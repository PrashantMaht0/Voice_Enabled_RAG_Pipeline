---
title: V-RAG PDF reader
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
short_description: Reads PDF tables and charts for the V-RAG live demo
startup_duration_timeout: 1h
---

# V-RAG PDF reader

The document reader behind the [V-RAG live demo](https://prashantmaht0.github.io/Voice_Enabled_RAG_Pipeline/)
(source: [Voice_Enabled_RAG_Pipeline](https://github.com/PrashantMaht0/Voice_Enabled_RAG_Pipeline)).
It runs the pipeline's own ingestion code on one uploaded document and returns its chunks: text by page, tables read
by a fine tuned LayoutLMv3 (`Prashant-Mahto/layoutlmv3-pubtables-table-words`, CC BY-NC-SA 4.0, non commercial), and
charts described by Qwen2.5 VL 3B. The demo then embeds and searches them in the visitor's own browser.

Limits: PDF, Markdown or text; 10 MB, 5 pages and 4 figures per document (one PDF of that size uses about a minute
and a half of the visitor's daily ZeroGPU allowance); one read at a time; 10 reads per address a
day. Each read uses the visitor's ZeroGPU quota. Uploaded files are deleted right after reading, and the log keeps only
a short hash of the file name, the page count and the timing (Hugging Face's own request logs are outside this app's control).

Deployed from the repository by `.github/workflows/demo_pages.yml`; do not edit files here directly.
