"""The answer rules as pure Python: one copy, run by the API server and by the browser demo (Pyodide).

Nothing here imports a model library, a database client or a web framework; `core.answer` yields effects (work it
cannot do itself) and a driver answers them."""
