# Ticket extract

Put your extract here as a **CSV or XLSX** file (the first one found is used). The app reads it at startup and
reloads it when the file changes.

Headers are matched ignoring accents, case and spacing. Required: `Ouvert le`, `Description courte`, `Description`.
The names of the columns and of the labels the alert rules look for are defined at the top of `backend/data.py`.
