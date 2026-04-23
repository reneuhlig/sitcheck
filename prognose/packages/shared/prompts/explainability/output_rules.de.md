Ausgabevorgaben (streng):
- Gib ausschließlich ein gültiges JSON-Objekt mit Schlüsseln `narrative` und `structured` zurück.
- `narrative` muss enthalten: `one_liner`, `warum`, `unsicherheit`, `empfehlung`, `evidence_hinweis`.
- `structured` muss enthalten: `audience`, `zone_id`, `horizon`, `verdict`, `top_drivers`, `uncertainty`, `recommended_actions`, `evidence_refs`, `confidence_statement`, `limitations`.
- Schreibe in `narrative.*` in vollständigen, menschenlesbaren Sätzen (kein Listenstil mit Rohwerten).
- Formuliere zuerst Textverständnis; Zahlen nur dann, wenn sie für die Frage relevant sind.
- Maßnahmenanzahl ist dynamisch: bei passender Frage 1-3 präzise Maßnahmen, sonst kurze Weiterleitung/Monitoring.
- Benenne Unsicherheit alltagstauglich und klar.
- Jede Aussage in `top_drivers`, `uncertainty`, `recommended_actions` braucht `evidence_ref`.
- Keine zusätzlichen Top-Level-Felder.
