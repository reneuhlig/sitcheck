-- =============================================================================
-- NÜTZLICHE SQL-ABFRAGEN FÜR DAS LIVE-PERSONENERKENNUNG-SYSTEM
-- =============================================================================

-- 1. ROHDATEN (live_detections)
-- =============================================================================

-- Letzte 20 Detections anzeigen
SELECT 
    id,
    timestamp,
    source,
    persons_detected,
    ROUND(avg_confidence::numeric, 3) as avg_conf,
    ROUND(max_confidence::numeric, 3) as max_conf
FROM live_detections 
ORDER BY timestamp DESC 
LIMIT 20;

-- Detections pro Quelle (letzte Stunde)
SELECT 
    source,
    COUNT(*) as total_detections,
    AVG(persons_detected) as avg_persons,
    MIN(persons_detected) as min_persons,
    MAX(persons_detected) as max_persons,
    ROUND(AVG(avg_confidence)::numeric, 3) as avg_confidence
FROM live_detections 
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY source
ORDER BY source;

-- Zeitreihe der Detections (5-Minuten-Intervalle)
SELECT 
    DATE_TRUNC('minute', timestamp) - 
    (EXTRACT(MINUTE FROM timestamp)::int % 5) * INTERVAL '1 minute' as time_bucket,
    source,
    COUNT(*) as detections,
    AVG(persons_detected) as avg_persons
FROM live_detections 
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY time_bucket, source
ORDER BY time_bucket DESC, source;

-- Detections mit niedriger Konfidenz finden
SELECT 
    id,
    timestamp,
    source,
    persons_detected,
    ROUND(avg_confidence::numeric, 3) as avg_conf
FROM live_detections 
WHERE avg_confidence < 0.6
ORDER BY timestamp DESC 
LIMIT 20;

-- =============================================================================
-- 2. KORRELIERTE DATEN (correlated_persons)
-- =============================================================================

-- Letzte 20 korrelierte Ergebnisse
SELECT 
    id,
    timestamp,
    persons_x,
    persons_y,
    estimated_actual_persons,
    ROUND(confidence_score::numeric, 3) as confidence,
    ROUND(time_diff_seconds::numeric, 3) as time_diff
FROM correlated_persons 
ORDER BY timestamp DESC 
LIMIT 20;

-- Übereinstimmungen vs. Abweichungen
SELECT 
    CASE 
        WHEN persons_x = persons_y THEN 'Übereinstimmung'
        WHEN ABS(persons_x - persons_y) = 1 THEN 'Abweichung ±1'
        WHEN ABS(persons_x - persons_y) = 2 THEN 'Abweichung ±2'
        ELSE 'Große Abweichung (>2)'
    END as kategorie,
    COUNT(*) as anzahl,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence
FROM correlated_persons 
GROUP BY kategorie
ORDER BY anzahl DESC;

-- Durchschnittliche Personenanzahl über Zeit (10-Minuten-Intervalle)
SELECT 
    DATE_TRUNC('hour', timestamp) + 
    (EXTRACT(MINUTE FROM timestamp)::int / 10) * INTERVAL '10 minutes' as time_bucket,
    COUNT(*) as correlations,
    ROUND(AVG(estimated_actual_persons)::numeric, 2) as avg_persons,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence
FROM correlated_persons 
WHERE timestamp >= NOW() - INTERVAL '1 day'
GROUP BY time_bucket
ORDER BY time_bucket DESC;

-- Beste und schlechteste Schätzungen (nach Konfidenz)
(SELECT 'Top 5' as typ, timestamp, persons_x, persons_y, estimated_actual_persons,
        ROUND(confidence_score::numeric, 3) as confidence
 FROM correlated_persons 
 ORDER BY confidence_score DESC 
 LIMIT 5)
UNION ALL
(SELECT 'Bottom 5' as typ, timestamp, persons_x, persons_y, estimated_actual_persons,
        ROUND(confidence_score::numeric, 3) as confidence
 FROM correlated_persons 
 ORDER BY confidence_score ASC 
 LIMIT 5)
ORDER BY typ, confidence DESC;

-- =============================================================================
-- 3. KOMBINIERTE ANALYSEN
-- =============================================================================

-- Vergleich Rohdaten vs. Korrelierte Daten
WITH raw_stats AS (
    SELECT 
        AVG(persons_detected) as avg_raw_persons,
        STDDEV(persons_detected) as stddev_raw
    FROM live_detections 
    WHERE timestamp >= NOW() - INTERVAL '1 hour'
),
correlated_stats AS (
    SELECT 
        AVG(estimated_actual_persons) as avg_corr_persons,
        STDDEV(estimated_actual_persons) as stddev_corr
    FROM correlated_persons 
    WHERE timestamp >= NOW() - INTERVAL '1 hour'
)
SELECT 
    ROUND(r.avg_raw_persons::numeric, 2) as avg_raw,
    ROUND(r.stddev_raw::numeric, 2) as stddev_raw,
    ROUND(c.avg_corr_persons::numeric, 2) as avg_correlated,
    ROUND(c.stddev_corr::numeric, 2) as stddev_correlated
FROM raw_stats r, correlated_stats c;

-- Zeitdifferenz-Analyse
SELECT 
    CASE 
        WHEN time_diff_seconds < 1 THEN '< 1s'
        WHEN time_diff_seconds < 2 THEN '1-2s'
        WHEN time_diff_seconds < 3 THEN '2-3s'
        WHEN time_diff_seconds < 4 THEN '3-4s'
        ELSE '≥ 4s'
    END as zeitdifferenz,
    COUNT(*) as anzahl,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence
FROM correlated_persons 
GROUP BY 
    CASE 
        WHEN time_diff_seconds < 1 THEN '< 1s'
        WHEN time_diff_seconds < 2 THEN '1-2s'
        WHEN time_diff_seconds < 3 THEN '2-3s'
        WHEN time_diff_seconds < 4 THEN '3-4s'
        ELSE '≥ 4s'
    END
ORDER BY anzahl DESC;

-- Join: Korrelierte Daten mit Original-Detections
SELECT 
    cp.timestamp as corr_time,
    cp.persons_x,
    cp.persons_y,
    cp.estimated_actual_persons,
    ROUND(cp.confidence_score::numeric, 3) as confidence,
    ROUND(ldx.avg_confidence::numeric, 3) as x_conf,
    ROUND(ldy.avg_confidence::numeric, 3) as y_conf,
    ldx.timestamp as x_time,
    ldy.timestamp as y_time
FROM correlated_persons cp
LEFT JOIN live_detections ldx ON cp.source_x_id = ldx.id
LEFT JOIN live_detections ldy ON cp.source_y_id = ldy.id
ORDER BY cp.timestamp DESC
LIMIT 10;

-- =============================================================================
-- 4. STATISTIKEN & MONITORING
-- =============================================================================

-- Gesamtstatistik
SELECT 
    'Rohdaten' as tabelle,
    COUNT(*) as gesamt,
    MIN(timestamp) as erste_detection,
    MAX(timestamp) as letzte_detection,
    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 3600 as stunden
FROM live_detections
UNION ALL
SELECT 
    'Korreliert' as tabelle,
    COUNT(*) as gesamt,
    MIN(timestamp) as erste_detection,
    MAX(timestamp) as letzte_detection,
    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 3600 as stunden
FROM correlated_persons;

-- Detections pro Stunde (letzte 24h)
SELECT 
    EXTRACT(HOUR FROM timestamp) as stunde,
    COUNT(*) as detections,
    ROUND(AVG(persons_detected)::numeric, 2) as avg_persons
FROM live_detections 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY EXTRACT(HOUR FROM timestamp)
ORDER BY stunde;

-- Aktuelle System-Performance
SELECT 
    'Letzte 5 Minuten' as zeitraum,
    COUNT(DISTINCT ld.id) as rohdaten,
    COUNT(DISTINCT cp.id) as korreliert,
    ROUND((COUNT(DISTINCT cp.id)::float / NULLIF(COUNT(DISTINCT ld.id), 0) * 100)::numeric, 1) 
        as korrelations_rate_prozent
FROM live_detections ld
LEFT JOIN correlated_persons cp ON cp.timestamp >= NOW() - INTERVAL '5 minutes'
WHERE ld.timestamp >= NOW() - INTERVAL '5 minutes';

-- =============================================================================
-- 5. DATENBEREINIGUNG
-- =============================================================================

-- Alte Daten löschen (älter als 7 Tage)
-- ACHTUNG: Erst testen mit SELECT!
-- DELETE FROM live_detections WHERE timestamp < NOW() - INTERVAL '7 days';
-- DELETE FROM correlated_persons WHERE timestamp < NOW() - INTERVAL '7 days';

-- Anzahl Datensätze zum Löschen anzeigen
SELECT 
    'live_detections' as tabelle,
    COUNT(*) as zu_loeschen
FROM live_detections 
WHERE timestamp < NOW() - INTERVAL '7 days'
UNION ALL
SELECT 
    'correlated_persons' as tabelle,
    COUNT(*) as zu_loeschen
FROM correlated_persons 
WHERE timestamp < NOW() - INTERVAL '7 days';

-- =============================================================================
-- 6. SPEZIELLE ANALYSEN
-- =============================================================================

-- Personenanzahl-Verteilung
SELECT 
    estimated_actual_persons as personen,
    COUNT(*) as haeufigkeit,
    ROUND(AVG(confidence_score)::numeric, 3) as avg_confidence
FROM correlated_persons 
GROUP BY estimated_actual_persons
ORDER BY personen;

-- Peak-Zeiten identifizieren
SELECT 
    DATE_TRUNC('hour', timestamp) as stunde,
    MAX(estimated_actual_persons) as max_persons,
    AVG(estimated_actual_persons) as avg_persons
FROM correlated_persons 
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY max_persons DESC
LIMIT 10;