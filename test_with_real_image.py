#!/usr/bin/env python3
"""
Test der Personenerkennung mit einem echten Foto
Lädt ein Bild herunter oder verwendet ein vorhandenes
Testet auch die Zeitreihenanalyse mit simulierten Daten
"""

import cv2
import argparse
import sys
import time
from pathlib import Path
import urllib.request
import ssl
from datetime import datetime

from UltralyticsPersonDetector import UltralyticsPersonDetector
from DatabaseHandler import DatabaseHandler
from TimeSeriesAnalyzer import TimeSeriesAnalyzer


# Beispiel-URLs mit Personen (lizenzfrei)
SAMPLE_IMAGES = {
    'crowd': 'https://images.pexels.com/photos/2747449/pexels-photo-2747449.jpeg?auto=compress&cs=tinysrgb&w=640',
    'person': 'https://images.pexels.com/photos/1181690/pexels-photo-1181690.jpeg?auto=compress&cs=tinysrgb&w=640',
    'people': 'https://images.pexels.com/photos/1267360/pexels-photo-1267360.jpeg?auto=compress&cs=tinysrgb&w=640'
}


def download_sample_image(url: str, output_path: str) -> bool:
    """Lädt ein Beispielbild herunter"""
    try:
        print(f"📥 Lade Bild herunter...")
        # SSL-Kontext für HTTPS
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        # Request mit User-Agent wie ein Browser
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        with urllib.request.urlopen(req, context=context) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"✓ Bild gespeichert: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Fehler beim Herunterladen: {e}")
        return False


def test_with_image(image_path: str, confidence_threshold: float = 0.5, 
                   save_result: bool = False, verbose: bool = True) -> bool:
    """
    Testet Personenerkennung mit einem Bild
    
    Args:
        image_path: Pfad zum Bild
        confidence_threshold: Konfidenz-Schwelle
        save_result: Ob das Ergebnis-Bild gespeichert werden soll
        verbose: Detaillierte Ausgabe
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 PERSONENERKENNUNG TEST")
        print(f"{'='*80}")
        print(f"  Bild: {image_path}")
        print(f"  Konfidenz-Schwelle: {confidence_threshold}")
        print(f"{'='*80}\n")
    
    # Bild laden
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ Fehler: Bild konnte nicht geladen werden: {image_path}")
        return False
    
    if verbose:
        print(f"✓ Bild geladen: {img.shape[1]}x{img.shape[0]} px")
    
    # Detector initialisieren
    if verbose:
        print("🚀 Initialisiere YOLO-Modell...")
    detector = UltralyticsPersonDetector(confidence_threshold=confidence_threshold)
    if verbose:
        print("✓ Modell geladen")
    
    # Detection durchführen
    if verbose:
        print("\n🔍 Führe Personenerkennung durch...")
    result = detector.detect(img)
    
    # Ergebnisse anzeigen
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 ERGEBNISSE")
        print(f"{'='*80}")
    print(f"  Erkannte Personen: {result['persons_detected']}")
    
    if result['persons_detected'] > 0:
        print(f"  Durchschnittliche Konfidenz: {result['avg_confidence']:.3f}")
        print(f"  Maximale Konfidenz: {result['max_confidence']:.3f}")
        print(f"  Minimale Konfidenz: {result['min_confidence']:.3f}")
        
        if verbose:
            print(f"\n  Details zu jeder Person:")
            for i, person in enumerate(result['persons'], 1):
                bbox = person['bbox']
                conf = person['confidence']
                print(f"    Person {i}: Konfidenz={conf:.3f}, "
                      f"Position=[x:{bbox[0]:.0f}, y:{bbox[1]:.0f}, "
                      f"w:{bbox[2]-bbox[0]:.0f}, h:{bbox[3]-bbox[1]:.0f}]")
    else:
        print("  ⚠️  Keine Personen erkannt")
        if 'error' in result:
            print(f"  Fehler: {result['error']}")
    
    # Ergebnis-Bild speichern
    if save_result and result['persons_detected'] > 0:
        output_path = str(Path(image_path).parent / f"result_{Path(image_path).name}")
        
        # Zeichne Bounding Boxes
        result_img = img.copy()
        for person in result['persons']:
            bbox = person['bbox']
            conf = person['confidence']
            
            # Box zeichnen
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label zeichnen
            label = f"Person: {conf:.2f}"
            cv2.putText(result_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, result_img)
        print(f"\n✓ Ergebnis-Bild gespeichert: {output_path}")
    
    if verbose:
        print(f"{'='*80}\n")
    
    return result['persons_detected'] > 0


def test_timeseries_analysis(db_config: dict, num_test_pairs: int = 5):
    """
    Testet die Zeitreihenanalyse mit simulierten Detections
    
    Args:
        db_config: Datenbank-Konfiguration
        num_test_pairs: Anzahl der Test-Paare
    """
    print(f"\n{'='*80}")
    print(f"📊 ZEITREIHENANALYSE TEST")
    print(f"{'='*80}")
    print(f"  Test-Paare: {num_test_pairs}")
    print(f"{'='*80}\n")
    
    # Datenbank verbinden
    db = DatabaseHandler(**db_config)
    if not db.connect():
        print("✗ Datenbankverbindung fehlgeschlagen")
        return False
    
    # Tabellen erstellen
    if not db.create_tables():
        print("✗ Tabellenerstellung fehlgeschlagen")
        db.close()
        return False
    
    print("✓ Datenbank verbunden\n")
    
    # Simuliere Test-Detections
    print(f"📝 Erzeuge {num_test_pairs} Test-Paare...")
    
    test_data = [
        # (persons_x, persons_y, confidence_x, confidence_y)
        (3, 3, 0.85, 0.82),  # Perfekte Übereinstimmung
        (5, 4, 0.78, 0.81),  # Abweichung ±1
        (2, 4, 0.65, 0.88),  # Abweichung ±2
        (7, 7, 0.92, 0.90),  # Übereinstimmung, hohe Konfidenz
        (1, 2, 0.55, 0.60),  # Kleine Anzahl
    ]
    
    inserted_ids = {'x': [], 'y': []}
    
    for i, (persons_x, persons_y, conf_x, conf_y) in enumerate(test_data[:num_test_pairs], 1):
        # Insert für source X
        detection_data_x = {
            'persons_detected': persons_x,
            'avg_confidence': conf_x,
            'max_confidence': conf_x + 0.05,
            'min_confidence': conf_x - 0.05,
            'test_pair': i
        }
        
        id_x = db.insert_detection(
            source='input_x',
            persons_detected=persons_x,
            avg_confidence=conf_x,
            max_confidence=conf_x + 0.05,
            min_confidence=conf_x - 0.05,
            detection_data=detection_data_x
        )
        
        # Kleine Zeitverzögerung (simuliert echte Situation)
        time.sleep(0.2)
        
        # Insert für source Y
        detection_data_y = {
            'persons_detected': persons_y,
            'avg_confidence': conf_y,
            'max_confidence': conf_y + 0.05,
            'min_confidence': conf_y - 0.05,
            'test_pair': i
        }
        
        id_y = db.insert_detection(
            source='input_y',
            persons_detected=persons_y,
            avg_confidence=conf_y,
            max_confidence=conf_y + 0.05,
            min_confidence=conf_y - 0.05,
            detection_data=detection_data_y
        )
        
        if id_x and id_y:
            inserted_ids['x'].append(id_x)
            inserted_ids['y'].append(id_y)
            print(f"  Paar {i}: X={persons_x} (conf={conf_x:.2f}), "
                  f"Y={persons_y} (conf={conf_y:.2f}) ✓")
        else:
            print(f"  Paar {i}: Fehler beim Speichern ✗")
    
    print(f"\n✓ {len(inserted_ids['x'])} Test-Paare in Datenbank gespeichert\n")
    
    # Zeitreihenanalyse durchführen
    print("🔍 Führe Zeitreihenanalyse durch...\n")
    
    analyzer = TimeSeriesAnalyzer(db_config)
    
    # Einmalige Analyse
    analyzer.analyze_and_store(interval_seconds=1, continuous=False)
    
    # Ergebnisse abrufen
    print("\n📈 Analysierte Ergebnisse aus Datenbank:")
    cursor = db.connection.cursor()
    query = """
    SELECT 
        persons_x,
        persons_y,
        estimated_actual_persons,
        ROUND(confidence_score::numeric, 3) as confidence,
        ROUND(time_diff_seconds::numeric, 3) as time_diff,
        (analysis_data->>'agreement')::boolean as agreement,
        (analysis_data->>'difference')::int as difference
    FROM correlated_persons
    ORDER BY id DESC
    LIMIT %s
    """
    
    cursor.execute(query, (num_test_pairs,))
    results = cursor.fetchall()
    cursor.close()
    
    if not results:
        print("  ⚠️  Keine Ergebnisse gefunden")
        db.close()
        return False
    
    print(f"\n{'='*80}")
    print(f"{'X':>4} | {'Y':>4} | {'Geschätzt':>10} | {'Konfidenz':>10} | "
          f"{'Zeitdiff':>9} | {'Match':>5} | {'Diff':>4}")
    print('-'*80)
    
    for row in results:
        persons_x, persons_y, estimated, conf, time_diff, agreement, diff = row
        match_symbol = '✓' if agreement else '✗'
        print(f"{persons_x:4d} | {persons_y:4d} | {estimated:10d} | "
              f"{conf:10.3f} | {time_diff:9.3f}s | {match_symbol:>5} | {diff:4d}")
    
    print('='*80)
    
    # Statistiken
    agreements = sum(1 for r in results if r[5])  # agreement column
    avg_confidence = sum(r[3] for r in results) / len(results)
    avg_time_diff = sum(r[4] for r in results) / len(results)
    
    print(f"\n📊 STATISTIK:")
    print(f"  Übereinstimmungen: {agreements}/{len(results)} "
          f"({agreements/len(results)*100:.1f}%)")
    print(f"  Durchschn. Konfidenz: {avg_confidence:.3f}")
    print(f"  Durchschn. Zeitdifferenz: {avg_time_diff:.3f}s")
    
    # Validierung
    success = True
    if agreements < len(results) * 0.4:  # Mind. 40% Übereinstimmungen erwartet
        print("\n  ⚠️  Wenige Übereinstimmungen - Analyse könnte verbessert werden")
    
    if avg_confidence < 0.5:
        print("\n  ⚠️  Niedrige durchschnittliche Konfidenz")
        success = False
    
    if avg_time_diff > 2.0:
        print("\n  ⚠️  Hohe durchschnittliche Zeitdifferenz")
    
    print(f"{'='*80}\n")
    
    db.close()
    
    if success:
        print("✓ Zeitreihenanalyse-Test bestanden\n")
    else:
        print("⚠️  Zeitreihenanalyse-Test mit Warnungen\n")
    
    return success


def test_full_pipeline(db_config: dict, image_path: str, confidence_threshold: float = 0.5):
    """
    Testet die komplette Pipeline: Detection + Zeitreihenanalyse
    
    Args:
        db_config: Datenbank-Konfiguration
        image_path: Pfad zum Testbild
        confidence_threshold: Konfidenz-Schwelle
    """
    print(f"\n{'='*80}")
    print(f"🔄 KOMPLETTER PIPELINE-TEST")
    print(f"{'='*80}\n")
    
    # Schritt 1: Detection-Test
    print("SCHRITT 1: Personenerkennung")
    print("-" * 80)
    detection_success = test_with_image(image_path, confidence_threshold, save_result=False)
    
    if not detection_success:
        print("\n⚠️  Keine Personen erkannt - Pipeline-Test abgebrochen")
        print("💡 Tipp: Verwende ein Bild mit sichtbaren Personen")
        return False
    
    # Schritt 2: Simuliere Detections in DB
    print("\n\nSCHRITT 2: Datenbank-Operationen")
    print("-" * 80)
    
    db = DatabaseHandler(**db_config)
    if not db.connect():
        print("✗ Datenbankverbindung fehlgeschlagen")
        return False
    
    db.create_tables()
    
    # Lade Bild und führe Detection durch
    img = cv2.imread(image_path)
    detector = UltralyticsPersonDetector(confidence_threshold=confidence_threshold)
    result = detector.detect(img)
    
    # Speichere als input_x
    id_x = db.insert_detection(
        source='input_x',
        persons_detected=result['persons_detected'],
        avg_confidence=result['avg_confidence'],
        max_confidence=result['max_confidence'],
        min_confidence=result['min_confidence'],
        detection_data=result
    )
    print(f"✓ Detection X gespeichert (ID: {id_x})")
    
    time.sleep(0.5)
    
    # Simuliere leicht abweichendes Ergebnis für input_y
    variation = 0 if result['persons_detected'] < 2 else 1
    result_y = result.copy()
    result_y['persons_detected'] = max(1, result['persons_detected'] - variation)
    
    id_y = db.insert_detection(
        source='input_y',
        persons_detected=result_y['persons_detected'],
        avg_confidence=result['avg_confidence'] * 0.95,
        max_confidence=result['max_confidence'] * 0.95,
        min_confidence=result['min_confidence'] * 0.95,
        detection_data=result_y
    )
    print(f"✓ Detection Y gespeichert (ID: {id_y})")
    
    db.close()
    
    # Schritt 3: Zeitreihenanalyse
    print("\n\nSCHRITT 3: Zeitreihenanalyse")
    print("-" * 80)
    
    analyzer = TimeSeriesAnalyzer(db_config)
    analyzer.analyze_and_store(interval_seconds=1, continuous=False)
    
    # Ergebnis prüfen
    db.connect()
    cursor = db.connection.cursor()
    cursor.execute("""
        SELECT estimated_actual_persons, confidence_score
        FROM correlated_persons
        ORDER BY id DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    cursor.close()
    db.close()
    
    if result:
        estimated, confidence = result
        print(f"\n✓ Korreliertes Ergebnis:")
        print(f"  Geschätzte Personen: {estimated}")
        print(f"  Konfidenz: {confidence:.3f}")
        print(f"\n{'='*80}")
        print(f"✓ KOMPLETTER PIPELINE-TEST ERFOLGREICH")
        print(f"{'='*80}\n")
        return True
    else:
        print("\n✗ Keine korrelierten Ergebnisse gefunden")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test Personenerkennung mit echtem Foto',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--image', '-i', help='Pfad zum Bild')
    parser.add_argument('--sample', choices=list(SAMPLE_IMAGES.keys()),
                       help='Beispielbild herunterladen und testen')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Konfidenz-Schwelle (0.0-1.0)')
    parser.add_argument('--save-result', '-s', action='store_true',
                       help='Speichere Ergebnis-Bild mit Bounding Boxes')
    parser.add_argument('--list-samples', action='store_true',
                       help='Zeige verfügbare Beispielbilder')
    parser.add_argument('--test-timeseries', action='store_true',
                       help='Teste nur Zeitreihenanalyse')
    parser.add_argument('--test-full-pipeline', action='store_true',
                       help='Teste komplette Pipeline (Detection + Zeitreihenanalyse)')
    parser.add_argument('--num-pairs', type=int, default=5,
                       help='Anzahl Test-Paare für Zeitreihenanalyse')
    
    # Datenbank-Argumente
    parser.add_argument('--db-host', default='localhost', help='PostgreSQL Host')
    parser.add_argument('--db-user', default='aiuser', help='PostgreSQL Benutzer')
    parser.add_argument('--db-password', default='DHBW1234!?', help='PostgreSQL Passwort')
    parser.add_argument('--db-name', default='ai_detection', help='PostgreSQL Datenbank')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL Port')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'user': args.db_user,
        'password': args.db_password,
        'database': args.db_name,
        'port': args.db_port
    }
    
    # Liste Beispielbilder
    if args.list_samples:
        print("\n📋 Verfügbare Beispielbilder:")
        for name, url in SAMPLE_IMAGES.items():
            print(f"  {name}: {url}")
        print("\nVerwendung: python3 test_with_real_image.py --sample <name>")
        return
    
    # Nur Zeitreihenanalyse testen
    if args.test_timeseries:
        success = test_timeseries_analysis(db_config, args.num_pairs)
        sys.exit(0 if success else 1)
    
    # Beispielbild herunterladen
    if args.sample:
        temp_dir = Path("temp_test_images")
        temp_dir.mkdir(exist_ok=True)
        
        image_path = str(temp_dir / f"{args.sample}.jpg")
        
        if not download_sample_image(SAMPLE_IMAGES[args.sample], image_path):
            sys.exit(1)
        
        if args.test_full_pipeline:
            success = test_full_pipeline(db_config, image_path, args.confidence)
        else:
            success = test_with_image(image_path, args.confidence, args.save_result)
        
        if not success and not args.test_full_pipeline:
            print("⚠️  Keine Personen erkannt. Versuche niedrigere Konfidenz-Schwelle:")
            print(f"    python3 {sys.argv[0]} --sample {args.sample} --confidence 0.3")
        
        sys.exit(0 if success else 1)
    
    # Eigenes Bild testen
    if args.image:
        if not Path(args.image).exists():
            print(f"✗ Fehler: Bild nicht gefunden: {args.image}")
            sys.exit(1)
        
        # Test vollständige Pipeline oder nur Detection
        if args.test_full_pipeline:
            success = test_full_pipeline(db_config, args.image, args.confidence)
        else:
            success = test_with_image(args.image, args.confidence, args.save_result)
        
        sys.exit(0 if success else 1)
    
    # Keine Argumente
    parser.print_help()
    print("\n💡 Beispiele:")
    print(f"\n  # Nur Detection testen:")
    print(f"    python3 {sys.argv[0]} --image /pfad/zu/bild.jpg --save-result")
    print(f"    python3 {sys.argv[0]} --sample crowd --save-result")
    print(f"\n  # Komplette Pipeline testen (Detection + Zeitreihenanalyse):")
    print(f"    python3 {sys.argv[0]} --sample crowd --test-full-pipeline")
    print(f"    python3 {sys.argv[0]} --image foto.jpg --test-full-pipeline")
    print(f"\n  # Nur Zeitreihenanalyse testen:")
    print(f"    python3 {sys.argv[0]} --test-timeseries --num-pairs 10")
    print(f"\n  # Liste verfügbare Beispiele:")
    print(f"    python3 {sys.argv[0]} --list-samples")


if __name__ == "__main__":
    main()