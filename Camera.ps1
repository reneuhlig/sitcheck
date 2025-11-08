# ============================================================================
# OPTIMIERTES KAMERA-UPLOAD-SYSTEM (KORRIGIERT)
# ============================================================================

# === KONFIGURATION ===
$Config = @{
    # Server
    ServerIP    = "192.168.194.65"
    ServerUser  = "kiadmin"
    ServerPass  = "DHBW1234!?"
    RemotePathX = "/sitcheck/input_x"
    RemotePathY = "/sitcheck/input_y"
    
    # Kameras
    CameraX     = 2
    CameraY     = 3
    
    # Performance
    CaptureMode = "parallel"  # "parallel" oder "sequential"
    UploadMode  = "batch"     # "batch" oder "immediate"
    BatchSize   = 3           # Anzahl Bilder pro Batch-Upload (reduziert auf 3)
    
    # Timing
    Interval    = 0.1         # Sekunden zwischen Captures
    
    # Qualitaet
    Quality     = 80
    Resolution  = @{
        Width  = 1920
        Height = 1080
    }
    
    # Dateien
    TempDir     = "$env:TEMP\camera_snapshots"
    UsePuTTY    = $true
    
    # Debug
    Debug       = $true       # Aktiviert detaillierte Ausgaben
}

# Verzeichnisse erstellen
$QueueX = Join-Path $Config.TempDir "queue_x"
$QueueY = Join-Path $Config.TempDir "queue_y"

New-Item -ItemType Directory -Path $Config.TempDir, $QueueX, $QueueY -Force -ErrorAction SilentlyContinue | Out-Null

# ============================================================================
# PYTHON-SCRIPTS
# ============================================================================

$PythonCaptureScript = @"
import cv2
import sys

camera_index = int(sys.argv[1])
output_path = sys.argv[2]
quality = int(sys.argv[3])
width = int(sys.argv[4])
height = int(sys.argv[5])

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Cannot open camera", file=sys.stderr)
    sys.exit(1)

# Nur 2 Frames ueberspringen
for _ in range(2):
    cap.read()

ret, frame = cap.read()
cap.release()

if ret:
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print("OK")
else:
    print("ERROR: Cannot capture frame", file=sys.stderr)
    sys.exit(1)
"@

$PythonDiagnoseScript = @"
import cv2

print("[DIAGNOSE] Suche verfuegbare Kameras...")
available = []

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            available.append(i)
            print(f"  [OK] Kamera {i}: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"  [WARN] Kamera {i}: Geoeffnet, aber kein Frame")
        cap.release()

if not available:
    print("[ERROR] Keine Kameras gefunden!")
else:
    print(f"[INFO] Verfuegbare Indizes: {available}")
"@

# Python-Scripts speichern
$ScriptCapture = Join-Path $Config.TempDir "capture.py"
$ScriptDiagnose = Join-Path $Config.TempDir "diagnose.py"

$PythonCaptureScript | Out-File -FilePath $ScriptCapture -Encoding UTF8
$PythonDiagnoseScript | Out-File -FilePath $ScriptDiagnose -Encoding UTF8

# ============================================================================
# FUNKTIONEN: KAMERA-CAPTURE
# ============================================================================

function Invoke-CameraCapture {
    param(
        [int]$DeviceIndex,
        [string]$OutputPath
    )
    
    try {
        $args = @(
            $ScriptCapture,
            $DeviceIndex,
            $OutputPath,
            $Config.Quality,
            $Config.Resolution.Width,
            $Config.Resolution.Height
        )
        
        $result = & python @args 2>&1
        
        if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputPath)) {
            return @{
                Success = $true
                FilePath = $OutputPath
                Size = (Get-Item $OutputPath).Length
            }
        } else {
            return @{ Success = $false; Error = $result }
        }
    } catch {
        return @{ Success = $false; Error = $_.Exception.Message }
    }
}

# ============================================================================
# FUNKTIONEN: UPLOAD (KORRIGIERT)
# ============================================================================

function Send-FileBatch {
    param(
        [string[]]$LocalFiles,
        [string]$RemotePath,
        [string]$CameraName
    )
    
    if ($LocalFiles.Count -eq 0) {
        return $true
    }
    
    $timestamp = Get-Date -Format "HH:mm:ss"
    $successCount = 0
    $errorCount = 0
    
    try {
        foreach ($file in $LocalFiles) {
            $fileName = Split-Path $file -Leaf
            $remoteFile = "$RemotePath/$fileName"
            
            if ($Config.Debug) {
                Write-Host "[$timestamp] [DEBUG] Upload: $fileName -> $remoteFile" -ForegroundColor Gray
            }
            
            # pscp Upload mit Fehlerbehandlung
            $output = & pscp.exe -batch -pw $Config.ServerPass $file "$($Config.ServerUser)@$($Config.ServerIP):$remoteFile" 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                $successCount++
                Remove-Item $file -Force -ErrorAction SilentlyContinue
                
                if ($Config.Debug) {
                    Write-Host "[$timestamp] [DEBUG] OK: $fileName" -ForegroundColor Gray
                }
            } else {
                $errorCount++
                Write-Host "[$timestamp] [ERROR] $CameraName Upload: $fileName" -ForegroundColor Red
                
                if ($Config.Debug) {
                    Write-Host "[$timestamp] [DEBUG] pscp output: $output" -ForegroundColor Red
                }
                
                # Datei bei Fehler loeschen um Queue nicht zu blockieren
                Remove-Item $file -Force -ErrorAction SilentlyContinue
            }
        }
        
        if ($successCount -gt 0) {
            Write-Host "[$timestamp] [OK] $CameraName Batch: $successCount/$($LocalFiles.Count) Bilder" -ForegroundColor Green
        }
        
        return ($errorCount -eq 0)
        
    } catch {
        Write-Host "[$timestamp] [ERROR] $CameraName Exception: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# ============================================================================
# FUNKTIONEN: BATCH-QUEUE
# ============================================================================

function Get-BatchQueueFiles {
    param(
        [string]$QueueDir,
        [int]$MaxCount
    )
    
    try {
        $files = Get-ChildItem -Path $QueueDir -Filter "*.jpg" -File -ErrorAction SilentlyContinue | 
                 Sort-Object LastWriteTime | 
                 Select-Object -First $MaxCount
        
        return $files.FullName
    } catch {
        return @()
    }
}

# ============================================================================
# FUNKTIONEN: PARALLEL CAPTURE
# ============================================================================

function Start-ParallelCapture {
    param(
        [int]$CameraX,
        [int]$CameraY,
        [string]$OutputDirX,
        [string]$OutputDirY
    )
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $fileX = Join-Path $OutputDirX "x_$timestamp.jpg"
    $fileY = Join-Path $OutputDirY "y_$timestamp.jpg"
    
    # Runspaces fuer parallele Ausfuehrung
    $rsPoolSize = 2
    $rsPool = [runspacefactory]::CreateRunspacePool(1, $rsPoolSize)
    $rsPool.Open()
    
    # Job X
    $psX = [powershell]::Create().AddScript({
        param($ScriptPath, $Index, $Output, $Quality, $Width, $Height)
        $result = & python $ScriptPath $Index $Output $Quality $Width $Height 2>&1
        return @{
            Success = ($LASTEXITCODE -eq 0 -and (Test-Path $Output))
            Path = $Output
            Error = $result
        }
    }).AddArgument($ScriptCapture).AddArgument($CameraX).AddArgument($fileX).AddArgument($Config.Quality).AddArgument($Config.Resolution.Width).AddArgument($Config.Resolution.Height)
    
    $psX.RunspacePool = $rsPool
    
    # Job Y
    $psY = [powershell]::Create().AddScript({
        param($ScriptPath, $Index, $Output, $Quality, $Width, $Height)
        $result = & python $ScriptPath $Index $Output $Quality $Width $Height 2>&1
        return @{
            Success = ($LASTEXITCODE -eq 0 -and (Test-Path $Output))
            Path = $Output
            Error = $result
        }
    }).AddArgument($ScriptCapture).AddArgument($CameraY).AddArgument($fileY).AddArgument($Config.Quality).AddArgument($Config.Resolution.Width).AddArgument($Config.Resolution.Height)
    
    $psY.RunspacePool = $rsPool
    
    # Starten
    $handleX = $psX.BeginInvoke()
    $handleY = $psY.BeginInvoke()
    
    # Warten
    $resultX = $psX.EndInvoke($handleX)
    $resultY = $psY.EndInvoke($handleY)
    
    # Cleanup
    $psX.Dispose()
    $psY.Dispose()
    $rsPool.Close()
    $rsPool.Dispose()
    
    return @{
        X = $resultX
        Y = $resultY
    }
}

# ============================================================================
# SYSTEM-CHECKS
# ============================================================================

Write-Host "`n[SYSTEM] Pruefe Systemvoraussetzungen..." -ForegroundColor Cyan

# Python
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python nicht gefunden!" -ForegroundColor Red
    exit 1
}

# OpenCV
try {
    $opencvCheck = & python -c "import cv2; print('OK')" 2>&1
    if ($opencvCheck -match "OK") {
        Write-Host "[OK] OpenCV installiert" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] OpenCV nicht gefunden!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "[ERROR] OpenCV nicht gefunden!" -ForegroundColor Red
    exit 1
}

# pscp
if (Get-Command "pscp.exe" -ErrorAction SilentlyContinue) {
    Write-Host "[OK] pscp.exe gefunden" -ForegroundColor Green
} else {
    Write-Host "[ERROR] pscp.exe nicht gefunden!" -ForegroundColor Red
    exit 1
}

# Server-Ping
try {
    $ping = Test-Connection -ComputerName $Config.ServerIP -Count 1 -Quiet -ErrorAction Stop
    if ($ping) {
        Write-Host "[OK] Server erreichbar: $($Config.ServerIP)" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Server nicht erreichbar!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "[ERROR] Ping fehlgeschlagen" -ForegroundColor Red
    exit 1
}

# SSH-Verbindung testen
Write-Host "`n[TEST] Teste SSH-Verbindung und Upload..." -ForegroundColor Cyan
try {
    # Test-Datei erstellen
    $testFile = Join-Path $Config.TempDir "test_upload.txt"
    "Test" | Out-File -FilePath $testFile -Encoding ASCII
    
    # Test-Upload zu X
    $testRemoteX = "$($Config.RemotePathX)/test_upload.txt"
    $result = & pscp.exe -batch -pw $Config.ServerPass $testFile "$($Config.ServerUser)@$($Config.ServerIP):$testRemoteX" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] SSH-Upload funktioniert" -ForegroundColor Green
        
        # Test-Datei auf Server loeschen
        $null = & plink.exe -batch -pw $Config.ServerPass "$($Config.ServerUser)@$($Config.ServerIP)" "rm -f $testRemoteX" 2>&1
    } else {
        Write-Host "[ERROR] SSH-Upload fehlgeschlagen!" -ForegroundColor Red
        Write-Host "[DEBUG] Output: $result" -ForegroundColor Red
        
        Write-Host "`n[HINWEIS] Moegliche Ursachen:" -ForegroundColor Yellow
        Write-Host "  1. Falsches Passwort" -ForegroundColor Yellow
        Write-Host "  2. Remote-Pfad existiert nicht: $($Config.RemotePathX)" -ForegroundColor Yellow
        Write-Host "  3. Keine Schreibrechte auf Server" -ForegroundColor Yellow
        Write-Host "`n[TIPP] Pruefe auf Server:" -ForegroundColor Yellow
        Write-Host "  ssh $($Config.ServerUser)@$($Config.ServerIP)" -ForegroundColor Gray
        Write-Host "  ls -la $($Config.RemotePathX)" -ForegroundColor Gray
        Write-Host "  touch $($Config.RemotePathX)/test.txt" -ForegroundColor Gray
        
        exit 1
    }
    
    # Cleanup
    Remove-Item $testFile -Force -ErrorAction SilentlyContinue
    
} catch {
    Write-Host "[ERROR] SSH-Test fehlgeschlagen: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Kamera-Diagnose
Write-Host "`n[DIAGNOSE] Verfuegbare Kameras:" -ForegroundColor Cyan
& python $ScriptDiagnose

# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

Write-Host "`n[SYSTEM] Starte Kamera-Upload-System" -ForegroundColor Cyan
Write-Host "[CONFIG] Capture-Modus: $($Config.CaptureMode)" -ForegroundColor Gray
Write-Host "[CONFIG] Upload-Modus: $($Config.UploadMode)" -ForegroundColor Gray
Write-Host "[CONFIG] Intervall: $($Config.Interval)s" -ForegroundColor Gray
Write-Host "[CONFIG] Batch-Groesse: $($Config.BatchSize)" -ForegroundColor Gray
Write-Host "[CONFIG] Server: $($Config.ServerUser)@$($Config.ServerIP)" -ForegroundColor Gray
Write-Host "[CONFIG] Debug: $($Config.Debug)" -ForegroundColor Gray
Write-Host "`n[INFO] Druecke Strg+C zum Beenden...`n" -ForegroundColor Yellow

$stats = @{
    Success = 0
    Error = 0
    StartTime = Get-Date
    LastStatTime = Get-Date
}

try {
    while ($true) {
        $loopStart = Get-Date
        
        # === CAPTURE ===
        if ($Config.CaptureMode -eq "parallel") {
            # Parallel Capture
            $results = Start-ParallelCapture -CameraX $Config.CameraX -CameraY $Config.CameraY -OutputDirX $QueueX -OutputDirY $QueueY
            
            if ($results.X.Success -and $results.Y.Success) {
                $stats.Success += 2
            } else {
                $stats.Error++
                if (-not $results.X.Success) { 
                    Write-Host "[ERROR] Kamera X Capture fehlgeschlagen" -ForegroundColor Red 
                }
                if (-not $results.Y.Success) { 
                    Write-Host "[ERROR] Kamera Y Capture fehlgeschlagen" -ForegroundColor Red 
                }
            }
        } else {
            # Sequential Capture
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
            $fileX = Join-Path $QueueX "x_$timestamp.jpg"
            $fileY = Join-Path $QueueY "y_$timestamp.jpg"
            
            $snapX = Invoke-CameraCapture -DeviceIndex $Config.CameraX -OutputPath $fileX
            $snapY = Invoke-CameraCapture -DeviceIndex $Config.CameraY -OutputPath $fileY
            
            if ($snapX.Success -and $snapY.Success) {
                $stats.Success += 2
            } else {
                $stats.Error++
            }
        }
        
        # === UPLOAD (BATCH) ===
        if ($Config.UploadMode -eq "batch") {
            # Pruefe Queue X
            $filesX = Get-BatchQueueFiles -QueueDir $QueueX -MaxCount $Config.BatchSize
            if ($filesX.Count -ge $Config.BatchSize) {
                $uploadSuccess = Send-FileBatch -LocalFiles $filesX -RemotePath $Config.RemotePathX -CameraName "X"
            }
            
            # Pruefe Queue Y
            $filesY = Get-BatchQueueFiles -QueueDir $QueueY -MaxCount $Config.BatchSize
            if ($filesY.Count -ge $Config.BatchSize) {
                $uploadSuccess = Send-FileBatch -LocalFiles $filesY -RemotePath $Config.RemotePathY -CameraName "Y"
            }
        } elseif ($Config.UploadMode -eq "immediate") {
            # Sofort-Upload: Upload neueste Dateien
            $filesX = Get-BatchQueueFiles -QueueDir $QueueX -MaxCount 1
            if ($filesX.Count -gt 0) {
                Send-FileBatch -LocalFiles $filesX -RemotePath $Config.RemotePathX -CameraName "X"
            }
            
            $filesY = Get-BatchQueueFiles -QueueDir $QueueY -MaxCount 1
            if ($filesY.Count -gt 0) {
                Send-FileBatch -LocalFiles $filesY -RemotePath $Config.RemotePathY -CameraName "Y"
            }
        }
        
        # === STATISTIK ===
        $timeSinceStats = (Get-Date) - $stats.LastStatTime
        if ($timeSinceStats.TotalSeconds -ge 10) {
            $runtime = (Get-Date) - $stats.StartTime
            $totalOps = $stats.Success + $stats.Error
            
            if ($totalOps -gt 0) {
                $rate = [math]::Round(($stats.Success / $totalOps) * 100, 1)
                $fps = [math]::Round($totalOps / $runtime.TotalSeconds, 2)
                
                # Queue-Status
                $queueXCount = (Get-ChildItem -Path $QueueX -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
                $queueYCount = (Get-ChildItem -Path $QueueY -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
                
                Write-Host "`n[STATS] Erfolg: $($stats.Success) | Fehler: $($stats.Error) | Rate: $rate% | FPS: $fps | Queue: X=$queueXCount Y=$queueYCount" -ForegroundColor Cyan
                
                $stats.LastStatTime = Get-Date
            }
        }
        
        # === INTERVALL ===
        $elapsed = ((Get-Date) - $loopStart).TotalSeconds
        $sleepTime = [math]::Max(0, $Config.Interval - $elapsed)
        if ($sleepTime -gt 0) {
            Start-Sleep -Seconds $sleepTime
        }
    }
} catch {
    Write-Host "`n[ERROR] Exception: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Write-Host "`n[INFO] System beendet" -ForegroundColor Yellow
}