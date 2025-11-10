# ============================================================================
# ULTRA-OPTIMIERTES KAMERA-SYSTEM (2+ FPS)
# ============================================================================

$Config = @{
    ServerIP    = "192.168.194.65"
    ServerUser  = "kiadmin"
    ServerPass  = "DHBW1234!?"
    RemotePathX = "/sitcheck/input_x"
    RemotePathY = "/sitcheck/input_y"
    CameraX     = 2
    CameraY     = 3
    Quality     = 80
    Resolution  = @{ Width = 1920; Height = 1080 }
    TempDir     = "$env:TEMP\camera_snapshots"
    BatchSize   = 5        # Upload alle 5 Bilder
    TargetFPS   = 2        # Ziel: 2 FPS pro Kamera
    Debug       = $true
}

$QueueX = Join-Path $Config.TempDir "queue_x"
$QueueY = Join-Path $Config.TempDir "queue_y"
New-Item -ItemType Directory -Path $Config.TempDir, $QueueX, $QueueY -Force -ErrorAction SilentlyContinue | Out-Null

# Python Worker Script erstellen
$WorkerScriptPath = Join-Path $Config.TempDir "camera_worker.py"

$PythonWorkerCode = @"
import cv2
import sys
import time
import os

def main():
    if len(sys.argv) != 6:
        print("ERROR: Invalid arguments", file=sys.stderr)
        sys.exit(1)
    
    camera_index = int(sys.argv[1])
    output_dir = sys.argv[2]
    quality = int(sys.argv[3])
    width = int(sys.argv[4])
    height = int(sys.argv[5])
    
    # Kamera einmalig öffnen
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera", file=sys.stderr)
        sys.exit(1)
    
    # Initial frames überspringen
    for _ in range(2):
        cap.read()
    
    print("READY")
    sys.stdout.flush()
    
    frame_count = 0
    
    while True:
        try:
            cmd = input().strip()
            
            if cmd == "CAPTURE":
                ret, frame = cap.read()
                
                if ret:
                    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{frame_count:04d}"
                    filename = f"{timestamp}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    print(f"OK:{filepath}")
                    frame_count += 1
                else:
                    print("ERROR:Frame capture failed")
                
                sys.stdout.flush()
                
            elif cmd == "EXIT":
                break
                
        except EOFError:
            break
        except Exception as e:
            print(f"ERROR:{str(e)}", file=sys.stderr)
            sys.stderr.flush()
    
    cap.release()
    print("CLOSED")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
"@

$PythonWorkerCode | Out-File -FilePath $WorkerScriptPath -Encoding UTF8

# ============================================================================
# KLASSE: Persistenter Kamera-Worker
# ============================================================================

class CameraWorker {
    [System.Diagnostics.Process]$Process
    [string]$Name
    [bool]$IsReady = $false
    [string]$WorkerScriptPath
    
    CameraWorker([int]$cameraIndex, [string]$outputDir, [hashtable]$config, [string]$name, [string]$scriptPath) {
        $this.Name = $name
        $this.WorkerScriptPath = $scriptPath
        
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "python"
        $psi.Arguments = @(
            $this.WorkerScriptPath,
            $cameraIndex,
            $outputDir,
            $config.Quality,
            $config.Resolution.Width,
            $config.Resolution.Height
        ) -join " "
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true
        
        $this.Process = New-Object System.Diagnostics.Process
        $this.Process.StartInfo = $psi
        $this.Process.Start() | Out-Null
        
        # Warte auf READY Signal
        $timeout = [DateTime]::Now.AddSeconds(5)
        $output = ""
        
        while ([DateTime]::Now -lt $timeout) {
            if ($this.Process.StandardOutput.Peek() -ge 0) {
                $output = $this.Process.StandardOutput.ReadLine()
                break
            }
            Start-Sleep -Milliseconds 100
        }
        
        if ($output -eq "READY") {
            $this.IsReady = $true
            Write-Host "[OK] Worker $($this.Name) bereit" -ForegroundColor Green
        } else {
            $error = $this.Process.StandardError.ReadToEnd()
            throw "Worker $($this.Name) failed to start. Output: $output, Error: $error"
        }
    }
    
    [string] Capture() {
        if (-not $this.IsReady) { return $null }
        
        try {
            $this.Process.StandardInput.WriteLine("CAPTURE")
            $this.Process.StandardInput.Flush()
            
            $result = $this.Process.StandardOutput.ReadLine()
            
            if ($result -match "^OK:(.+)$") {
                return $Matches[1]
            }
        } catch {
            Write-Host "[ERROR] Capture failed for $($this.Name): $_" -ForegroundColor Red
        }
        
        return $null
    }
    
    [void] Stop() {
        if ($this.Process -and -not $this.Process.HasExited) {
            try {
                $this.Process.StandardInput.WriteLine("EXIT")
                $this.Process.StandardInput.Flush()
                $this.Process.WaitForExit(2000)
            } catch {}
            
            if (-not $this.Process.HasExited) {
                $this.Process.Kill()
            }
        }
    }
}

# ============================================================================
# KLASSE: Asynchroner Upload-Manager
# ============================================================================

class UploadManager {
    [System.Collections.Concurrent.ConcurrentQueue[string]]$QueueX
    [System.Collections.Concurrent.ConcurrentQueue[string]]$QueueY
    [hashtable]$Config
    [bool]$Running = $true
    [int]$UploadedCount = 0
    [System.Management.Automation.Runspaces.Runspace]$Runspace
    [System.Management.Automation.PowerShell]$PowerShell
    
    UploadManager([hashtable]$config) {
        $this.Config = $config
        $this.QueueX = New-Object System.Collections.Concurrent.ConcurrentQueue[string]
        $this.QueueY = New-Object System.Collections.Concurrent.ConcurrentQueue[string]
        
        # Runspace für Upload-Thread
        $this.Runspace = [runspacefactory]::CreateRunspace()
        $this.Runspace.Open()
        
        $this.PowerShell = [powershell]::Create()
        $this.PowerShell.Runspace = $this.Runspace
        
        $scriptBlock = {
            param($manager)
            
            while ($manager.Running) {
                $uploadedAny = $false
                
                # Upload Queue X
                if ($manager.QueueX.Count -ge $manager.Config.BatchSize) {
                    $files = @()
                    for ($i = 0; $i -lt $manager.Config.BatchSize; $i++) {
                        $file = $null
                        if ($manager.QueueX.TryDequeue([ref]$file)) {
                            $files += $file
                        }
                    }
                    if ($files.Count -gt 0) {
                        $manager.UploadBatch($files, $manager.Config.RemotePathX, "X")
                        $uploadedAny = $true
                    }
                }
                
                # Upload Queue Y
                if ($manager.QueueY.Count -ge $manager.Config.BatchSize) {
                    $files = @()
                    for ($i = 0; $i -lt $manager.Config.BatchSize; $i++) {
                        $file = $null
                        if ($manager.QueueY.TryDequeue([ref]$file)) {
                            $files += $file
                        }
                    }
                    if ($files.Count -gt 0) {
                        $manager.UploadBatch($files, $manager.Config.RemotePathY, "Y")
                        $uploadedAny = $true
                    }
                }
                
                if (-not $uploadedAny) {
                    Start-Sleep -Milliseconds 200
                }
            }
        }
        
        $this.PowerShell.AddScript($scriptBlock).AddArgument($this) | Out-Null
        $this.PowerShell.BeginInvoke() | Out-Null
    }
    
    [void] AddFile([string]$path, [string]$camera) {
        if ($camera -eq "X") {
            $this.QueueX.Enqueue($path)
        } else {
            $this.QueueY.Enqueue($path)
        }
    }
    
    [void] UploadBatch([string[]]$files, [string]$remotePath, [string]$camera) {
        $timestamp = Get-Date -Format "HH:mm:ss"
        $success = 0
        
        foreach ($file in $files) {
            if (-not (Test-Path $file)) {
                continue
            }
            
            $fileName = Split-Path $file -Leaf
            $remoteFile = "$remotePath/$fileName"
            
            $null = & pscp.exe -batch -pw $this.Config.ServerPass $file "$($this.Config.ServerUser)@$($this.Config.ServerIP):$remoteFile" 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                $success++
            }
            
            Remove-Item $file -Force -ErrorAction SilentlyContinue
        }
        
        $this.UploadedCount += $success
        Write-Host "[$timestamp] [UPLOAD] $camera`: $success/$($files.Count) Bilder" -ForegroundColor Cyan
    }
    
    [void] Stop() {
        $this.Running = $false
        Start-Sleep -Seconds 2
        
        if ($this.PowerShell) {
            $this.PowerShell.Stop()
            $this.PowerShell.Dispose()
        }
        
        if ($this.Runspace) {
            $this.Runspace.Close()
            $this.Runspace.Dispose()
        }
    }
}

# ============================================================================
# SYSTEM-CHECKS
# ============================================================================

Write-Host "`n[SYSTEM] Starte Ultra-Fast System..." -ForegroundColor Cyan

try {
    $null = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Python nicht gefunden" }
    
    $null = & python -c "import cv2" 2>&1
    if ($LASTEXITCODE -ne 0) { throw "OpenCV nicht gefunden" }
    
    $null = Get-Command pscp.exe -ErrorAction Stop
    
    Write-Host "[OK] Alle Dependencies gefunden" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Fehlende Dependencies: $_" -ForegroundColor Red
    exit 1
}

# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

Write-Host "[SYSTEM] Starte persistente Kamera-Worker..." -ForegroundColor Cyan

try {
    $workerX = [CameraWorker]::new($Config.CameraX, $QueueX, $Config, "X", $WorkerScriptPath)
    $workerY = [CameraWorker]::new($Config.CameraY, $QueueY, $Config, "Y", $WorkerScriptPath)
} catch {
    Write-Host "[ERROR] Konnte Worker nicht starten: $_" -ForegroundColor Red
    exit 1
}

Write-Host "[SYSTEM] Starte Upload-Manager..." -ForegroundColor Cyan
$uploadMgr = [UploadManager]::new($Config)

Write-Host "`n[INFO] System läuft - Ziel: $($Config.TargetFPS) FPS" -ForegroundColor Yellow
Write-Host "[INFO] Druecke Strg+C zum Beenden`n" -ForegroundColor Yellow

$stats = @{
    Captured = 0
    StartTime = Get-Date
    LastStatTime = Get-Date
}

$intervalMs = [int]((1.0 / $Config.TargetFPS) * 1000)

try {
    while ($true) {
        $loopStart = Get-Date
        
        # Capture beide Kameras
        $fileX = $workerX.Capture()
        $fileY = $workerY.Capture()
        
        if ($fileX) {
            $uploadMgr.AddFile($fileX, "X")
            $stats.Captured++
        }
        if ($fileY) {
            $uploadMgr.AddFile($fileY, "Y")
            $stats.Captured++
        }
        
        # Stats alle 5 Sekunden
        $timeSinceStats = (Get-Date) - $stats.LastStatTime
        if ($timeSinceStats.TotalSeconds -ge 5) {
            $runtime = (Get-Date) - $stats.StartTime
            $fps = [math]::Round($stats.Captured / $runtime.TotalSeconds, 2)
            
            Write-Host "[STATS] Captured: $($stats.Captured) | FPS: $fps | Uploaded: $($uploadMgr.UploadedCount) | Queue: X=$($uploadMgr.QueueX.Count) Y=$($uploadMgr.QueueY.Count)" -ForegroundColor Green
            $stats.LastStatTime = Get-Date
        }
        
        # Timing
        $elapsed = ((Get-Date) - $loopStart).TotalMilliseconds
        $sleepMs = [math]::Max(0, $intervalMs - $elapsed)
        if ($sleepMs -gt 0) {
            Start-Sleep -Milliseconds $sleepMs
        }
    }
} catch {
    Write-Host "`n[ERROR] Exception: $_" -ForegroundColor Red
} finally {
    Write-Host "`n[CLEANUP] Beende System..." -ForegroundColor Yellow
    $workerX.Stop()
    $workerY.Stop()
    $uploadMgr.Stop()
    Write-Host "[INFO] System beendet" -ForegroundColor Green
}