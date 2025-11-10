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
    
    print("READY")  # Signal an PowerShell
    sys.stdout.flush()
    
    frame_count = 0
    
    # Endlosschleife - wartet auf "CAPTURE" Kommando
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