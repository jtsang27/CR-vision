from ultralytics import YOLO
import cv2, os, json, time, numpy as np
from pathlib import Path

MODEL  = "cr_vision/v1/weights/best.pt"   # adjust if needed
SOURCE = "/gameplay.mp4"                     # folder or a .mp4
W, H   = 32, 18                            # arena grid size

# If you've calibrated a homography H (3x3), load it here.
Hmat = None  # np.load("homography.npy")  # set later

def px_to_arena(cx, cy, fw, fh):
    if Hmat is None:
        # simple normalize to grid (placeholder until you use homography)
        gx = (cx / fw) * W
        gy = (cy / fh) * H
        return float(gx), float(gy)
    v = np.array([cx, cy, 1.0])
    u = Hmat @ v
    return float(u[0]/u[2]), float(u[1]/u[2])

def owner_from_y(gy):
    return "ALLY" if gy > H/2 else "ENEMY"

def run_on_image_dir(model, folder):
    out = open("gsi.jsonl", "w")
    imgs = sorted([p for p in Path(folder).glob("*.jpg")] + [p for p in Path(folder).glob("*.png")])
    for p in imgs:
        frame = cv2.imread(str(p))
        fh, fw = frame.shape[:2]
        res = model.predict(frame, verbose=False, conf=0.25)[0]
        towers, units = [], []
        for b in res.boxes:
            cls = int(b.cls[0].item()); conf = float(b.conf[0].item())
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            cx, cy = (x1+x2)/2, (y1+y2)/2
            gx, gy = px_to_arena(cx, cy, fw, fh)
            name = res.names[cls]
            # simple rules; adjust for your class set
            if "Tower" in name:
                ttype = "king" if "King" in name else "princess"
                towers.append({"owner": owner_from_y(gy), "type": ttype,
                               "x": gx, "y": gy, "hp_frac": 1.0, "conf": conf})
            elif name.endswith("_Troop") or "Troop" in name:
                units.append({"owner": owner_from_y(gy), "coarse_type":"troop",
                              "x": gx, "y": gy, "vx": 0.0, "vy": 0.0, "conf": conf})
        gsi = {
            "t_ms": int(time.time()*1000),
            "source": "vision",
            "arena_size": {"W": W, "H": H},
            "towers": towers,
            "units": units,
            "resources": {"ally_elixir": {"value": None, "conf": 0.0}}
        }
        out.write(json.dumps(gsi) + "\n")
    out.close()
    print("Wrote gsi.jsonl")

def run_on_video(model, path):
    out = open("gsi.jsonl", "w")
    cap = cv2.VideoCapture(path)
    while True:
        ok, frame = cap.read()
        if not ok: break
        fh, fw = frame.shape[:2]
        res = model.predict(frame, verbose=False, conf=0.25)[0]
        towers, units = [], []
        for b in res.boxes:
            cls = int(b.cls[0].item()); conf = float(b.conf[0].item())
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            cx, cy = (x1+x2)/2, (y1+y2)/2
            gx, gy = px_to_arena(cx, cy, fw, fh)
            name = res.names[cls]
            if "Tower" in name:
                ttype = "king" if "King" in name else "princess"
                towers.append({"owner": owner_from_y(gy), "type": ttype,
                               "x": gx, "y": gy, "hp_frac": 1.0, "conf": conf})
            elif name.endswith("_Troop") or "Troop" in name:
                units.append({"owner": owner_from_y(gy), "coarse_type":"troop",
                              "x": gx, "y": gy, "vx": 0.0, "vy": 0.0, "conf": conf})
        gsi = {
            "t_ms": int(time.time()*1000),
            "source": "vision",
            "arena_size": {"W": W, "H": H},
            "towers": towers,
            "units": units,
            "resources": {"ally_elixir": {"value": None, "conf": 0.0}}
        }
        out.write(json.dumps(gsi) + "\n")
    cap.release()
    out.close()
    print("Wrote gsi.jsonl")

if __name__ == "__main__":
    model = YOLO(MODEL)
    if SOURCE.lower().endswith(".mp4"):
        run_on_video(model, SOURCE)
    else:
        run_on_image_dir(model, SOURCE)