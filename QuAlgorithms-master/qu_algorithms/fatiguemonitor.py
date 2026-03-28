import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from collections import deque
import time
import threading
import sounddevice as sd

# ═══════════════════════════════════════════════════════════════
# SOUND
# ═══════════════════════════════════════════════════════════════
def play_beep(frequency=1000, duration=0.5, volume=0.3):
    sr   = 44100
    t    = np.linspace(0, duration, int(sr * duration), False)
    wave = (volume * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    sd.play(wave, sr)
    sd.wait()

def beep1():
    threading.Thread(target=play_beep,
                     kwargs=dict(frequency=1200, duration=0.4, volume=0.3),
                     daemon=True).start()

def beep2():
    threading.Thread(target=play_beep,
                     kwargs=dict(frequency=900, duration=1.8, volume=0.90),
                     daemon=True).start()

# ═══════════════════════════════════════════════════════════════
# MEDIAPIPE
# ═══════════════════════════════════════════════════════════════
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

TESS_SPEC = mp_draw.DrawingSpec(color=(0, 200, 180), thickness=1, circle_radius=0)
CONT_SPEC = mp_draw.DrawingSpec(color=(0, 255, 200), thickness=1, circle_radius=1)

# ═══════════════════════════════════════════════════════════════
# LANDMARK INDICES
# ═══════════════════════════════════════════════════════════════
LEFT_EYE     = [362, 385, 387, 263, 373, 380]
RIGHT_EYE    = [33,  160, 158, 133, 153, 144]
MOUTH_TOP    = 13    # upper inner lip center
MOUTH_BOTTOM = 14    # lower inner lip center
MOUTH_LEFT   = 61    # left outer mouth corner
MOUTH_RIGHT  = 291   # right outer mouth corner
NOSE_TIP     = 4     # nose tip (for pitch estimation)
FOREHEAD     = 10    # forehead center
CHIN         = 152   # chin center

# ═══════════════════════════════════════════════════════════════
# THRESHOLDS  (tune these if needed for your face/lighting)
# ═══════════════════════════════════════════════════════════════
EAR_THRESH          = 0.22   # below = eye closing
ALERT1_SECS         = 3.0    # first warning
ALERT2_SECS         = 6.0    # critical alert
YAWN_MAR_THRESH     = 0.55   # mouth open ratio to qualify as yawn
YAWN_SQUINT_THRESH  = 0.35   # EAR must drop below this simultaneously (squint check)
YAWN_MIN_SECS       = 1.3    # mouth must stay open this long to count
NOD_FWD_THRESH      = 0.60   # pitch ratio above this = head tilting forward
NOD_RETURN_THRESH   = 0.50   # pitch ratio below this = head returned upright
NOD_SUSTAIN_SECS    = 3.0    # seconds head stays down to count as sustained nod
NOD_ACTIVE_COUNT    = 3      # number of dips in window to count as active nodding
NOD_ACTIVE_WINDOW   = 8.0    # time window (s) for counting active nod dips
BLINK_WIN_SECS      = 5.0    # blink rate sampling window
BLINK_HIGH_THRESH   = 2.5    # blinks/5s above this = anomaly (>30/min, fatigued)
BLINK_LOW_THRESH    = 0.4    # blinks/5s below this = staring anomaly (<5/min)
RATIO_WIN_SECS      = 10.0   # eye open/closed ratio window
RATIO_OPEN_MIN      = 0.80   # eyes should be open >80% of the time
RISK_UPDATE_SECS    = 10.0   # how often risk score updates  (was 60 — too slow)
ACCUM_RESET_SECS    = 60.0   # how often per-minute accumulators reset

# ═══════════════════════════════════════════════════════════════
# STATE  (single dict avoids all global keyword issues)
# ═══════════════════════════════════════════════════════════════
_t0 = time.time()
S = dict(
    # ── Eyes closed / alert ────────────────────────────────
    blink_count         = 0,
    eyes_closed_start   = None,
    alert1_fired        = False,
    alert2_fired        = False,
    resumed             = False,
    resumed_time        = None,

    # ── Yawn ───────────────────────────────────────────────
    yawn_count          = 0,
    yawn_start          = None,
    yawn_active         = False,

    # ── Blink rate (5-second windows) ─────────────────────
    blink_win_start     = _t0,
    blink_win_count     = 0,
    blink_rate_hist     = deque(maxlen=12),  # up to 12 windows/minute

    # ── Head nod ───────────────────────────────────────────
    nod_count           = 0,
    nod_sus_start       = None,             # start of sustained forward tilt
    nod_sus_fired       = False,            # already counted this sustained event?
    nod_dip_times       = deque(maxlen=10), # timestamps of completed dips

    # ── Eyes open/closed ratio (10-second windows) ────────
    ratio_win_start     = _t0,
    ratio_open_frames   = 0,
    ratio_total_frames  = 0,
    ratio_hist          = deque(maxlen=6),  # last 6 windows = 60 seconds

    # ── Risk score ─────────────────────────────────────────
    risk_score          = 0.0,
    risk_last_update    = _t0,

    # ── Per-minute accumulators (reset every 60s) ─────────
    min_yawns           = 0,
    min_blink_anomalies = 0,
    min_nods            = 0,
    min_alert1          = 0,
    min_alert2          = 0,
    min_reset_time      = _t0,              # separate timer for accumulator reset
)

# ═══════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════
def compute_ear(pts):
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def compute_mar(lm, fw, fh):
    top    = np.array([lm[MOUTH_TOP].x    * fw, lm[MOUTH_TOP].y    * fh])
    bottom = np.array([lm[MOUTH_BOTTOM].x * fw, lm[MOUTH_BOTTOM].y * fh])
    left   = np.array([lm[MOUTH_LEFT].x   * fw, lm[MOUTH_LEFT].y   * fh])
    right  = np.array([lm[MOUTH_RIGHT].x  * fw, lm[MOUTH_RIGHT].y  * fh])
    vert   = distance.euclidean(top, bottom)
    horiz  = distance.euclidean(left, right)
    return vert / (horiz + 1e-6)

def get_pitch_ratio(lm):
    """
    Returns nose-tip y position normalized between forehead (0) and chin (1).
    Straight ahead  ≈ 0.45
    Nodding forward > 0.60  (nose moves toward chin in screen space)
    """
    ny   = lm[NOSE_TIP].y
    fy   = lm[FOREHEAD].y
    cy   = lm[CHIN].y
    span = cy - fy
    return (ny - fy) / span if abs(span) > 1e-6 else 0.5

def lm_pts(lm, indices, fw, fh):
    return [(lm[i].x * fw, lm[i].y * fh) for i in indices]

# ═══════════════════════════════════════════════════════════════
# RISK COMPUTATION  (FIXED)
# ═══════════════════════════════════════════════════════════════
def compute_risk_delta():

    yawn_s  = min(S['min_yawns']           / 2.0, 1.0)
    blink_s = min(S['min_blink_anomalies'] / 6.0, 1.0)
    nod_s   = min(S['min_nods']            / 3.0, 1.0)
    a1_s    = min(S['min_alert1']          / 3.0, 1.0)
    a2_s    = min(S['min_alert2']          / 2.0, 1.0)

    if len(S['ratio_hist']) > 0:
        avg_r   = sum(S['ratio_hist']) / len(S['ratio_hist'])
        ratio_s = max(0.0, (RATIO_OPEN_MIN - avg_r) / RATIO_OPEN_MIN)
    else:
        ratio_s = 0.0

    # Weighted badness (same weights as before, sum = 1.0)
    badness = (
        yawn_s  * 0.15 +   # 
        blink_s * 0.03 +   #
        nod_s   * 0.30 +   # 
        ratio_s * 0.07 +   # 
        a1_s    * 0.10 +   #
        a2_s    * 0.35     # 
    )

    # ── New responsive mapping ────────────────────────────
    # Clean window (no signals at all) → slowly recover
    if badness < 0.8:
        return -2.0

    # Any fatigue signal → proportional increase
    # badness 0.10 → +2,  badness 0.25 → +5,  badness 1.0 → +20
    return badness * 20.0

def reset_minute():
    for k in ('min_yawns', 'min_blink_anomalies', 'min_nods', 'min_alert1', 'min_alert2'):
        S[k] = 0

# ═══════════════════════════════════════════════════════════════
# HUD DRAWING
# ═══════════════════════════════════════════════════════════════
def draw_hud(frame, ear_val, closed_secs, alert_level):
    h, w = frame.shape[:2]
    now  = time.time()

    # ── Top bar ───────────────────────────────────────────────
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 95), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # Title (left)
    cv2.putText(frame, "F1 DRIVER FATIGUE MONITOR",
                (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 30, 30), 2)

    # Risk score (right)
    risk = S['risk_score']
    if   risk < 30: rcol = (0, 220, 0)
    elif risk < 60: rcol = (0, 165, 255)
    else:           rcol = (0, 50,  255)
    cv2.putText(frame, f"RISK: {risk:.0f}%",
                (w - 190, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, rcol, 2)

    # Stats row
    ear_col = (0, 255, 0) if ear_val >= EAR_THRESH else (0, 80, 255)
    row = [
        (20,  f"EAR:{ear_val:.2f}",         ear_col),
        (148, f"BLINKS:{S['blink_count']}",  (255, 255, 255)),
        (298, f"YAWNS:{S['yawn_count']}",    (255, 200,   0)),
        (435, f"NODS:{S['nod_count']}",      (200, 180, 255)),
    ]
    if closed_secs > 0:
        row.append((565, f"CLOSED:{closed_secs:.1f}s", (0, 80, 255)))
    for x, txt, col in row:
        cv2.putText(frame, txt, (x, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 1)

    # ── Bottom-left info panel ────────────────────────────────
    pnl = frame.copy()
    cv2.rectangle(pnl, (0, h - 85), (310, h), (0, 0, 0), -1)
    cv2.addWeighted(pnl, 0.55, frame, 0.45, 0, frame)

    # ── Blink rate: now displayed as blinks per 10 seconds ────
    if len(S['blink_rate_hist']) > 0:
        b10s = S['blink_rate_hist'][-1] * (10.0 / BLINK_WIN_SECS)
        # Normal: ~2–4 per 10s (12–24/min).  Fatigued: >5 per 10s (>30/min)
        bcol = (0, 255, 0) if 1.3 <= b10s <= 4.2 else (0, 165, 255)
        cv2.putText(frame, f"BLINK RATE:  {b10s:.1f}/10s",
                    (12, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bcol, 1)

    if len(S['ratio_hist']) > 0:
        rat  = S['ratio_hist'][-1]
        rcl  = (0, 255, 0) if rat >= RATIO_OPEN_MIN else (0, 165, 255)
        cv2.putText(frame, f"EYE OPEN:    {rat * 100:.0f}%",
                    (12, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rcl, 1)

    # ── Yawn indicator ────────────────────────────────────────
    if S['yawn_active']:
        cv2.putText(frame, "YAWNING...",
                    (w // 2 - 105, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 220, 255), 2)

    # ── Alert 1 ────────────────────────────────────────────────
    if alert_level == 1:
        bot = frame.copy()
        cv2.rectangle(bot, (0, h - 60), (w, h), (0, 100, 220), -1)
        cv2.addWeighted(bot, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "!! DRIVER ALERT  -  EYES CLOSING !!",
                    (w // 2 - 280, h - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)

    # ── Alert 2 — pulsing full-screen takeover ────────────────
    if alert_level == 2:
        pulse = 0.35 + 0.20 * abs(np.sin(now * 6))
        red   = frame.copy()
        cv2.rectangle(red, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(red, pulse, frame, 1 - pulse, 0, frame)
        for y_off, txt, scale, col in [
            (h // 2 - 90, "!! CRITICAL !!",          1.7, (0,   0, 255)),
            (h // 2 - 10, "DRIVER UNRESPONSIVE",      1.2, (255, 255, 255)),
            (h // 2 + 65, "ACTIVATING SELF-DRIVING",  1.1, (0, 255, 230)),
        ]:
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
            tx  = w // 2 - tw // 2
            pad = 12
            cv2.rectangle(frame,
                          (tx - pad, y_off - th - pad),
                          (tx + tw + pad, y_off + pad),
                          (0, 0, 0), -1)
            cv2.putText(frame, txt, (tx, y_off),
                        cv2.FONT_HERSHEY_DUPLEX, scale, col, 3)

    # ── Driver resumed banner ─────────────────────────────────
    if S['resumed'] and S['resumed_time']:
        age = now - S['resumed_time']
        if age < 3.0:
            alpha = 1.0 - (age / 3.0)
            grn   = frame.copy()
            cy_   = h // 2
            cv2.rectangle(grn, (0, cy_ - 45), (w, cy_ + 45), (0, 120, 0), -1)
            cv2.addWeighted(grn, alpha * 0.75, frame, 1 - alpha * 0.75, 0, frame)
            cv2.putText(frame, "DRIVER RESUMED CONTROL",
                        (w // 2 - 240, cy_ + 14),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 80), 2)
        else:
            S['resumed'] = False

# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WIN = "F1 Driver Fatigue Monitor"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)
print("F1 Fatigue Monitor running — press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    now     = time.time()
    ear_val = 0.30

    # ════════════════════════════════════════════════════════
    # FACE DETECTED
    # ════════════════════════════════════════════════════════
    if results.multi_face_landmarks:
        face_lm = results.multi_face_landmarks[0]
        lm      = face_lm.landmark
        fh, fw  = frame.shape[:2]

        # Draw full mesh
        mp_draw.draw_landmarks(frame, face_lm, mp_face.FACEMESH_TESSELATION,
                               landmark_drawing_spec=None,
                               connection_drawing_spec=TESS_SPEC)
        mp_draw.draw_landmarks(frame, face_lm, mp_face.FACEMESH_CONTOURS,
                               landmark_drawing_spec=None,
                               connection_drawing_spec=CONT_SPEC)

        # Measurements
        ear_val = (compute_ear(lm_pts(lm, LEFT_EYE,  fw, fh)) +
                   compute_ear(lm_pts(lm, RIGHT_EYE, fw, fh))) / 2.0
        mar_val = compute_mar(lm, fw, fh)
        pitch   = get_pitch_ratio(lm)

        # ── Ratio window: track every frame ───────────────
        S['ratio_total_frames'] += 1
        if ear_val >= EAR_THRESH:
            S['ratio_open_frames'] += 1

        # ── Eyes-closed / alert timer ─────────────────────
        if ear_val < EAR_THRESH:
            if S['eyes_closed_start'] is None:
                S['eyes_closed_start'] = now
        else:
            if S['eyes_closed_start'] is not None:
                dur = now - S['eyes_closed_start']
                if dur >= 0.1:                          # filter out noise
                    S['blink_count']     += 1
                    S['blink_win_count'] += 1
                if S['alert1_fired'] or S['alert2_fired']:
                    S['resumed']      = True
                    S['resumed_time'] = now
                S['eyes_closed_start'] = None
                S['alert1_fired']      = False
                S['alert2_fired']      = False

        # ── Yawn detection ────────────────────────────────
        # Condition: mouth wide open (MAR) AND eyes squinting (EAR drops)
        yawning = mar_val > YAWN_MAR_THRESH and ear_val < YAWN_SQUINT_THRESH
        if yawning:
            if S['yawn_start'] is None:
                S['yawn_start'] = now
            S['yawn_active'] = True
        else:
            if S['yawn_start'] is not None:
                if now - S['yawn_start'] >= YAWN_MIN_SECS:
                    S['yawn_count'] += 1
                    S['min_yawns']  += 1
                S['yawn_start']  = None
                S['yawn_active'] = False

        # ── Head nod detection ────────────────────────────
        if pitch > NOD_FWD_THRESH:
            if S['nod_sus_start'] is None:
                S['nod_sus_start'] = now
                S['nod_sus_fired'] = False
            elif (not S['nod_sus_fired'] and
                  now - S['nod_sus_start'] >= NOD_SUSTAIN_SECS):
                S['nod_count']    += 1
                S['min_nods']     += 1
                S['nod_sus_fired'] = True
        else:
            if S['nod_sus_start'] is not None:
                if pitch < NOD_RETURN_THRESH:
                    S['nod_dip_times'].append(now)
                    recent = [t for t in S['nod_dip_times']
                              if now - t < NOD_ACTIVE_WINDOW]
                    if len(recent) >= NOD_ACTIVE_COUNT:
                        S['nod_count'] += 1
                        S['min_nods']  += 1
                        S['nod_dip_times'].clear()
            S['nod_sus_start'] = None
            S['nod_sus_fired'] = False

    # ════════════════════════════════════════════════════════
    # FACE NOT DETECTED  — reset all active timers
    # ════════════════════════════════════════════════════════
    else:
        if S['eyes_closed_start'] is not None:
            S['eyes_closed_start'] = None
            S['alert1_fired']      = False
            S['alert2_fired']      = False
        S['yawn_start']    = None
        S['yawn_active']   = False
        S['nod_sus_start'] = None
        S['nod_sus_fired'] = False
        S['ratio_total_frames'] += 1   # unknown = not open

    # ════════════════════════════════════════════════════════
    # PERIODIC WINDOW UPDATES
    # ════════════════════════════════════════════════════════

    # 5-second blink rate window
    if now - S['blink_win_start'] >= BLINK_WIN_SECS:
        rate = S['blink_win_count']
        S['blink_rate_hist'].append(rate)
        if rate > BLINK_HIGH_THRESH or rate < BLINK_LOW_THRESH:
            S['min_blink_anomalies'] += 1
        S['blink_win_count'] = 0
        S['blink_win_start'] = now

    # 10-second eyes-open ratio window
    if now - S['ratio_win_start'] >= RATIO_WIN_SECS:
        if S['ratio_total_frames'] > 0:
            S['ratio_hist'].append(
                S['ratio_open_frames'] / S['ratio_total_frames']
            )
        S['ratio_open_frames']  = 0
        S['ratio_total_frames'] = 0
        S['ratio_win_start']    = now

    # ── Risk score update (every 10s) ─────────────────────
    if now - S['risk_last_update'] >= RISK_UPDATE_SECS:
        delta           = compute_risk_delta()
        S['risk_score'] = max(0.0, min(100.0, S['risk_score'] + delta))
        S['risk_last_update'] = now

    # ── Per-minute accumulator reset (every 60s, separate) ─
    if now - S['min_reset_time'] >= ACCUM_RESET_SECS:
        reset_minute()
        S['min_reset_time'] = now

    # ════════════════════════════════════════════════════════
    # ALERT TRIGGER
    # ════════════════════════════════════════════════════════
    closed_secs = (now - S['eyes_closed_start']) if S['eyes_closed_start'] else 0.0
    alert_level = 0

    if closed_secs >= ALERT2_SECS:
        alert_level = 2
        if not S['alert2_fired']:
            S['alert2_fired'] = True
            S['min_alert2']  += 1
            beep2()
    elif closed_secs >= ALERT1_SECS:
        alert_level = 1
        if not S['alert1_fired']:
            S['alert1_fired'] = True
            S['min_alert1']  += 1
            beep1()

    draw_hud(frame, ear_val, closed_secs, alert_level)
    cv2.imshow(WIN, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()