#!/usr/bin/env python3
"""
walk_cycle.py
Composite walk cycle animation:
  - Spritesheet built from walk1.mp4 (person-centric crops, 742×2000, 8×15 grid)
  - Legs drawn procedurally in side-view from walkdata.json gait data
  - Upper body (above pelvis) taken from spritesheet cells
  - Clean 40-frame loop (frames 43-82) exported as GIF + contact sheet
"""

import cv2, json, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── Paths ─────────────────────────────────────────────────────────────────────
WALKDIR    = '/home/user/sprite-retarget/sprite-retarget/walk anim'
BASE       = '/home/user/sprite-retarget/sprite-retarget'
SS_PATH    = os.path.join(WALKDIR, 'spritesheet_113f_8col__1_.png')
VIDEO_PATH = os.path.join(WALKDIR, 'walk1.mp4')
JSON_PATH  = os.path.join(WALKDIR, 'walkdata.json')
OUT_DIR    = os.path.join(BASE, 'test_frames_walk2')
SHEET_PATH = os.path.join(BASE, 'walk2_contact_sheet.png')
GIF_PATH   = os.path.join(BASE, 'walk2_loop.gif')

# ── Spritesheet layout ────────────────────────────────────────────────────────
SS_W, SS_H   = 742, 2000
SS_COLS      = 8
SS_ROWS      = 15
N_FRAMES     = 113
FPS          = 12
VIDEO_FPS    = 30.0
VIDEO_W      = 1080
VIDEO_H      = 1920
CELL_W       = SS_W / SS_COLS   # 92.75
CELL_H       = SS_H / SS_ROWS   # 133.333...

# ── Joint indices ──────────────────────────────────────────────────────────────
HEAD, LS, RS, LE, RE, LW, RW = 0, 1, 2, 3, 4, 5, 6
LH, RH, LK, RK, LA, RA       = 7, 8, 9, 10, 11, 12
NECK, PELVIS                  = 13, 14

# ── Render constants ───────────────────────────────────────────────────────────
DISPLAY_SCALE  = 4          # scale spritesheet cell by this for display
CANVAS_W       = 400
CANVAS_H       = 560
PELVIS_CX      = CANVAS_W // 2
PELVIS_CY      = 240        # pelvis anchor Y in canvas
LEG_SCALE      = 150        # canvas pixels per normalised unit (torso heights)
LIMB_W         = 16         # limb line width (px)
JOINT_R        = 8          # joint dot radius
GROUND_Y       = PELVIS_CY + 210  # approximate ground line
SHADOW_OFFSET  = 6

BG_COLOR       = (28, 32, 40)
GROUND_COLOR   = (45, 65, 45)

# Leg colours: (fill front, fill back)
LEG_L_FRONT    = (120, 175, 245)
LEG_L_BACK     = (55, 90, 155)
LEG_R_FRONT    = (245, 130, 90)
LEG_R_BACK     = (145, 65, 40)
JOINT_FRONT    = (255, 235, 110)
JOINT_BACK     = (170, 155, 70)

# ── Gait loop window (frame 43-82 inclusive = 40 frames, one full stride) ─────
LOOP_START = 43
LOOP_LEN   = 40


# ── Helpers ───────────────────────────────────────────────────────────────────

def cell_bounds(fi: int):
    """Pixel bounds of cell fi in the spritesheet (fractional cell sizes)."""
    col, row = fi % SS_COLS, fi // SS_COLS
    x0 = round(col * CELL_W)
    y0 = round(row * CELL_H)
    x1 = round((col + 1) * CELL_W)
    y1 = round((row + 1) * CELL_H)
    return x0, y0, x1, y1


def clamp_joint(jt):
    return {'x': max(0.0, min(1.0, jt['x'])),
            'y': max(0.0, min(1.0, jt['y'])),
            'v': jt['v']}


# ── Step 1 : Build spritesheet ────────────────────────────────────────────────

def make_spritesheet(frames):
    """
    Extract 113 person-centric crops from walk1.mp4 and arrange into
    742×2000 spritesheet.  Returns pelvis_y_fracs[fi] – the fraction of
    each cell height at which the pelvis sits (used to cut the upper body).
    """
    print('Building spritesheet from video …')
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    spritesheet    = Image.new('RGB', (SS_W, SS_H), (10, 10, 10))
    pelvis_y_fracs = []

    for fi in range(N_FRAMES):
        # Map JSON frame → video frame index
        vid_fi = min(round(fi * VIDEO_FPS / FPS), total_vid - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_fi)
        ret, vframe = cap.read()
        if not ret:
            print(f'  Warning: could not read video frame {vid_fi}')
            pelvis_y_fracs.append(0.50)
            continue

        fr = frames[fi]

        # Bounding box of high-confidence joints
        valid = [(clamp_joint(fr[i])['x'] * VIDEO_W,
                  clamp_joint(fr[i])['y'] * VIDEO_H)
                 for i in range(15) if fr[i]['v'] > 0.25]
        if not valid:
            valid = [(clamp_joint(fr[i])['x'] * VIDEO_W,
                      clamp_joint(fr[i])['y'] * VIDEO_H)
                     for i in range(15)]

        xs, ys = [p[0] for p in valid], [p[1] for p in valid]
        bx0, bx1 = min(xs), max(xs)
        by0, by1 = min(ys), max(ys)

        # 18 % padding
        pw = (bx1 - bx0) * 0.18 + 10
        ph = (by1 - by0) * 0.18 + 10
        cx = (bx0 + bx1) / 2
        cy = (by0 + by1) / 2
        hw = (bx1 - bx0) / 2 + pw
        hh = (by1 - by0) / 2 + ph

        # Enforce cell aspect ratio so no squishing when resizing
        ar = CELL_W / CELL_H  # 0.695
        if hw / hh > ar:
            hh = hw / ar
        else:
            hw = hh * ar

        # Clamp to video bounds
        cx0 = max(0, int(cx - hw))
        cy0 = max(0, int(cy - hh))
        cx1 = min(VIDEO_W, int(cx + hw))
        cy1 = min(VIDEO_H, int(cy + hh))
        crop_h = cy1 - cy0

        # Pelvis y-fraction within this crop
        py_vid = clamp_joint(fr[PELVIS])['y'] * VIDEO_H
        p_frac = (py_vid - cy0) / crop_h if crop_h > 0 else 0.50
        pelvis_y_fracs.append(max(0.20, min(0.85, p_frac)))

        # Crop, resize, paste into sheet
        cropped = vframe[cy0:cy1, cx0:cx1]
        bx0s, by0s, bx1s, by1s = cell_bounds(fi)
        cw_cell = bx1s - bx0s
        ch_cell = by1s - by0s
        if cropped.size > 0 and cw_cell > 0 and ch_cell > 0:
            scaled  = cv2.resize(cropped, (cw_cell, ch_cell), interpolation=cv2.INTER_AREA)
            rgb     = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
            spritesheet.paste(Image.fromarray(rgb), (bx0s, by0s))

        if fi % 15 == 0:
            print(f'  [{fi+1:3d}/{N_FRAMES}] vid_frame={vid_fi}  pelvis_frac={p_frac:.2f}')

    cap.release()
    spritesheet.save(SS_PATH)
    print(f'  → Saved spritesheet: {SS_PATH}')
    return pelvis_y_fracs


# ── Step 2 : Gait normalisation ───────────────────────────────────────────────

def compute_gait(frames):
    """
    For each frame compute normalised leg-joint positions relative to pelvis,
    scaled by torso height (neck-to-pelvis distance in pixel space).
    Also applies per-frame centering so that L/R leg offsets are symmetric.
    """
    gait = []
    for fi, fr in enumerate(frames):
        def jv(idx):
            jt = clamp_joint(fr[idx])
            return jt['x'], jt['y'], jt['v']

        px, py, _ = jv(PELVIS)
        nx, ny, _ = jv(NECK)
        torso_h = math.hypot((nx - px) * VIDEO_W, (ny - py) * VIDEO_H)
        if torso_h < 20:
            torso_h = 20

        def nrm(idx):
            jx, jy, jv_ = jv(idx)
            rx = (jx - px) * VIDEO_W  / torso_h
            ry = (jy - py) * VIDEO_H  / torso_h
            return [rx, ry, jv_]

        g = {
            'torso_h': torso_h,
            'l_hip':   nrm(LH),  'r_hip':   nrm(RH),
            'l_knee':  nrm(LK),  'r_knee':  nrm(RK),
            'l_ankle': nrm(LA),  'r_ankle': nrm(RA),
        }
        gait.append(g)

    return gait


def smooth_low_conf(gait, win=5):
    """Interpolate joints with v < 0.3 from high-confidence neighbours."""
    n = len(gait)
    for name in ('l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle'):
        for fi in range(n):
            rx, ry, v = gait[fi][name]
            if v >= 0.3:
                continue
            nbx, nby, tot_w = 0.0, 0.0, 0.0
            for di in range(-win, win + 1):
                ni = fi + di
                if ni < 0 or ni >= n or ni == fi:
                    continue
                nx2, ny2, nv = gait[ni][name]
                if nv >= 0.3:
                    w = 1.0 / (abs(di) + 1)
                    nbx += nx2 * w; nby += ny2 * w; tot_w += w
            if tot_w > 0:
                gait[fi][name] = [nbx / tot_w, nby / tot_w, 0.31]
    return gait


def center_leg_joints(gait):
    """
    Per-frame: subtract the mean x of each L/R joint pair so that in side-
    view the two legs are symmetric around the body centre.
    Also applies a gentle temporal smoothing to reduce jitter.
    """
    for g in gait:
        for ln, rn in (('l_hip','r_hip'), ('l_knee','r_knee'), ('l_ankle','r_ankle')):
            lx, ly, lv = g[ln];  rx, ry, rv = g[rn]
            mid = (lx + rx) / 2
            g[ln] = [lx - mid, ly, lv]
            g[rn] = [rx - mid, ry, rv]

    # Light temporal smoothing on x (3-frame boxcar)
    n = len(gait)
    for name in ('l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle'):
        xs = [gait[i][name][0] for i in range(n)]
        xs_s = xs[:]
        for i in range(1, n - 1):
            xs_s[i] = (xs[i-1] + xs[i] + xs[i+1]) / 3
        for i in range(n):
            gait[i][name][0] = xs_s[i]
    return gait


def find_loop(gait):
    """Return (start, length) of the cleanest full-stride loop."""
    sig = np.array([g['l_ankle'][0] - g['r_ankle'][0] for g in gait])
    # Known clean window from analysis: frames 43-82
    return LOOP_START, LOOP_LEN


# ── Step 3 : Rendering ────────────────────────────────────────────────────────

def draw_limb(draw, p1, p2, color, width, alpha=255):
    """Draw a rounded limb segment."""
    c = color + (alpha,)
    draw.line([p1, p2], fill=c, width=width)
    r = width // 2
    for pt in (p1, p2):
        draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=c)


def draw_leg(draw, hip, knee, ankle, fill_col, shadow_col, front=True):
    alpha_f = 255 if front else 180
    alpha_s = 180 if front else 120
    jcol = JOINT_FRONT if front else JOINT_BACK

    # Shadow
    sh = SHADOW_OFFSET
    shadow_hip    = (hip[0]+sh,    hip[1]+sh)
    shadow_knee   = (knee[0]+sh,   knee[1]+sh)
    shadow_ankle  = (ankle[0]+sh,  ankle[1]+sh)
    draw_limb(draw, shadow_hip,   shadow_knee,  (0,0,0), LIMB_W, alpha_s)
    draw_limb(draw, shadow_knee,  shadow_ankle, (0,0,0), LIMB_W, alpha_s)

    # Limb
    draw_limb(draw, hip,  knee,   fill_col, LIMB_W, alpha_f)
    draw_limb(draw, knee, ankle,  shadow_col, LIMB_W, alpha_f)

    # Joint dots
    r = JOINT_R
    draw.ellipse([knee[0]-r, knee[1]-r, knee[0]+r, knee[1]+r],
                 fill=jcol + (alpha_f,))
    if front:
        draw.ellipse([ankle[0]-r, ankle[1]-r, ankle[0]+r, ankle[1]+r],
                     fill=jcol + (255,))


def to_canvas(norm_x, norm_y, clamp_x=True):
    """Convert normalised leg coords → canvas pixels."""
    cx = PELVIS_CX + norm_x * LEG_SCALE
    cy = PELVIS_CY + norm_y * LEG_SCALE
    if clamp_x:
        cx = max(LIMB_W, min(CANVAS_W - LIMB_W, cx))
    return (int(round(cx)), int(round(cy)))


def render_frame(spritesheet, fi_sprite, fi_gait, gait, pelvis_y_fracs):
    """
    Render one composite frame.
      fi_sprite : which spritesheet cell to use for upper body
      fi_gait   : which gait entry to use for leg animation
    """
    g = gait[fi_gait]
    canvas = Image.new('RGBA', (CANVAS_W, CANVAS_H), BG_COLOR + (255,))
    draw   = ImageDraw.Draw(canvas, 'RGBA')

    # Ground line
    draw.rectangle([0, GROUND_Y, CANVAS_W, CANVAS_H],
                   fill=GROUND_COLOR + (255,))
    draw.line([0, GROUND_Y, CANVAS_W, GROUND_Y],
              fill=(80, 120, 80, 255), width=2)

    # ── Leg joint positions ───────────────────────────────────────────────────
    l_hip   = to_canvas(*g['l_hip'][:2])
    r_hip   = to_canvas(*g['r_hip'][:2])
    l_knee  = to_canvas(*g['l_knee'][:2])
    r_knee  = to_canvas(*g['r_knee'][:2])
    l_ankle = to_canvas(*g['l_ankle'][:2])
    r_ankle = to_canvas(*g['r_ankle'][:2])

    # Clamp ankle y to ground
    l_ankle = (l_ankle[0], min(l_ankle[1], GROUND_Y - 2))
    r_ankle = (r_ankle[0], min(r_ankle[1], GROUND_Y - 2))

    # Determine which leg is "front" (higher x in side-view = further forward)
    l_forward = g['l_ankle'][0] >= g['r_ankle'][0]

    # Draw back leg first, then front leg
    if l_forward:
        # Right leg is back
        draw_leg(draw, r_hip, r_knee, r_ankle,
                 LEG_R_BACK, LEG_R_BACK, front=False)
    else:
        draw_leg(draw, l_hip, l_knee, l_ankle,
                 LEG_L_BACK, LEG_L_BACK, front=False)

    # ── Upper body from spritesheet ───────────────────────────────────────────
    bx0, by0, bx1, by1 = cell_bounds(fi_sprite)
    cell_w, cell_h     = bx1 - bx0, by1 - by0
    cell_img           = spritesheet.crop((bx0, by0, bx1, by1))

    disp_w = cell_w * DISPLAY_SCALE
    disp_h = cell_h * DISPLAY_SCALE
    scaled = cell_img.resize((disp_w, disp_h), Image.LANCZOS)

    p_frac  = pelvis_y_fracs[fi_sprite]
    ub_h    = int(p_frac * disp_h)          # pixels from top to pelvis
    ub_img  = scaled.crop((0, 0, disp_w, ub_h)).convert('RGBA')

    # Paste upper body so its bottom edge (pelvis) aligns with PELVIS_CY.
    # Also account for the person's horizontal centre within the cell
    # (approximate: person is roughly centred in each crop).
    ub_x = PELVIS_CX - disp_w // 2
    ub_y = PELVIS_CY - ub_h
    # Clamp so the sprite doesn't overflow canvas
    ub_x = max(-disp_w // 4, min(CANVAS_W - 3 * disp_w // 4, ub_x))
    canvas.paste(ub_img, (ub_x, ub_y), ub_img)

    # ── Front leg (on top of upper body) ─────────────────────────────────────
    if l_forward:
        draw_leg(draw, l_hip, l_knee, l_ankle,
                 LEG_L_FRONT, LEG_L_BACK, front=True)
    else:
        draw_leg(draw, r_hip, r_knee, r_ankle,
                 LEG_R_FRONT, LEG_R_BACK, front=True)

    return canvas.convert('RGB')


# ── Step 4 : Contact sheet & GIF ─────────────────────────────────────────────

def make_contact_sheet(frames_list, path, cols=8):
    n   = len(frames_list)
    rows = math.ceil(n / cols)
    tw, th = frames_list[0].size
    tw //= 3; th //= 3
    sheet = Image.new('RGB', (cols * tw, rows * th), (15, 15, 20))
    for i, fr in enumerate(frames_list):
        thumb = fr.resize((tw, th), Image.LANCZOS)
        sheet.paste(thumb, ((i % cols) * tw, (i // cols) * th))
    sheet.save(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load JSON
    with open(JSON_PATH) as f:
        data = json.load(f)
    frames = data['frames']
    print(f'JSON: {len(frames)} frames @ {data["fps"]}fps')

    # ── 1. Build spritesheet ──────────────────────────────────────────────────
    if os.path.exists(SS_PATH):
        print(f'Spritesheet already exists, loading …')
        spritesheet = Image.open(SS_PATH).convert('RGB')
        # Recompute pelvis fracs from JSON (no video needed)
        pelvis_y_fracs = []
        for fi, fr in enumerate(frames):
            px = clamp_joint(fr[PELVIS])['x']
            py = clamp_joint(fr[PELVIS])['y']
            nx = clamp_joint(fr[NECK])['x']
            ny = clamp_joint(fr[NECK])['y']
            # In the person-centric crop the pelvis fraction was stored during build.
            # Approximate from joint bounding box:
            valid = [(clamp_joint(fr[i])['x'], clamp_joint(fr[i])['y'])
                     for i in range(15) if fr[i]['v'] > 0.25]
            if not valid:
                valid = [(clamp_joint(fr[i])['x'], clamp_joint(fr[i])['y'])
                         for i in range(15)]
            min_y = min(p[1] for p in valid)
            max_y = max(p[1] for p in valid)
            span  = max(max_y - min_y, 0.01)
            pad   = span * 0.18 + 10 / VIDEO_H
            by0   = max(0.0, min_y - pad)
            by1   = min(1.0, max_y + pad)
            crop_span = by1 - by0
            p_frac = (py - by0) / crop_span if crop_span > 0 else 0.5
            pelvis_y_fracs.append(max(0.20, min(0.85, p_frac)))
    else:
        pelvis_y_fracs = make_spritesheet(frames)
        spritesheet    = Image.open(SS_PATH).convert('RGB')

    print(f'Spritesheet: {spritesheet.size}')
    print(f'Pelvis fracs sample: {[round(p,2) for p in pelvis_y_fracs[:8]]}')

    # ── 2. Gait analysis ──────────────────────────────────────────────────────
    print('Computing gait data …')
    gait = compute_gait(frames)
    gait = smooth_low_conf(gait)
    gait = center_leg_joints(gait)

    loop_start, loop_len = find_loop(gait)
    loop_end = loop_start + loop_len
    print(f'Gait loop: frames {loop_start}–{loop_end-1} ({loop_len} frames @ {FPS}fps = {loop_len/FPS:.1f}s)')

    # Print stride signal for loop window
    stride = [gait[i]['l_ankle'][0] - gait[i]['r_ankle'][0]
              for i in range(loop_start, loop_end)]
    print(f'Stride range in loop: [{min(stride):.3f}, {max(stride):.3f}]')

    # ── 3. Render loop frames ─────────────────────────────────────────────────
    print('Rendering frames …')
    rendered = []
    for li in range(loop_len):
        fi_gait   = loop_start + li      # gait frame
        fi_sprite = loop_start + li      # same frame from spritesheet
        img = render_frame(spritesheet, fi_sprite, fi_gait, gait, pelvis_y_fracs)
        path = os.path.join(OUT_DIR, f'walk2_{li:03d}.png')
        img.save(path)
        rendered.append(img)
        if li % 8 == 0:
            print(f'  [{li+1}/{loop_len}]')

    # ── 4. Contact sheet ──────────────────────────────────────────────────────
    make_contact_sheet(rendered, SHEET_PATH, cols=8)
    print(f'Contact sheet → {SHEET_PATH}')

    # ── 5. GIF loop ───────────────────────────────────────────────────────────
    duration_ms = int(1000 / FPS)
    gif_frames  = [img.convert('P', palette=Image.ADAPTIVE, colors=128)
                   for img in rendered]
    gif_frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(f'GIF loop     → {GIF_PATH}  ({loop_len} frames, {duration_ms}ms/frame)')
    print('Done.')


if __name__ == '__main__':
    main()
