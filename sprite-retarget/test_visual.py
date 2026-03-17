"""
Visual test: renders sprite frames using rigid bone cutout rotation.
Each bone segment rotates as a solid piece with small overlap (dilation) at joints.
Saves test_frames/ PNGs and test_contact_sheet.png.
"""
import json, math, os
from PIL import Image, ImageDraw, ImageFilter, ImageChops

# ── Constants ──────────────────────────────────────────────────────────────────
J = dict(HEAD=0,LS=1,RS=2,LE=3,RE=4,LW=5,RW=6,LH=7,RH=8,LK=9,RK=10,LA=11,RA=12,NECK=13,PELVIS=14)
J_NAMES = ['Head','L Shoulder','R Shoulder','L Elbow','R Elbow','L Wrist','R Wrist',
           'L Hip','R Hip','L Knee','R Knee','L Ankle','R Ankle','Neck','Pelvis']
J_COLORS_HEX = ['#ff9f43','#48dbfb','#ff6b9d','#48dbfb','#ff6b9d','#48dbfb','#ff6b9d',
                '#54a0ff','#ee5a24','#54a0ff','#ee5a24','#54a0ff','#ee5a24','#ffd32a','#c8d6e5']

DEFAULT_JOINTS = [
    (0.50,0.08),(0.22,0.33),(0.77,0.33),
    (0.15,0.49),(0.84,0.49),(0.11,0.63),
    (0.87,0.63),(0.36,0.49),(0.60,0.49),
    (0.33,0.78),(0.64,0.78),(0.31,0.92),
    (0.67,0.94),(0.50,0.28),(0.50,0.49),
]

BONE_DEFS = [
    dict(id=0,name='head',     pivot=J['NECK'],  end=J['HEAD']),
    dict(id=1,name='torso',    pivot=J['PELVIS'],end=J['NECK']),
    dict(id=2,name='upperArmL',pivot=J['LS'],    end=J['LE']),
    dict(id=3,name='forearmL', pivot=J['LE'],    end=J['LW']),
    dict(id=4,name='upperArmR',pivot=J['RS'],    end=J['RE']),
    dict(id=5,name='forearmR', pivot=J['RE'],    end=J['RW']),
    dict(id=6,name='thighL',   pivot=J['LH'],    end=J['LK']),
    dict(id=7,name='shinL',    pivot=J['LK'],    end=J['LA']),
    dict(id=8,name='thighR',   pivot=J['RH'],    end=J['RK']),
    dict(id=9,name='shinR',    pivot=J['RK'],    end=J['RA']),
]
BONE_PARENT = [-1,-1,1,2,1,4,1,6,1,8]
# Draw order: legs (back) → torso → head → right arm (back) → left arm (front)
DRAW_ORDER  = [6,7,8,9,1,0,4,5,2,3]
MIRROR_IDX  = [0,2,1,4,3,6,5,8,7,10,9,12,11,13,14]

Y_SCALE      = 0.4   # hip bounce scale (dampened for 2D)
CANVAS_PAD   = 200   # extra pixels around frame so arms don't clip
ANGLE_DAMP   = 0.65  # scale all angle deltas (3D motion too extreme for 2D)
MAX_ANGLE    = 1.2    # max radians (~69°) per bone rotation
JOINT_DILATION = 8    # pixels of overlap at bone boundaries


def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def mirror_frame(frame):
    return [{'x':1-frame[MIRROR_IDX[i]]['x'], 'y':frame[MIRROR_IDX[i]]['y'], 'v':frame[MIRROR_IDX[i]]['v']}
            for i in range(15)]

def bone_angle(joints, p, e):
    return math.atan2(joints[e]['y']-joints[p]['y'], joints[e]['x']-joints[p]['x'])

def angle_delta(ref, frame, p, e):
    d = bone_angle(frame, p, e) - bone_angle(ref, p, e)
    while d >  math.pi: d -= 2*math.pi
    while d < -math.pi: d += 2*math.pi
    d *= ANGLE_DAMP
    d = max(-MAX_ANGLE, min(MAX_ANGLE, d))
    return d

def seg_bone_xy(x, y, sp):
    """Zone-based bone assignment."""
    neck_y = sp[J['NECK']][1]
    ls_x, rs_x = sp[J['LS']][0], sp[J['RS']][0]
    lh_x, rh_x = sp[J['LH']][0], sp[J['RH']][0]

    def seg_dist(ax, ay, bx, by):
        dx, dy = bx-ax, by-ay
        l2 = dx*dx + dy*dy
        if l2 < 1: return math.hypot(x-ax, y-ay)
        t = max(0, min(1, ((x-ax)*dx+(y-ay)*dy)/l2))
        return math.hypot(x-(ax+t*dx), y-(ay+t*dy))

    if y <= neck_y:
        return 0  # head

    torso_left  = min(ls_x, lh_x)
    torso_right = max(rs_x, rh_x)
    in_torso = torso_left <= x <= torso_right
    d_to_ls = math.hypot(x - sp[J['LS']][0], y - sp[J['LS']][1])
    d_to_rs = math.hypot(x - sp[J['RS']][0], y - sp[J['RS']][1])
    near_shoulder = min(d_to_ls, d_to_rs) < 55

    scores = [
        seg_dist(*sp[J['PELVIS']],*sp[J['NECK']]) * (0.6 if (in_torso and not near_shoulder) else 1.0),
        seg_dist(*sp[J['LS']],    *sp[J['LE']]),
        seg_dist(*sp[J['LE']],    *sp[J['LW']]),
        seg_dist(*sp[J['RS']],    *sp[J['RE']]),
        seg_dist(*sp[J['RE']],    *sp[J['RW']]),
        seg_dist(*sp[J['LH']],    *sp[J['LK']]),
        seg_dist(*sp[J['LK']],    *sp[J['LA']]),
        seg_dist(*sp[J['RH']],    *sp[J['RK']]),
        seg_dist(*sp[J['RK']],    *sp[J['RA']]),
    ]
    return [1,2,3,4,5,6,7,8,9][scores.index(min(scores))]

# ── Rigid bone cutout approach ─────────────────────────────────────────────────

def build_bone_images(sprite, sp_px, dilation=JOINT_DILATION):
    """Pre-cut sprite into per-bone RGBA images with dilated masks for joint overlap."""
    W, H = sprite.size
    alpha = sprite.split()[3]
    alpha_data = list(alpha.getdata())

    # Assign every opaque pixel to a bone
    print('  Segmenting pixels into bones...')
    bone_assign = bytearray(W * H)  # bone id per pixel (0-9), 255 = no bone
    for y in range(H):
        for x in range(W):
            if alpha_data[y*W+x] > 10:
                bone_assign[y*W+x] = seg_bone_xy(x, y, sp_px)
            else:
                bone_assign[y*W+x] = 255

    bone_images = {}
    for bone_id in range(10):
        # Create binary mask for this bone
        mask_data = bytearray(W * H)
        for i in range(W * H):
            if bone_assign[i] == bone_id:
                mask_data[i] = 255
        mask = Image.new('L', (W, H))
        mask.putdata(list(mask_data))

        # Dilate to create overlap at joints
        dilated = mask
        for _ in range(dilation):
            dilated = dilated.filter(ImageFilter.MaxFilter(3))

        # Only keep dilated pixels that are within the sprite's opaque area
        final_mask = ImageChops.darker(dilated, alpha)

        # Extract sprite pixels with this mask
        bone_img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        bone_img.paste(sprite, mask=final_mask)
        bone_images[bone_id] = bone_img

    return bone_images


def compute_fk(frame_motion, ref_motion, sp_px, H):
    """Forward kinematics: compute new joint positions and per-bone angle deltas."""
    yD = (frame_motion[J['PELVIS']]['y'] - ref_motion[J['PELVIS']]['y']) * H * Y_SCALE
    nj = [{'x':p[0],'y':p[1]+yD} for p in sp_px]

    def fk(par, child, delta):
        ox = sp_px[child][0]-sp_px[par][0]; oy = sp_px[child][1]-sp_px[par][1]
        c, s = math.cos(delta), math.sin(delta)
        nj[child] = {'x': nj[par]['x']+ox*c-oy*s, 'y': nj[par]['y']+ox*s+oy*c}

    fk(J['PELVIS'],J['NECK'], angle_delta(ref_motion,frame_motion,J['PELVIS'],J['NECK']))
    fk(J['PELVIS'],J['LH'],   angle_delta(ref_motion,frame_motion,J['PELVIS'],J['LH']))
    fk(J['PELVIS'],J['RH'],   angle_delta(ref_motion,frame_motion,J['PELVIS'],J['RH']))
    fk(J['NECK'],  J['HEAD'], angle_delta(ref_motion,frame_motion,J['NECK'],  J['HEAD']))
    # Shoulders: fixed at rest-pose offset from NECK (NECK→LS/RS angle is unstable
    # in 2D projection — both shoulders can appear near the neck causing false +60° swings)
    nj[J['LS']] = {'x': nj[J['NECK']]['x'] + (sp_px[J['LS']][0]-sp_px[J['NECK']][0]),
                   'y': nj[J['NECK']]['y'] + (sp_px[J['LS']][1]-sp_px[J['NECK']][1])}
    nj[J['RS']] = {'x': nj[J['NECK']]['x'] + (sp_px[J['RS']][0]-sp_px[J['NECK']][0]),
                   'y': nj[J['NECK']]['y'] + (sp_px[J['RS']][1]-sp_px[J['NECK']][1])}
    fk(J['LS'],    J['LE'],   angle_delta(ref_motion,frame_motion,J['LS'],    J['LE']))
    fk(J['RS'],    J['RE'],   angle_delta(ref_motion,frame_motion,J['RS'],    J['RE']))
    fk(J['LE'],    J['LW'],   angle_delta(ref_motion,frame_motion,J['LE'],    J['LW']))
    fk(J['RE'],    J['RW'],   angle_delta(ref_motion,frame_motion,J['RE'],    J['RW']))
    fk(J['LH'],    J['LK'],   angle_delta(ref_motion,frame_motion,J['LH'],    J['LK']))
    fk(J['RH'],    J['RK'],   angle_delta(ref_motion,frame_motion,J['RH'],    J['RK']))
    fk(J['LK'],    J['LA'],   angle_delta(ref_motion,frame_motion,J['LK'],    J['LA']))
    fk(J['RK'],    J['RA'],   angle_delta(ref_motion,frame_motion,J['RK'],    J['RA']))

    bone_deltas = [angle_delta(ref_motion, frame_motion, b['pivot'], b['end']) for b in BONE_DEFS]
    return nj, bone_deltas


def render_frame_rigid(bone_images, nj, bone_deltas, sp_px, W, H, pad=CANVAS_PAD):
    """Render one frame by rigidly rotating each bone cutout."""
    OW, OH = W + 2*pad, H + 2*pad
    out = Image.new('RGBA', (OW, OH), (0, 0, 0, 0))

    for bone_id in DRAW_ORDER:
        bd = BONE_DEFS[bone_id]
        angle = bone_deltas[bone_id]
        px, py = sp_px[bd['pivot']]
        npx, npy = nj[bd['pivot']]['x'], nj[bd['pivot']]['y']

        # PIL AFFINE coefficients: maps output (padded canvas) coords → input (sprite) coords
        # Forward: out = R(angle) * (in - orig_pivot) + new_pivot + pad
        # Inverse: in = R(-angle) * (out - new_pivot - pad) + orig_pivot
        ca, sa = math.cos(angle), math.sin(angle)
        # Coefficients for inverse mapping (output → input)
        a_coef = ca
        b_coef = sa
        c_coef = px - ca * (npx + pad) - sa * (npy + pad)
        d_coef = -sa
        e_coef = ca
        f_coef = py + sa * (npx + pad) - ca * (npy + pad)

        rotated = bone_images[bone_id].transform(
            (OW, OH), Image.AFFINE,
            (a_coef, b_coef, c_coef, d_coef, e_coef, f_coef),
            resample=Image.BICUBIC)

        out = Image.alpha_composite(out, rotated)

    return out


def draw_skeleton_overlay(img, nj, pad=CANVAS_PAD):
    draw = ImageDraw.Draw(img, 'RGBA')
    pairs = [(J['NECK'],J['HEAD']),(J['NECK'],J['LS']),(J['NECK'],J['RS']),
             (J['LS'],J['LE']),(J['LE'],J['LW']),(J['RS'],J['RE']),(J['RE'],J['RW']),
             (J['PELVIS'],J['NECK']),(J['PELVIS'],J['LH']),(J['PELVIS'],J['RH']),
             (J['LH'],J['LK']),(J['LK'],J['LA']),(J['RH'],J['RK']),(J['RK'],J['RA'])]
    for a, b in pairs:
        draw.line([(int(nj[a]['x'])+pad, int(nj[a]['y'])+pad),
                   (int(nj[b]['x'])+pad, int(nj[b]['y'])+pad)],
                  fill=(255,255,255,180), width=2)
    for i, jp in enumerate(nj):
        x, y = int(jp['x'])+pad, int(jp['y'])+pad
        c = hex_rgb(J_COLORS_HEX[i]) + (220,)
        draw.ellipse([x-4,y-4,x+4,y+4], fill=c)


def main():
    base = '/home/user/sprite-retarget/sprite-retarget'
    sprite_path = os.path.join(base, 'bodywhole1.png')
    json_path   = os.path.join(base, 'motion_data.json')
    out_dir     = os.path.join(base, 'test_frames')

    if not os.path.exists(sprite_path): print('ERROR: sprite not found'); return
    if not os.path.exists(json_path):   print('ERROR: JSON not found');   return
    os.makedirs(out_dir, exist_ok=True)

    sprite = Image.open(sprite_path).convert('RGBA')
    W, H = sprite.size
    print(f'Sprite: {W}×{H}')

    with open(json_path) as f:
        data = json.load(f)

    fps = data['fps']
    mirrored = [mirror_frame(fr) for fr in data['frames']]
    ref = mirrored[0]
    n_frames = len(mirrored)
    print(f'Pose: {n_frames} frames @ {fps}fps')

    # Sprite joint positions in pixels
    sp_px = [(jx*W, jy*H) for jx, jy in DEFAULT_JOINTS]

    # Build rigid bone cutouts
    print('Building bone cutouts...')
    bone_images = build_bone_images(sprite, sp_px)
    print(f'  Built {len(bone_images)} bone images (dilation={JOINT_DILATION}px)')

    # Render all frames
    print('Rendering frames...')
    FW, FH = W + 2*CANVAS_PAD, H + 2*CANVAS_PAD
    global_bbox = [FW, FH, 0, 0]
    raw_frames = []

    for fi in range(n_frames):
        nj, bone_deltas = compute_fk(mirrored[fi], ref, sp_px, H)
        frame_img = render_frame_rigid(bone_images, nj, bone_deltas, sp_px, W, H)

        bbox = frame_img.getbbox()
        if bbox:
            global_bbox[0] = min(global_bbox[0], bbox[0])
            global_bbox[1] = min(global_bbox[1], bbox[1])
            global_bbox[2] = max(global_bbox[2], bbox[2])
            global_bbox[3] = max(global_bbox[3], bbox[3])

        raw_frames.append((frame_img, nj))
        if fi % 5 == 0: print(f'  Frame {fi+1}/{n_frames}')

    # Crop to global content bounding box
    margin = 10
    crop_x0 = max(0, global_bbox[0] - margin)
    crop_y0 = max(0, global_bbox[1] - margin)
    crop_x1 = min(FW, global_bbox[2] + margin)
    crop_y1 = min(FH, global_bbox[3] + margin)
    cw, ch = crop_x1 - crop_x0, crop_y1 - crop_y0
    print(f'  Global bbox: {cw}×{ch} (cropped from {FW}×{FH})')

    # Save cropped frames
    rendered = []
    for fi, (frame_img, nj) in enumerate(raw_frames):
        draw_skeleton_overlay(frame_img, nj)
        bg = Image.new('RGBA', (FW, FH), (240, 240, 240, 255))
        bg.paste(frame_img, (0, 0), frame_img)
        cropped = bg.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        cropped.save(os.path.join(out_dir, f'frame_{fi:03d}.png'))
        rendered.append(cropped)

    # Contact sheet
    cols = 7
    rows = math.ceil(n_frames / cols)
    tw, th = cw // 3, ch // 3
    sheet = Image.new('RGB', (cols*tw, rows*th), (20, 20, 30))
    for i, fr in enumerate(rendered):
        thumb = fr.resize((tw, th), Image.LANCZOS)
        sheet.paste(thumb, ((i%cols)*tw, (i//cols)*th))
    sheet_path = os.path.join(base, 'test_contact_sheet.png')
    sheet.save(sheet_path)
    print(f'\nSaved {n_frames} frames → {out_dir}/')
    print(f'Contact sheet → {sheet_path}')


if __name__ == '__main__':
    main()
