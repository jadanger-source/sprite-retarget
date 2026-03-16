"""
Visual test for sprite animator FK math.
Loads sprite PNG + motion_data.json, runs FK, draws skeleton overlay on each frame,
saves test_frames/ as PNGs. Also saves a contact sheet.
"""
import json, math, os
from PIL import Image, ImageDraw, ImageFont

# ── Constants ──────────────────────────────────────────────────────────────────
J = dict(HEAD=0,LS=1,RS=2,LE=3,RE=4,LW=5,RW=6,LH=7,RH=8,LK=9,RK=10,LA=11,RA=12,NECK=13,PELVIS=14)
J_NAMES = ['Head','L Shoulder','R Shoulder','L Elbow','R Elbow','L Wrist','R Wrist',
           'L Hip','R Hip','L Knee','R Knee','L Ankle','R Ankle','Neck','Pelvis']

DEFAULT_JOINTS = [
    (0.50,0.08),(0.22,0.33),(0.77,0.33),
    (0.15,0.49),(0.84,0.49),(0.11,0.63),
    (0.87,0.63),(0.36,0.49),(0.60,0.49),
    (0.33,0.78),(0.64,0.78),(0.31,0.92),
    (0.67,0.94),(0.50,0.28),(0.50,0.49),
]

# Mirror: swap L/R to fix MediaPipe handedness, flip x->1-x
MIRROR_IDX = [0,2,1,4,3,6,5,8,7,10,9,12,11,13,14]

BONE_PAIRS = [
    (J['NECK'],J['HEAD']),(J['NECK'],J['LS']),(J['NECK'],J['RS']),
    (J['LS'],J['LE']),(J['LE'],J['LW']),(J['RS'],J['RE']),(J['RE'],J['RW']),
    (J['PELVIS'],J['NECK']),(J['PELVIS'],J['LH']),(J['PELVIS'],J['RH']),
    (J['LH'],J['LK']),(J['LK'],J['LA']),(J['RH'],J['RK']),(J['RK'],J['RA']),
]

J_COLORS = [
    '#ff9f43','#48dbfb','#ff6b9d','#48dbfb','#ff6b9d','#48dbfb','#ff6b9d',
    '#54a0ff','#ee5a24','#54a0ff','#ee5a24','#54a0ff','#ee5a24','#ffd32a','#c8d6e5'
]

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def mirror_frame(frame):
    return [{'x': 1 - frame[MIRROR_IDX[i]]['x'],
             'y': frame[MIRROR_IDX[i]]['y'],
             'v': frame[MIRROR_IDX[i]]['v']} for i in range(15)]

def bone_angle(joints, p, e):
    return math.atan2(joints[e]['y'] - joints[p]['y'], joints[e]['x'] - joints[p]['x'])

def angle_delta(ref, frame, p, e):
    d = bone_angle(frame, p, e) - bone_angle(ref, p, e)
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d

def compute_new_joints(sprite_joints_px, ref_motion, frame_motion, W, H, y_scale=1.0):
    sp = sprite_joints_px  # list of (x,y) in pixels

    y_delta = (frame_motion[J['PELVIS']]['y'] - ref_motion[J['PELVIS']]['y']) * H * y_scale

    # Initialize new joints at sprite positions + global Y
    nj = [{'x': p[0], 'y': p[1] + y_delta} for p in sp]

    def fk(parent, child, delta):
        ox = sp[child][0] - sp[parent][0]
        oy = sp[child][1] - sp[parent][1]
        cos_d, sin_d = math.cos(delta), math.sin(delta)
        nj[child] = {
            'x': nj[parent]['x'] + ox * cos_d - oy * sin_d,
            'y': nj[parent]['y'] + ox * sin_d + oy * cos_d,
        }

    fk(J['PELVIS'], J['NECK'],  angle_delta(ref_motion, frame_motion, J['PELVIS'], J['NECK']))
    fk(J['PELVIS'], J['LH'],    angle_delta(ref_motion, frame_motion, J['PELVIS'], J['LH']))
    fk(J['PELVIS'], J['RH'],    angle_delta(ref_motion, frame_motion, J['PELVIS'], J['RH']))
    fk(J['NECK'],   J['HEAD'],  angle_delta(ref_motion, frame_motion, J['NECK'],   J['HEAD']))
    fk(J['NECK'],   J['LS'],    angle_delta(ref_motion, frame_motion, J['NECK'],   J['LS']))
    fk(J['NECK'],   J['RS'],    angle_delta(ref_motion, frame_motion, J['NECK'],   J['RS']))
    fk(J['LS'],     J['LE'],    angle_delta(ref_motion, frame_motion, J['LS'],     J['LE']))
    fk(J['RS'],     J['RE'],    angle_delta(ref_motion, frame_motion, J['RS'],     J['RE']))
    fk(J['LE'],     J['LW'],    angle_delta(ref_motion, frame_motion, J['LE'],     J['LW']))
    fk(J['RE'],     J['RW'],    angle_delta(ref_motion, frame_motion, J['RE'],     J['RW']))
    fk(J['LH'],     J['LK'],    angle_delta(ref_motion, frame_motion, J['LH'],     J['LK']))
    fk(J['RH'],     J['RK'],    angle_delta(ref_motion, frame_motion, J['RH'],     J['RK']))
    fk(J['LK'],     J['LA'],    angle_delta(ref_motion, frame_motion, J['LK'],     J['LA']))
    fk(J['RK'],     J['RA'],    angle_delta(ref_motion, frame_motion, J['RK'],     J['RA']))

    return nj

def draw_skeleton(img, joints, alpha=200):
    draw = ImageDraw.Draw(img, 'RGBA')
    W, H = img.size

    def px(j):
        return (int(joints[j]['x']), int(joints[j]['y']))

    # Draw bones
    for (a, b) in BONE_PAIRS:
        pa, pb = px(a), px(b)
        draw.line([pa, pb], fill=(255, 255, 255, alpha), width=2)

    # Draw joint dots
    for i, jp in enumerate(joints):
        x, y = int(jp['x']), int(jp['y'])
        r = 5
        col = hex_to_rgb(J_COLORS[i]) + (alpha,)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=col, outline=(255,255,255,180))

    return img

def main():
    sprite_path = 'bodywhole1.png'
    json_path   = 'motion_data.json'
    out_dir     = 'test_frames'

    if not os.path.exists(sprite_path):
        print(f'ERROR: {sprite_path} not found'); return
    if not os.path.exists(json_path):
        print(f'ERROR: {json_path} not found'); return

    os.makedirs(out_dir, exist_ok=True)

    sprite = Image.open(sprite_path).convert('RGBA')
    W, H = sprite.size
    print(f'Sprite: {W}×{H}')

    with open(json_path) as f:
        data = json.load(f)

    fps = data['fps']
    raw_frames = data['frames']
    mirrored = [mirror_frame(fr) for fr in raw_frames]
    ref = mirrored[0]

    print(f'Pose: {len(mirrored)} frames @ {fps}fps')

    # Sprite joint positions in pixels
    sp = [(jx * W, jy * H) for (jx, jy) in DEFAULT_JOINTS]

    # Verify rest pose (frame 0 should show minimal movement)
    print('\nRest pose joint positions (frame 0 FK):')
    nj0 = compute_new_joints(sp, ref, ref, W, H)
    for i, jp in enumerate(nj0):
        sx, sy = sp[i]
        print(f'  {J_NAMES[i]:12s}: sprite=({sx:.0f},{sy:.0f}) -> FK=({jp["x"]:.0f},{jp["y"]:.0f})')

    print('\nRendering frames...')
    frames_to_render = list(range(len(mirrored)))
    rendered = []

    for fi in frames_to_render:
        nj = compute_new_joints(sp, ref, mirrored[fi], W, H)
        frame_img = sprite.copy()
        frame_img = draw_skeleton(frame_img, nj)
        # Add frame number
        draw = ImageDraw.Draw(frame_img)
        draw.text((10, 10), f'Frame {fi+1}/{len(mirrored)}', fill=(255,255,100,255))
        out_path = os.path.join(out_dir, f'frame_{fi:03d}.png')
        frame_img.save(out_path)
        rendered.append(frame_img)
        if fi % 5 == 0:
            print(f'  Frame {fi+1}/{len(mirrored)}')

    # Save contact sheet
    cols = 8
    rows = math.ceil(len(rendered) / cols)
    thumb_w, thumb_h = W // 4, H // 4
    sheet = Image.new('RGBA', (cols * thumb_w, rows * thumb_h), (20, 20, 30, 255))
    for i, fr in enumerate(rendered):
        thumb = fr.resize((thumb_w, thumb_h), Image.LANCZOS)
        cx, cy = (i % cols) * thumb_w, (i // cols) * thumb_h
        sheet.paste(thumb, (cx, cy))
    sheet.save('test_contact_sheet.png')
    print(f'\nSaved {len(rendered)} frames to {out_dir}/')
    print('Saved test_contact_sheet.png')

    # Print FK sanity check: verify joints move meaningfully across frames
    print('\nFK range check (joint travel across all frames):')
    all_nj = [compute_new_joints(sp, ref, mirrored[fi], W, H) for fi in range(len(mirrored))]
    for ji in [J['HEAD'], J['LW'], J['RW'], J['LA'], J['RA'], J['PELVIS']]:
        xs = [nj[ji]['x'] for nj in all_nj]
        ys = [nj[ji]['y'] for nj in all_nj]
        print(f'  {J_NAMES[ji]:12s}: x=[{min(xs):.0f},{max(xs):.0f}] y=[{min(ys):.0f},{max(ys):.0f}]')

if __name__ == '__main__':
    main()
