"""
Visual test: renders actual mesh-deformed sprite frames using FK + triangle texture mapping.
Saves test_frames/ PNGs and test_contact_sheet.png.
"""
import json, math, os
from PIL import Image, ImageDraw

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
DRAW_ORDER  = [6,7,8,9,1,0,4,5,2,3]
MIRROR_IDX  = [0,2,1,4,3,6,5,8,7,10,9,12,11,13,14]

MESH_SPACING = 18   # px between grid vertices
BLEND_RADIUS = 35   # px for joint blending
Y_SCALE      = 0.8  # hip bounce scale
CANVAS_PAD   = 200  # extra pixels around frame so arms don't clip


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
    return d

def seg_bone_xy(x, y, sp):
    """Zone-based bone assignment. sp = list of (px,py) sprite joint positions.
    No hard pelvis cutoff — arms extend below pelvis in rest pose and would
    otherwise be mis-assigned to leg bones, causing elbow separation."""
    neck_y   = sp[J['NECK']][1]
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

    # For ALL pixels below neck, compare distance to every body segment
    torso_left  = min(ls_x, lh_x)
    torso_right = max(rs_x, rh_x)
    in_torso = torso_left <= x <= torso_right
    # Don't bias toward torso near shoulder joints — those pixels belong to arm bones
    d_to_ls = math.hypot(x - sp[J['LS']][0], y - sp[J['LS']][1])
    d_to_rs = math.hypot(x - sp[J['RS']][0], y - sp[J['RS']][1])
    near_shoulder = min(d_to_ls, d_to_rs) < 55

    scores = [
        seg_dist(*sp[J['PELVIS']],*sp[J['NECK']]) * (0.6 if (in_torso and not near_shoulder) else 1.0),  # torso (1)
        seg_dist(*sp[J['LS']],    *sp[J['LE']]),     # upperArmL (2)
        seg_dist(*sp[J['LE']],    *sp[J['LW']]),     # forearmL  (3)
        seg_dist(*sp[J['RS']],    *sp[J['RE']]),     # upperArmR (4)
        seg_dist(*sp[J['RE']],    *sp[J['RW']]),     # forearmR  (5)
        seg_dist(*sp[J['LH']],    *sp[J['LK']]),     # thighL    (6)
        seg_dist(*sp[J['LK']],    *sp[J['LA']]),     # shinL     (7)
        seg_dist(*sp[J['RH']],    *sp[J['RK']]),     # thighR    (8)
        seg_dist(*sp[J['RK']],    *sp[J['RA']]),     # shinR     (9)
    ]
    return [1,2,3,4,5,6,7,8,9][scores.index(min(scores))]

def compute_vertex_blends(verts, vert_bone, sp, blend_radius):
    blends = []
    for i, v in enumerate(verts):
        bone = vert_bone[i]
        bd = BONE_DEFS[bone]

        # Special: torso vertices near shoulder joints blend with the arm bones
        # (shoulders are branch joints not on the torso bone path, so normal
        #  pivot/end blending doesn't reach them, causing seams when arms move)
        if bone == 1:  # torso
            d_ls = math.hypot(v[0]-sp[J['LS']][0], v[1]-sp[J['LS']][1])
            d_rs = math.hypot(v[0]-sp[J['RS']][0], v[1]-sp[J['RS']][1])
            if d_ls < blend_radius and d_ls <= d_rs:
                w = 0.5 + 0.5 * d_ls / blend_radius
                blends.append({'bone': bone, 'blend': 2, 'w': w}); continue  # → upperArmL
            if d_rs < blend_radius:
                w = 0.5 + 0.5 * d_rs / blend_radius
                blends.append({'bone': bone, 'blend': 4, 'w': w}); continue  # → upperArmR

        d_pivot = math.hypot(v[0]-sp[bd['pivot']][0], v[1]-sp[bd['pivot']][1])
        d_end   = math.hypot(v[0]-sp[bd['end']][0],   v[1]-sp[bd['end']][1])

        if d_end < blend_radius:
            child = next((b['id'] for b in BONE_DEFS if b['pivot']==bd['end'] and b['id']!=bone), -1)
            if child >= 0:
                w = 0.5 + 0.5 * d_end / blend_radius
                blends.append({'bone':bone,'blend':child,'w':w}); continue

        if d_pivot < blend_radius and BONE_PARENT[bone] >= 0:
            w = 0.5 + 0.5 * d_pivot / blend_radius
            blends.append({'bone':bone,'blend':BONE_PARENT[bone],'w':w}); continue

        blends.append({'bone':bone,'blend':-1,'w':1.0})
    return blends

def compute_fk(frame_motion, ref_motion, sp_px, H):
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
    fk(J['NECK'],  J['LS'],   angle_delta(ref_motion,frame_motion,J['NECK'],  J['LS']))
    fk(J['NECK'],  J['RS'],   angle_delta(ref_motion,frame_motion,J['NECK'],  J['RS']))
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

def rotate_around_bone(v, bone_id, nj, bone_deltas, sp_px):
    bd = BONE_DEFS[bone_id]
    ox = v[0] - sp_px[bd['pivot']][0]; oy = v[1] - sp_px[bd['pivot']][1]
    c, s = math.cos(bone_deltas[bone_id]), math.sin(bone_deltas[bone_id])
    return (nj[bd['pivot']]['x']+ox*c-oy*s, nj[bd['pivot']]['y']+ox*s+oy*c)

def transform_vertex(v, blend, nj, bone_deltas, sp_px):
    px = rotate_around_bone(v, blend['bone'], nj, bone_deltas, sp_px)
    if blend['blend'] < 0: return px
    sx = rotate_around_bone(v, blend['blend'], nj, bone_deltas, sp_px)
    w = blend['w']
    return (px[0]*w + sx[0]*(1-w), px[1]*w + sx[1]*(1-w))

def draw_tex_triangle(out_img, src_img, d0, d1, d2, s0, s1, s2):
    """Affine triangle texture mapping using PIL."""
    from PIL import Image as PILImage

    min_x = int(max(0, min(d0[0],d1[0],d2[0])))
    max_x = int(min(out_img.width,  max(d0[0],d1[0],d2[0]))) + 1
    min_y = int(max(0, min(d0[1],d1[1],d2[1])))
    max_y = int(min(out_img.height, max(d0[1],d1[1],d2[1]))) + 1

    if max_x <= min_x or max_y <= min_y: return

    # Affine: source from dest. Solve: M*dest = source
    dx1, dy1 = d1[0]-d0[0], d1[1]-d0[1]
    dx2, dy2 = d2[0]-d0[0], d2[1]-d0[1]
    sx1, sy1 = s1[0]-s0[0], s1[1]-s0[1]
    sx2, sy2 = s2[0]-s0[0], s2[1]-s0[1]
    det = dx1*dy2 - dx2*dy1
    if abs(det) < 0.01: return

    # M maps dest → source
    a = (sx1*dy2 - sx2*dy1) / det
    b = (sx2*dx1 - sx1*dx2) / det
    c = s0[0] - a*d0[0] - b*d0[1]
    dd = (sy1*dy2 - sy2*dy1) / det
    e = (sy2*dx1 - sy1*dx2) / det
    f = s0[1] - dd*d0[0] - e*d0[1]

    bw, bh = max_x - min_x, max_y - min_y
    # Adjust for bbox offset
    c2 = c + a*min_x + b*min_y
    f2 = f + dd*min_x + e*min_y

    try:
        region = src_img.transform((bw, bh), PILImage.AFFINE, (a, b, c2, dd, e, f2),
                                   resample=PILImage.BILINEAR)
    except Exception:
        return

    # Triangle mask within bbox
    mask = PILImage.new('L', (bw, bh), 0)
    md = ImageDraw.Draw(mask)
    md.polygon([(d0[0]-min_x, d0[1]-min_y),
                (d1[0]-min_x, d1[1]-min_y),
                (d2[0]-min_x, d2[1]-min_y)], fill=255)

    out_img.paste(region, (min_x, min_y), mask)

def generate_mesh(W, H, spacing, alpha_fn):
    cols = W // spacing + 2
    rows = H // spacing + 2
    vert_grid = {}
    verts = []

    for r in range(rows):
        for c in range(cols):
            x = min(c * spacing, W)
            y = min(r * spacing, H)
            has_alpha = any(alpha_fn(x+dx, y+dy) > 10
                           for dy in range(-spacing, spacing+1, spacing)
                           for dx in range(-spacing, spacing+1, spacing))
            if has_alpha:
                vert_grid[(r,c)] = len(verts)
                verts.append((float(x), float(y)))

    tris = []
    for r in range(rows-1):
        for c in range(cols-1):
            i00 = vert_grid.get((r,  c))
            i10 = vert_grid.get((r+1,c))
            i01 = vert_grid.get((r,  c+1))
            i11 = vert_grid.get((r+1,c+1))
            if i00 is not None and i10 is not None and i01 is not None:
                tris.append((i00,i10,i01))
            if i10 is not None and i11 is not None and i01 is not None:
                tris.append((i10,i11,i01))
    return verts, tris

def render_frame(sprite_rgba, verts, tris, vert_blends, tri_bone,
                 nj, bone_deltas, sp_px, W, H, pad=CANVAS_PAD):
    OW, OH = W + 2*pad, H + 2*pad
    out = Image.new('RGBA', (OW, OH), (0,0,0,0))
    # Offset transformed positions into the padded canvas so arms don't clip
    raw = [transform_vertex(v, vert_blends[i], nj, bone_deltas, sp_px)
           for i, v in enumerate(verts)]
    transformed = [(tx + pad, ty + pad) for tx, ty in raw]

    for bone_id in DRAW_ORDER:
        for t_idx, tri in enumerate(tris):
            if tri_bone[t_idx] != bone_id: continue
            s = [verts[tri[i]] for i in range(3)]        # source: sprite-space
            d = [transformed[tri[i]] for i in range(3)]  # dest:   padded canvas
            draw_tex_triangle(out, sprite_rgba, d[0],d[1],d[2], s[0],s[1],s[2])
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

    # Build alpha lookup
    alpha_data = list(sprite.getdata(3))
    def get_alpha(x, y):
        ix = max(0, min(W-1, int(round(x))))
        iy = max(0, min(H-1, int(round(y))))
        return alpha_data[iy*W+ix]

    with open(json_path) as f:
        data = json.load(f)

    fps = data['fps']
    mirrored = [mirror_frame(fr) for fr in data['frames']]
    ref = mirrored[0]
    n_frames = len(mirrored)
    print(f'Pose: {n_frames} frames @ {fps}fps')

    # Sprite joint positions in pixels
    sp_px = [(jx*W, jy*H) for jx, jy in DEFAULT_JOINTS]

    # Build mesh
    print('Building mesh...')
    verts, tris = generate_mesh(W, H, MESH_SPACING, get_alpha)
    print(f'  {len(verts)} verts, {len(tris)} tris')

    # Assign bones
    vert_bone = [seg_bone_xy(v[0], v[1], sp_px) for v in verts]
    vert_blends = compute_vertex_blends(verts, vert_bone, sp_px, BLEND_RADIUS)

    # Assign triangles by centroid
    tri_bone = []
    for tri in tris:
        cx = (verts[tri[0]][0]+verts[tri[1]][0]+verts[tri[2]][0]) / 3
        cy = (verts[tri[0]][1]+verts[tri[1]][1]+verts[tri[2]][1]) / 3
        tri_bone.append(seg_bone_xy(cx, cy, sp_px))

    print('Rendering frames...')
    rendered = []
    for fi in range(n_frames):
        nj, bone_deltas = compute_fk(mirrored[fi], ref, sp_px, H)
        frame_img = render_frame(sprite, verts, tris, vert_blends, tri_bone,
                                 nj, bone_deltas, sp_px, W, H)
        # Optional: overlay skeleton (pad already applied in draw_skeleton_overlay)
        draw_skeleton_overlay(frame_img, nj)
        # Add white background for visibility (frame is padded size)
        FW, FH = W + 2*CANVAS_PAD, H + 2*CANVAS_PAD
        bg = Image.new('RGBA', (FW, FH), (240, 240, 240, 255))
        bg.paste(frame_img, (0, 0), frame_img)
        bg.save(os.path.join(out_dir, f'frame_{fi:03d}.png'))
        rendered.append(bg)
        if fi % 5 == 0: print(f'  Frame {fi+1}/{n_frames}')

    # Contact sheet (thumbnails)
    cols = 7
    rows = math.ceil(n_frames / cols)
    FW, FH = W + 2*CANVAS_PAD, H + 2*CANVAS_PAD
    tw, th = FW//3, FH//3
    sheet = Image.new('RGB', (cols*tw, rows*th), (20,20,30))
    for i, fr in enumerate(rendered):
        thumb = fr.resize((tw, th), Image.LANCZOS)
        sheet.paste(thumb, ((i%cols)*tw, (i//cols)*th))
    sheet_path = os.path.join(base, 'test_contact_sheet.png')
    sheet.save(sheet_path)
    print(f'\nSaved {n_frames} frames → {out_dir}/')
    print(f'Contact sheet → {sheet_path}')

if __name__ == '__main__':
    main()
