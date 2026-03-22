#!/usr/bin/env python3.12
"""
sprite_animator_gui.py  –  Interactive sprite animator editor.
Built incrementally; run at any step to verify progress.
"""

from __future__ import annotations

import json
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageTk

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = '/home/user/sprite-retarget/sprite-retarget'
DEFAULT_SPRITE = os.path.join(BASE, 'bodywhole1.png')
DEFAULT_JSON   = os.path.join(BASE, 'motion_data.json')

# ── Skeleton constants (mirrored from test_visual.py) ─────────────────────────
J = dict(HEAD=0,LS=1,RS=2,LE=3,RE=4,LW=5,RW=6,
         LH=7,RH=8,LK=9,RK=10,LA=11,RA=12,NECK=13,PELVIS=14)

DEFAULT_JOINTS = [
    (0.50,0.08),(0.22,0.33),(0.77,0.33),
    (0.15,0.49),(0.84,0.49),(0.11,0.63),
    (0.87,0.63),(0.36,0.49),(0.60,0.49),
    (0.33,0.78),(0.64,0.78),(0.31,0.92),
    (0.67,0.94),(0.50,0.28),(0.50,0.49),
]

BONE_DEFS = [
    dict(id=0, name='head',      pivot=J['NECK'],   end=J['HEAD']),
    dict(id=1, name='torso',     pivot=J['PELVIS'], end=J['NECK']),
    dict(id=2, name='upperArmL', pivot=J['LS'],     end=J['LE']),
    dict(id=3, name='forearmL',  pivot=J['LE'],     end=J['LW']),
    dict(id=4, name='upperArmR', pivot=J['RS'],     end=J['RE']),
    dict(id=5, name='forearmR',  pivot=J['RE'],     end=J['RW']),
    dict(id=6, name='thighL',    pivot=J['LH'],     end=J['LK']),
    dict(id=7, name='shinL',     pivot=J['LK'],     end=J['LA']),
    dict(id=8, name='thighR',    pivot=J['RH'],     end=J['RK']),
    dict(id=9, name='shinR',     pivot=J['RK'],     end=J['RA']),
]

DRAW_ORDER  = [6, 7, 8, 9, 1, 0, 4, 5, 2, 3]
MIRROR_IDX  = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 13, 14]
CANVAS_PAD  = 200

BONE_COLORS = [          # (R, G, B) per bone id 0-9
    (255, 200,  50),     # 0  head       gold
    (100, 200, 255),     # 1  torso      sky-blue
    ( 50, 210, 110),     # 2  upperArmL  green
    ( 30, 150,  60),     # 3  forearmL   dark-green
    (255, 130,  50),     # 4  upperArmR  orange
    (200,  80,  20),     # 5  forearmR   burnt-orange
    (160, 110, 255),     # 6  thighL     purple
    (100,  60, 200),     # 7  shinL      dark-purple
    (255,  80, 140),     # 8  thighR     pink
    (190,  30,  80),     # 9  shinR      crimson
]
OVERLAY_ALPHA = 160      # 0-255


# ── Main application class ────────────────────────────────────────────────────
class AnimatorApp(tk.Tk):

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title('Sprite Animator')
        self.geometry('1280x860')
        self.configure(bg='#1a1a2e')

        # ── loaded data ───────────────────────────────────────────────────────
        self.sprite: Image.Image | None = None   # RGBA, native size
        self.frames: list | None        = None   # list of 15-joint frames
        self.fps: int                   = 12
        self.sp_px: list | None         = None   # [(x,y)…] joint pixels on sprite

        # bone data (populated in steps 5-7)
        self.bone_assign: np.ndarray | None = None   # H×W uint8, 255=empty
        self.bone_images: dict              = {}     # {0-9: RGBA PIL}
        self.alpha_pil:  Image.Image | None = None   # L-mode alpha

        self._build_ui()
        self._load_defaults()

    # ── UI scaffold ───────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menu()

        # Horizontal paned window: editor | controls+preview
        self.paned = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                                    bg='#1a1a2e', sashwidth=5,
                                    sashrelief=tk.RIDGE)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left pane – bone editor (placeholder label for now)
        self.left_frame = ttk.Frame(self.paned)
        self._build_editor_placeholder()
        self.paned.add(self.left_frame, minsize=380)

        # Right pane – parameters + preview
        self.right_frame = ttk.Frame(self.paned)
        self._build_right_placeholder()
        self.paned.add(self.right_frame, minsize=380)

        # Status bar
        self.status_var = tk.StringVar(value='No data loaded')
        tk.Label(self, textvariable=self.status_var, anchor='w',
                 bg='#0d0d1a', fg='#8080a0', font=('mono', 9)
                 ).pack(fill=tk.X, side=tk.BOTTOM, padx=4)

    def _build_menu(self):
        mb = tk.Menu(self, bg='#1a1a2e', fg='#c0c0e0',
                     activebackground='#2a2a4e', tearoff=False)
        self.configure(menu=mb)

        fm = tk.Menu(mb, tearoff=False, bg='#1a1a2e', fg='#c0c0e0')
        mb.add_cascade(label='File', menu=fm)
        fm.add_command(label='Load Sprite…',     command=self._open_sprite,   accelerator='Ctrl+O')
        fm.add_command(label='Load JSON…',       command=self._open_json,     accelerator='Ctrl+J')
        fm.add_separator()
        fm.add_command(label='Export GIF…',      command=self._export_gif,    accelerator='Ctrl+G')
        fm.add_command(label='Export Frames…',   command=self._export_frames, accelerator='Ctrl+E')
        fm.add_separator()
        fm.add_command(label='Quit',             command=self.destroy,         accelerator='Ctrl+Q')

        # Keyboard shortcuts
        self.bind('<Control-o>', lambda _: self._open_sprite())
        self.bind('<Control-j>', lambda _: self._open_json())
        self.bind('<Control-g>', lambda _: self._export_gif())
        self.bind('<Control-e>', lambda _: self._export_frames())
        self.bind('<Control-q>', lambda _: self.destroy())
        self.bind('<space>',     lambda _: self._toggle_play())
        self.bind('<Left>',      lambda _: self._step_back())
        self.bind('<Right>',     lambda _: self._step_fwd())
        self.bind('<Home>',      lambda _: self._go_first())
        self.bind('<End>',       lambda _: self._go_last())

    # ── Bone editor panel (Steps 6-8) ─────────────────────────────────────────
    def _build_editor_placeholder(self):
        self._build_bone_selector(self.left_frame)
        self._build_editor_canvas(self.left_frame)

    def _build_right_placeholder(self):
        self._build_param_panel(self.right_frame)
        self._build_preview_panel(self.right_frame)

    # ── Preview panel (Step 4) ─────────────────────────────────────────────────
    def _build_preview_panel(self, parent):
        self.v_frame = tk.IntVar(value=0)
        self._playing   = False
        self._play_job  = None
        self._render_cache = {}
        self._preview_photo = None   # keep PhotoImage reference

        # Preview canvas
        pf = ttk.LabelFrame(parent, text='Preview')
        pf.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.preview_canvas = tk.Canvas(pf, bg='#0d0d1a', width=360, height=460)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Frame scrubber
        sf = ttk.Frame(parent)
        sf.pack(fill=tk.X, padx=6, pady=(0, 2))
        self.scrubber = ttk.Scale(sf, from_=0, to=1, variable=self.v_frame,
                                  orient=tk.HORIZONTAL,
                                  command=self._on_scrub)
        self.scrubber.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 6))
        self.frame_lbl = ttk.Label(sf, text='0 / 0', width=8)
        self.frame_lbl.pack(side=tk.RIGHT)

        # Playback buttons
        bf = ttk.Frame(parent)
        bf.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(bf, text='|◀', width=3, command=self._go_first).pack(side=tk.LEFT)
        ttk.Button(bf, text='◀',  width=3, command=self._step_back).pack(side=tk.LEFT)
        self.play_btn = ttk.Button(bf, text='▶ Play', width=8,
                                   command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='▶',  width=3, command=self._step_fwd).pack(side=tk.LEFT)
        ttk.Button(bf, text='▶|', width=3, command=self._go_last).pack(side=tk.LEFT)

    # ── Playback logic ────────────────────────────────────────────────────────
    def _on_scrub(self, _=None):
        self._update_preview()

    def _go_first(self):  self.v_frame.set(0);                       self._update_preview()
    def _go_last(self):   self.v_frame.set(self._n_frames() - 1);    self._update_preview()
    def _step_back(self): self.v_frame.set(max(0, self.v_frame.get() - 1)); self._update_preview()
    def _step_fwd(self):  self.v_frame.set(min(self._n_frames()-1, self.v_frame.get()+1)); self._update_preview()

    def _n_frames(self) -> int:
        return len(self.frames) if self.frames else 1

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            if self._play_job:
                self.after_cancel(self._play_job)
                self._play_job = None
            self.play_btn.configure(text='▶ Play')
        else:
            self._playing = True
            self.play_btn.configure(text='⏸ Pause')
            self._advance_frame()

    def _advance_frame(self):
        if not self._playing:
            return
        fi = (self.v_frame.get() + 1) % self._n_frames()
        self.v_frame.set(fi)
        self._update_preview()
        interval = max(16, 1000 // max(1, self.v_fps.get()))
        self._play_job = self.after(interval, self._advance_frame)

    def _update_preview(self):
        if not hasattr(self, 'preview_canvas'):
            return
        fi = self.v_frame.get()
        n  = self._n_frames()

        # Update label
        if hasattr(self, 'frame_lbl'):
            self.frame_lbl.configure(text=f'{fi} / {n-1}')
        if hasattr(self, 'scrubber'):
            self.scrubber.configure(to=max(1, n - 1))

        img = self._render_frame_pil(fi)
        if img is None:
            self.preview_canvas.delete('all')
            self.preview_canvas.create_text(
                180, 230, text='Load sprite + JSON\nto preview',
                fill='#4060a0', font=('mono', 12), justify=tk.CENTER)
            return

        # Scale to fit canvas
        cw = self.preview_canvas.winfo_width()  or 360
        ch = self.preview_canvas.winfo_height() or 460
        iw, ih = img.size
        scale = min(cw / iw, ch / ih, 1.0)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        disp = img.resize((dw, dh), Image.LANCZOS).convert('RGBA')

        # Composite on dark background
        bg = Image.new('RGBA', (cw, ch), (20, 20, 32, 255))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        bg.paste(disp, (ox, oy), disp)

        photo = ImageTk.PhotoImage(bg)
        self._preview_photo = photo          # prevent GC
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(0, 0, anchor='nw', image=photo)

    def _schedule_preview_update(self):
        if hasattr(self, '_preview_job'):
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(120, self._update_preview)

    def _schedule_bone_image_rebuild(self):
        if hasattr(self, '_bone_rebuild_job'):
            self.after_cancel(self._bone_rebuild_job)
        self._bone_rebuild_job = self.after(300, self._rebuild_bone_images)

    # ── Numpy bone assignment (Step 5) ────────────────────────────────────────
    def _auto_assign(self):
        """Vectorised distance-to-segment bone assignment (replaces slow Python loop)."""
        if self.sprite is None:
            return
        W, H = self.sprite.size
        arr  = np.array(self.sprite)
        alpha = arr[:, :, 3]

        ys, xs = np.where(alpha > 10)
        if not len(xs):
            self.bone_assign = np.full((H, W), 255, np.uint8)
            return

        xf = xs.astype(np.float32)
        yf = ys.astype(np.float32)
        sp = self.sp_px

        def seg_dist(pi, ei):
            ax, ay = sp[pi]
            bx, by = sp[ei]
            dx, dy = bx - ax, by - ay
            l2 = max(dx*dx + dy*dy, 1e-8)
            t  = np.clip(((xf-ax)*dx + (yf-ay)*dy) / l2, 0.0, 1.0)
            return np.hypot(xf - (ax + t*dx), yf - (ay + t*dy))

        # Distances for non-head bones (map to bone IDs 1-9)
        bone_segs = [
            (J['PELVIS'], J['NECK']),  # → bone 1  torso
            (J['LS'],     J['LE']),    # → bone 2  upperArmL
            (J['LE'],     J['LW']),    # → bone 3  forearmL
            (J['RS'],     J['RE']),    # → bone 4  upperArmR
            (J['RE'],     J['RW']),    # → bone 5  forearmR
            (J['LH'],     J['LK']),    # → bone 6  thighL
            (J['LK'],     J['LA']),    # → bone 7  shinL
            (J['RH'],     J['RK']),    # → bone 8  thighR
            (J['RK'],     J['RA']),    # → bone 9  shinR
        ]
        dists = np.stack([seg_dist(pi, ei) for pi, ei in bone_segs])  # (9, N)

        # Torso-zone bias (prefers torso over arm/shoulder confusion)
        ls_x = sp[J['LS']][0]; rs_x = sp[J['RS']][0]
        lh_x = sp[J['LH']][0]; rh_x = sp[J['RH']][0]
        in_torso = (xf >= min(ls_x, lh_x)) & (xf <= max(rs_x, rh_x))
        d_ls = np.hypot(xf - sp[J['LS']][0], yf - sp[J['LS']][1])
        d_rs = np.hypot(xf - sp[J['RS']][0], yf - sp[J['RS']][1])
        near_shoulder = np.minimum(d_ls, d_rs) < 55
        dists[0] *= np.where(in_torso & ~near_shoulder, 0.6, 1.0)

        best = np.argmin(dists, axis=0).astype(np.uint8) + 1   # 1-9
        # Head: any opaque pixel above the neck line → bone 0
        neck_y = sp[J['NECK']][1]
        best[yf <= neck_y] = 0

        result = np.full((H, W), 255, np.uint8)
        result[ys, xs] = best
        self.bone_assign = result

        self._rebuild_bone_images()
        self._refresh_editor()
        self.status_var.set('Auto-assign complete')

    def _rebuild_bone_images(self):
        """Build per-bone RGBA cutout images from bone_assign + current dilation."""
        if self.sprite is None or self.bone_assign is None:
            return
        dil     = self.v_dil.get()
        alpha_p = self.alpha_pil

        for bi in range(10):
            mask_arr  = ((self.bone_assign == bi).astype(np.uint8) * 255)
            mask_pil  = Image.fromarray(mask_arr, 'L')
            dilated   = mask_pil
            for _ in range(dil):
                dilated = dilated.filter(ImageFilter.MaxFilter(3))
            final_mask = ImageChops.darker(dilated, alpha_p)
            bone_img   = Image.new('RGBA', self.sprite.size, (0, 0, 0, 0))
            bone_img.paste(self.sprite, mask=final_mask)
            self.bone_images[bi] = bone_img

        self._invalidate_render_cache()
        self._update_preview()

    def _on_sprite_loaded(self):
        self._auto_assign()
        self._update_preview()

    def _on_json_loaded(self):
        n = len(self.frames)
        self.v_frame.set(0)
        if hasattr(self, 'scrubber'):
            self.scrubber.configure(to=max(1, n - 1))
        self._update_preview()

    # ── Bone selector (Step 8) ────────────────────────────────────────────────
    def _build_bone_selector(self, parent):
        self.v_active_bone = tk.IntVar(value=1)
        self.v_brush_r     = tk.IntVar(value=15)
        self.v_show_overlay = tk.BooleanVar(value=True)

        top = ttk.LabelFrame(parent, text='Bone Painter')
        top.pack(fill=tk.X, padx=6, pady=4)

        NAMES = ['head','torso','uArmL','fArmL','uArmR','fArmR',
                 'thighL','shinL','thighR','shinR']
        for bi, name in enumerate(NAMES):
            r, g, b = BONE_COLORS[bi]
            hex_col = f'#{r:02x}{g:02x}{b:02x}'
            # Perceived luminance → choose black or white text
            lum = 0.299*r + 0.587*g + 0.114*b
            fg  = 'black' if lum > 140 else 'white'
            btn = tk.Radiobutton(top, text=name, variable=self.v_active_bone,
                                 value=bi, bg=hex_col, activebackground=hex_col,
                                 selectcolor=hex_col, fg=fg, indicatoron=False,
                                 font=('mono', 8), relief='raised', width=8)
            btn.grid(row=bi // 2, column=bi % 2, padx=3, pady=2, sticky='ew')

        brush_row = ttk.Frame(top)
        brush_row.grid(row=5, column=0, columnspan=2, sticky='ew', padx=4, pady=2)
        ttk.Label(brush_row, text='Brush px:').pack(side=tk.LEFT)
        ttk.Scale(brush_row, from_=2, to=80, variable=self.v_brush_r,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=4)
        self._brush_lbl = ttk.Label(brush_row, text='15', width=3)
        self._brush_lbl.pack(side=tk.LEFT)
        self.v_brush_r.trace_add('write',
            lambda *_: self._brush_lbl.configure(text=str(self.v_brush_r.get())))

        ctrl_row = ttk.Frame(top)
        ctrl_row.grid(row=6, column=0, columnspan=2, sticky='ew', padx=4, pady=4)
        ttk.Checkbutton(ctrl_row, text='Show overlay',
                        variable=self.v_show_overlay,
                        command=self._refresh_editor).pack(side=tk.LEFT)
        ttk.Button(ctrl_row, text='Auto-assign',
                   command=self._auto_assign).pack(side=tk.RIGHT, padx=4)

    # ── Editor canvas (Step 6) ────────────────────────────────────────────────
    def _build_editor_canvas(self, parent):
        self._editor_photo  = None
        self._editor_scale  = 1.0
        self._painting      = False

        cf = ttk.Frame(parent)
        cf.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.editor_canvas = tk.Canvas(cf, bg='#0d0d1a', cursor='crosshair')
        vsb = ttk.Scrollbar(cf, orient=tk.VERTICAL,   command=self.editor_canvas.yview)
        hsb = ttk.Scrollbar(cf, orient=tk.HORIZONTAL, command=self.editor_canvas.xview)
        self.editor_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        hsb.grid(row=1, column=0, sticky='ew')
        vsb.grid(row=0, column=1, sticky='ns')
        self.editor_canvas.grid(row=0, column=0, sticky='nsew')
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)

        # Paint bindings
        self.editor_canvas.bind('<ButtonPress-1>',   self._on_paint_press)
        self.editor_canvas.bind('<B1-Motion>',        self._on_paint_drag)
        self.editor_canvas.bind('<ButtonRelease-1>',  self._on_paint_release)
        # Zoom via scroll wheel
        self.editor_canvas.bind('<MouseWheel>',       self._on_editor_scroll)
        self.editor_canvas.bind('<Button-4>',  lambda e: self._zoom_editor(1.15))
        self.editor_canvas.bind('<Button-5>',  lambda e: self._zoom_editor(1/1.15))

    # ── Editor overlay rendering (Step 6) ─────────────────────────────────────
    def _refresh_editor(self):
        if self.sprite is None or not hasattr(self, 'editor_canvas'):
            return

        W, H = self.sprite.size
        base = self.sprite.convert('RGB')

        if self.v_show_overlay.get() and self.bone_assign is not None:
            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            for bi, (r, g, b) in enumerate(BONE_COLORS):
                mask = self.bone_assign == bi
                overlay[mask] = (r, g, b, OVERLAY_ALPHA)
            ov_pil  = Image.fromarray(overlay, 'RGBA')
            base_a  = self.sprite.copy()
            base_a.paste(ov_pil, mask=ov_pil)
            base    = base_a.convert('RGB')

        # Draw joint markers
        draw = ImageDraw.Draw(base)
        for i, (jx, jy) in enumerate(self.sp_px):
            r = 5
            draw.ellipse([jx-r, jy-r, jx+r, jy+r],
                         fill='yellow', outline='black', width=1)

        # Scale to fit canvas width
        cw = self.editor_canvas.winfo_width() or 440
        scale = min(cw / W, 1.0)
        scale = max(scale, self._editor_scale)   # respect zoom
        self._editor_scale = scale
        dw, dh = max(1, int(W * scale)), max(1, int(H * scale))
        disp = base.resize((dw, dh), Image.NEAREST if scale > 1 else Image.LANCZOS)

        photo = ImageTk.PhotoImage(disp)
        self._editor_photo = photo
        self.editor_canvas.delete('all')
        self.editor_canvas.create_image(0, 0, anchor='nw', image=photo)
        self.editor_canvas.configure(scrollregion=(0, 0, dw, dh))

    def _on_sprite_loaded(self):
        self._editor_scale = 1.0
        self._auto_assign()
        self._update_preview()

    # ── Bone painter (Step 7) ─────────────────────────────────────────────────
    def _canvas_to_sprite(self, cx, cy):
        """Map editor canvas coords → sprite pixel coords."""
        sx = int(self.editor_canvas.canvasx(cx) / self._editor_scale)
        sy = int(self.editor_canvas.canvasy(cy) / self._editor_scale)
        return sx, sy

    def _paint_at(self, cx, cy):
        if self.bone_assign is None:
            return
        sx, sy = self._canvas_to_sprite(cx, cy)
        H, W   = self.bone_assign.shape
        r      = max(1, int(self.v_brush_r.get() * self._editor_scale))
        bone   = self.v_active_bone.get()
        # Circular brush via numpy
        y0, y1 = max(0, sy-r), min(H, sy+r+1)
        x0, x1 = max(0, sx-r), min(W, sx+r+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy - sy)**2 + (xx - sx)**2 <= r*r
        # Only paint opaque pixels
        alpha_crop = np.array(self.alpha_pil)[y0:y1, x0:x1]
        self.bone_assign[y0:y1, x0:x1][circle & (alpha_crop > 10)] = bone
        self._refresh_editor()

    def _on_paint_press(self, event):
        self._painting = True
        self._paint_at(event.x, event.y)

    def _on_paint_drag(self, event):
        if self._painting:
            self._paint_at(event.x, event.y)

    def _on_paint_release(self, event):
        self._painting = False
        # Rebuild only the affected bone image(s) — fast enough at 15px brush
        self._rebuild_bone_images()

    def _on_editor_scroll(self, event):
        factor = 1.15 if event.delta > 0 else 1/1.15
        self._zoom_editor(factor)

    def _zoom_editor(self, factor):
        self._editor_scale = max(0.2, min(4.0, self._editor_scale * factor))
        self._refresh_editor()

    # ── Export (Step 9) ───────────────────────────────────────────────────────
    def _export_gif(self):
        if self.frames is None or not self.bone_images:
            messagebox.showwarning('Not ready', 'Load sprite and JSON first.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.gif',
            filetypes=[('GIF', '*.gif'), ('All', '*.*')],
            initialfile='export.gif')
        if not path:
            return
        self.status_var.set('Rendering…')
        self.update()
        gif_frames = []
        n = len(self.frames)
        for fi in range(n):
            img = self._render_frame_pil(fi)
            if img:
                # Resize to manageable GIF size
                scale = min(400 / img.width, 1.0)
                dw, dh = max(1, int(img.width*scale)), max(1, int(img.height*scale))
                gif_frames.append(img.resize((dw, dh), Image.LANCZOS).convert('RGBA'))
            if fi % 10 == 0:
                self.status_var.set(f'Rendering frame {fi}/{n}…')
                self.update()
        if not gif_frames:
            return
        dur = max(16, 1000 // max(1, self.v_fps.get()))
        pal_frames = [f.convert('P', palette=Image.ADAPTIVE, colors=128)
                      for f in gif_frames]
        pal_frames[0].save(path, save_all=True, append_images=pal_frames[1:],
                           loop=0, duration=dur)
        self.status_var.set(f'GIF saved → {path}')

    def _export_frames(self):
        if self.frames is None or not self.bone_images:
            messagebox.showwarning('Not ready', 'Load sprite and JSON first.')
            return
        folder = filedialog.askdirectory(title='Export frames to folder')
        if not folder:
            return
        n = len(self.frames)
        for fi in range(n):
            img = self._render_frame_pil(fi)
            if img:
                bg = Image.new('RGBA', img.size, (240, 240, 240, 255))
                bg.paste(img, mask=img)
                bg.convert('RGB').save(os.path.join(folder, f'frame_{fi:03d}.png'))
            if fi % 10 == 0:
                self.status_var.set(f'Exporting frame {fi}/{n}…')
                self.update()
        self.status_var.set(f'Exported {n} frames → {folder}')

    # ── Parameter panel (Step 2) ───────────────────────────────────────────────
    def _build_param_panel(self, parent):
        # Tkinter variables for all tunable parameters
        self.v_damp    = tk.DoubleVar(value=0.65)
        self.v_maxang  = tk.DoubleVar(value=1.20)
        self.v_yscale  = tk.DoubleVar(value=0.40)
        self.v_dil     = tk.IntVar(value=8)
        self.v_mirror  = tk.BooleanVar(value=False)
        self.v_fps     = tk.IntVar(value=12)

        outer = ttk.LabelFrame(parent, text='Parameters')
        outer.pack(fill=tk.X, padx=6, pady=6)

        # Continuous sliders
        sliders = [
            ('Dampening',  self.v_damp,   0.00, 1.50, self._on_anim_param),
            ('Max Angle',  self.v_maxang, 0.30, 3.00, self._on_anim_param),
            ('Y-Scale',    self.v_yscale, 0.00, 1.00, self._on_anim_param),
            ('Dil Radius', self.v_dil,    0,    20,   self._on_dil_param),
        ]
        self._val_labels = {}
        for row, (name, var, lo, hi, cb) in enumerate(sliders):
            ttk.Label(outer, text=name, width=11, anchor='w').grid(
                row=row, column=0, padx=6, sticky='w')
            sl = ttk.Scale(outer, from_=lo, to=hi, variable=var,
                           orient=tk.HORIZONTAL, length=180,
                           command=lambda v, c=cb: c())
            sl.grid(row=row, column=1, padx=4, pady=3)
            lbl = ttk.Label(outer, width=5, text=self._fmt(var))
            lbl.grid(row=row, column=2, padx=4)
            self._val_labels[name] = (lbl, var)
            var.trace_add('write', lambda *_, n=name: self._update_val_label(n))

        # Mirror checkbox
        ttk.Checkbutton(outer, text='Mirror L/R',
                        variable=self.v_mirror,
                        command=self._on_mirror_change
                        ).grid(row=len(sliders), column=0, columnspan=3,
                               pady=6, sticky='w', padx=6)

        # FPS spinner (used in Step 4 playback)
        fps_row = ttk.Frame(outer)
        fps_row.grid(row=len(sliders)+1, column=0, columnspan=3,
                     sticky='w', padx=6, pady=(0,6))
        ttk.Label(fps_row, text='FPS:').pack(side=tk.LEFT)
        ttk.Spinbox(fps_row, from_=1, to=60, textvariable=self.v_fps,
                    width=4).pack(side=tk.LEFT, padx=4)

    @staticmethod
    def _fmt(var):
        v = var.get()
        return f'{v:.2f}' if isinstance(v, float) else str(v)

    def _update_val_label(self, name):
        lbl, var = self._val_labels[name]
        lbl.configure(text=self._fmt(var))

    # Parameter-change callbacks (rendering hooks wired in Steps 3-4)
    def _on_anim_param(self):
        """Called when dampening / max-angle / y-scale changes."""
        self._invalidate_render_cache()
        self._schedule_preview_update()

    def _on_dil_param(self):
        """Dilation change requires rebuilding bone images."""
        self._invalidate_render_cache()
        self._schedule_bone_image_rebuild()

    def _on_mirror_change(self):
        self._invalidate_render_cache()
        self._schedule_preview_update()

    # Stubs – overridden / filled in later steps
    def _invalidate_render_cache(self):     pass
    def _schedule_preview_update(self):     pass
    def _schedule_bone_image_rebuild(self): pass

    # ── FK + render pipeline (Step 3) ─────────────────────────────────────────
    def _get_frame(self, fi: int) -> list:
        """Return the joint list for frame fi, with mirror applied if needed."""
        raw = self.frames[fi]
        if self.v_mirror.get():
            return [{'x': 1 - raw[MIRROR_IDX[i]]['x'],
                     'y':     raw[MIRROR_IDX[i]]['y'],
                     'v':     raw[MIRROR_IDX[i]]['v']}
                    for i in range(15)]
        return raw

    def _compute_fk(self, frame_joints: list, ref_joints: list):
        """
        Forward kinematics using current slider values.
        Returns (new_joints list, bone_deltas list).
        """
        damp    = self.v_damp.get()
        maxang  = self.v_maxang.get()
        yscale  = self.v_yscale.get()
        sp_px   = self.sp_px
        W, H    = self.sprite.size

        def bone_angle(joints, p, e):
            return math.atan2(joints[e]['y'] - joints[p]['y'],
                              joints[e]['x'] - joints[p]['x'])

        def angle_delta(ref, frm, p, e):
            d = bone_angle(frm, p, e) - bone_angle(ref, p, e)
            while d >  math.pi: d -= 2 * math.pi
            while d < -math.pi: d += 2 * math.pi
            d *= damp
            return max(-maxang, min(maxang, d))

        yD = (frame_joints[J['PELVIS']]['y'] - ref_joints[J['PELVIS']]['y']) * H * yscale
        nj = [{'x': p[0], 'y': p[1] + yD} for p in sp_px]

        def fk(par, child, delta):
            ox = sp_px[child][0] - sp_px[par][0]
            oy = sp_px[child][1] - sp_px[par][1]
            c, s = math.cos(delta), math.sin(delta)
            nj[child] = {'x': nj[par]['x'] + ox*c - oy*s,
                         'y': nj[par]['y'] + ox*s + oy*c}

        fk(J['PELVIS'], J['NECK'], angle_delta(ref_joints, frame_joints, J['PELVIS'], J['NECK']))
        fk(J['PELVIS'], J['LH'],   angle_delta(ref_joints, frame_joints, J['PELVIS'], J['LH']))
        fk(J['PELVIS'], J['RH'],   angle_delta(ref_joints, frame_joints, J['PELVIS'], J['RH']))
        fk(J['NECK'],   J['HEAD'], angle_delta(ref_joints, frame_joints, J['NECK'],   J['HEAD']))
        # Shoulders pinned to neck (avoids 2-D projection instability)
        for jid in (J['LS'], J['RS']):
            nj[jid] = {'x': nj[J['NECK']]['x'] + sp_px[jid][0] - sp_px[J['NECK']][0],
                       'y': nj[J['NECK']]['y'] + sp_px[jid][1] - sp_px[J['NECK']][1]}
        fk(J['LS'], J['LE'], angle_delta(ref_joints, frame_joints, J['LS'], J['LE']))
        fk(J['RS'], J['RE'], angle_delta(ref_joints, frame_joints, J['RS'], J['RE']))
        fk(J['LE'], J['LW'], angle_delta(ref_joints, frame_joints, J['LE'], J['LW']))
        fk(J['RE'], J['RW'], angle_delta(ref_joints, frame_joints, J['RE'], J['RW']))
        fk(J['LH'], J['LK'], angle_delta(ref_joints, frame_joints, J['LH'], J['LK']))
        fk(J['RH'], J['RK'], angle_delta(ref_joints, frame_joints, J['RH'], J['RK']))
        fk(J['LK'], J['LA'], angle_delta(ref_joints, frame_joints, J['LK'], J['LA']))
        fk(J['RK'], J['RA'], angle_delta(ref_joints, frame_joints, J['RK'], J['RA']))

        bone_deltas = [angle_delta(ref_joints, frame_joints, b['pivot'], b['end'])
                       for b in BONE_DEFS]
        return nj, bone_deltas

    def _render_frame_pil(self, fi: int) -> Image.Image | None:
        """Render frame fi → RGBA PIL Image, or None if data not ready."""
        if self.sprite is None or self.frames is None or not self.bone_images:
            return None

        cache_key = (fi,
                     self.v_mirror.get(),
                     round(self.v_damp.get(),   3),
                     round(self.v_maxang.get(),  3),
                     round(self.v_yscale.get(),  3))
        if hasattr(self, '_render_cache') and cache_key in self._render_cache:
            return self._render_cache[cache_key]

        ref_joints   = self._get_frame(0)
        frame_joints = self._get_frame(fi)
        nj, bone_deltas = self._compute_fk(frame_joints, ref_joints)

        W, H  = self.sprite.size
        pad   = CANVAS_PAD
        OW, OH = W + 2*pad, H + 2*pad
        out   = Image.new('RGBA', (OW, OH), (0, 0, 0, 0))

        for bone_id in DRAW_ORDER:
            bd      = BONE_DEFS[bone_id]
            angle   = bone_deltas[bone_id]
            px, py  = self.sp_px[bd['pivot']]
            npx     = nj[bd['pivot']]['x']
            npy     = nj[bd['pivot']]['y']
            ca, sa  = math.cos(angle), math.sin(angle)
            # Inverse affine: output → sprite coords
            coefs = (ca, sa, px - ca*(npx+pad) - sa*(npy+pad),
                     -sa, ca, py + sa*(npx+pad) - ca*(npy+pad))
            rotated = self.bone_images[bone_id].transform(
                (OW, OH), Image.AFFINE, coefs, resample=Image.BICUBIC)
            out = Image.alpha_composite(out, rotated)

        # Trim to content bounding box
        bbox = out.getbbox()
        if bbox:
            out = out.crop(bbox)

        if not hasattr(self, '_render_cache'):
            self._render_cache = {}
        self._render_cache[cache_key] = out
        return out

    def _invalidate_render_cache(self):
        self._render_cache = {}

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_defaults(self):
        if os.path.exists(DEFAULT_SPRITE):
            self._load_sprite(DEFAULT_SPRITE)
        if os.path.exists(DEFAULT_JSON):
            self._load_json(DEFAULT_JSON)

    def _open_sprite(self):
        path = filedialog.askopenfilename(
            title='Open sprite',
            filetypes=[('PNG', '*.png'), ('All', '*.*')])
        if path:
            self._load_sprite(path)

    def _open_json(self):
        path = filedialog.askopenfilename(
            title='Open motion JSON',
            filetypes=[('JSON', '*.json'), ('All', '*.*')])
        if path:
            self._load_json(path)

    def _load_sprite(self, path: str):
        try:
            img = Image.open(path).convert('RGBA')
        except Exception as e:
            messagebox.showerror('Load error', str(e))
            return
        self.sprite    = img
        self.alpha_pil = img.split()[3]        # L-mode alpha channel
        W, H = img.size

        # Joint positions in sprite pixels
        self.sp_px = [(jx * W, jy * H) for jx, jy in DEFAULT_JOINTS]

        self.bone_assign = None   # reset; will be rebuilt in step 5
        self.bone_images = {}
        self.status_var.set(f'Sprite loaded: {os.path.basename(path)}  {W}×{H}')
        self._on_sprite_loaded()

    def _load_json(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
            self.frames = data['frames']
            self.fps    = int(data.get('fps', 12))
        except Exception as e:
            messagebox.showerror('Load error', str(e))
            return
        n = len(self.frames)
        self.status_var.set(
            f'JSON loaded: {os.path.basename(path)}  {n} frames @ {self.fps} fps')
        self._on_json_loaded()

    # Hooks called after loading (subclasses / later steps override)
    def _on_sprite_loaded(self): pass
    def _on_json_loaded(self):   pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = AnimatorApp()
    app.mainloop()
