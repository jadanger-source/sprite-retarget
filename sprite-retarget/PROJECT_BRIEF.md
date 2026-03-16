# Sprite Motion Retarget — Project Brief

## Upload these files with this README to start a new conversation:
- motion-capture.html (DONE, works perfectly)
- motion_data.json (exported from motion capture app)  
- bodywhole1.png (the sprite)
- dancinghappydance.gif (reference dance)

## Paste this prompt:

I'm building a 2D sprite motion retargeting tool split into two apps. The motion capture app (motion-capture.html) is DONE and works perfectly — it uses MediaPipe Pose to extract 15-joint pose data from video at configurable FPS/smoothing and exports JSON.

The sprite animator app needs to be REBUILT FROM SCRATCH. It loads a sprite PNG + pose JSON and animates the sprite to match the dance, then exports a sprite sheet.

## 15-Joint System
Joints: Head(0), L Shoulder(1), R Shoulder(2), L Elbow(3), R Elbow(4), L Wrist(5), R Wrist(6), L Hip(7), R Hip(8), L Knee(9), R Knee(10), L Ankle(11), R Ankle(12), Neck(13, virtual avg of shoulders), Pelvis(14, virtual avg of hips)

MediaPipe indices mapped: [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] + 2 virtual

## Critical Lessons From 20+ Iterations

### What DOESN'T work:
1. **LBS mesh deformation** — spreads rotation across the body, warps torso/head, distorts proportions. Every attempt to fix with weight tuning (k=12, k=20, k=40) either dampens movement or creates worse distortion.
2. **Rigid parts with masking/dilation** — preserves shape perfectly but creates unfixable gaps at joints. Dilation, overlap zones, and joint patches all failed to seamlessly connect parts.
3. **Pure bone-distance weights** — head bone [neck→head] passes through chest, giving head ownership of shoulder pixels. Zone-based segmentation (cutting at neck line, hip line) works much better.

### What DOES work:
1. **Motion capture app** — MediaPipe with modelComplexity:2, low smoothing captures excellent pose data
2. **FK with pelvis root** — pelvis(14)→spine→neck→head/shoulders→arms, pelvis→hips→legs. Bone lengths preserved exactly.
3. **Angle unwrapping** — CRITICAL. atan2 deltas cross ±180° boundary causing fake 260° rotations. Must unwrap: `while(d>PI) d-=2*PI; while(d<-PI) d+=2*PI`
4. **L/R mirroring** — MediaPipe "Left" = person's left = viewer's RIGHT. Sprite "L Shoulder" = viewer's LEFT. Must swap indices [0,2,1,4,3,6,5,8,7,10,9,12,11,13,14] and flip x→1-x.
5. **Canvas triangle texture mapping** — affine transform per triangle works well for rendering deformed mesh
6. **Delaunay triangulation** (Bowyer-Watson) — proven working, handles 200+ vertices in <30ms
7. **Zone-based segmentation** — head=above neck line within head width, torso=between neck/hip lines within torso x-bounds, limbs=nearest limb bone

### The Correct Architecture (not yet implemented):
Each pixel belongs to a BONE (not a joint). 10 bones:
- head(0): neck→head, rotates around neck
- torso(1): pelvis→neck, rotates around pelvis  
- upperArmL(2): shoulder→elbow, rotates around shoulder
- forearmL(3): elbow→wrist, rotates around elbow
- upperArmR(4), forearmR(5), thighL(6), shinL(7), thighR(8), shinR(9)

Each bone's transform: rotate the pixel around the bone's PARENT JOINT by the bone's angle delta. The pixel stays at a fixed offset from the parent joint, just rotated.

At joints, a narrow deformable mesh band (15-20px) bridges two adjacent bones with blended rotation. This is the ONLY place deformation occurs.

Arms should draw ON TOP of torso (z-ordering triangle sort).

Global Y translation from pelvis joint for hip bounce. No other spatial translation.

### Sprite Proportions
The sprite (785×1302) has longer arms relative to torso than the video kid. The FK preserves sprite bone lengths while applying video angles, which is correct — we WANT the sprite's proportions maintained. Don't try to scale or adjust for proportion differences.

### Default Joint Positions (normalized, tuned to bodywhole1.png)
```
Head: 0.50, 0.08    L Shoulder: 0.22, 0.33    R Shoulder: 0.77, 0.33
L Elbow: 0.15, 0.49  R Elbow: 0.84, 0.49      L Wrist: 0.11, 0.63
R Wrist: 0.87, 0.63  L Hip: 0.36, 0.49         R Hip: 0.60, 0.49
L Knee: 0.33, 0.78   R Knee: 0.64, 0.78        L Ankle: 0.31, 0.92
R Ankle: 0.67, 0.94  Neck: 0.50, 0.28          Pelvis: 0.50, 0.49
```

### UI Features Needed
- Drop zones for sprite PNG and pose JSON
- Draggable joints on sprite view
- Animate button with progress bar
- Preview with playback controls
- Export sprite sheet (FPS, columns configurable)
- Scale and mesh density sliders

Build the sprite animator HTML file. Test the deformation visually if possible before shipping.
