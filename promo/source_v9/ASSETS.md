# CardPlay promo v9: asset and licence log

Video: `promo/cardplay_concept_demo_v9.mp4` (57 s, 1920×1080, 30 fps). Source: `promo/source_v9/`. Build: `python3 mix.py`, then `node render.cjs`.

Changes from v8: the AI transition is rebuilt on a calibrated 3D model of the gym and lasts 2 s. Both licensed shots were calibrated into one shared room (vanishing points of the walls, f ≈ 1290 px), so the walls, floor, court lines and ceiling are real planes textured from the footage, and the rim and net are modelled in 3D. The virtual camera leaves the clip's camera, swings round to face the backboard, rises and settles exactly on the card's camera. The two takes run at the hoop from opposite directions (the clip shows his front, the card shows his back), so the player now turns in the air, like a 180 dunk. He is built as two relief shells cut from the real footage, one per take, at the same real-world size. His jersey and shorts are graded to the card's colours, and the slow motion is smoothed with optical-flow in-betweens. In the 10 s output the card moment and the stats each start 0.5 s and 0.3 s later.

Changes from v7: the AI transition is rebuilt as a 1.5 s bullet-time camera orbit. The player is a separate cut-out layer, so the background can no longer distort him. The jump plays in slow motion, first from the clip's angle and then from the card's angle of the same jump. Meanwhile the camera arcs 45° around him through the gym rebuilt in 3D.

Changes from v6: the order is reversed. The 10 s output now plays the licensed clip (44455, 1.8–4.5 s, `assets/clip/`) up to the break point, then the AI transition, and ends on the card moment. STEP 05 and the finale follow the same order. The transition is still the depth-based camera move, played in reverse, as a placeholder until a video-generation model is approved. Keyframes for that model are in `gen/`.

Changes from v5: the AI bridge is no longer a 2D morph. It is now a 1.2 s 3D camera move that orbits from the card's side angle down and around to the under-the-rim angle of the licensed clip (details below).

Changes from v4: the 3-point clip (44444) is gone, because it had nothing to do with the card moment; the 10 s output now runs card → AI bridge → dunk → slow-mo replay of the same dunk → stats. STEP 05 first plays the whole licensed take (44455, all 7.25 s) with a timeline, stops on the break point at 4.5 s, and only then generates the bridge.

Changes from v3: no licensed clip matches the card exactly, so the video now shows AI bridging the gap. A new STEP 05 "Bridge the gap" puts the card moment (side angle, mid-air) next to the closest licensed clip (under the rim, take-off) and generates the frames between them. The 10 s output and the finale both go card → AI bridge → licensed clip, with the generated frames labelled on screen.
Compiled 2026-10-24.

## Footage used

All real footage comes from one Mixkit stock series, shot in the same gym with the same model (jersey "MVP 67").
That lets one real person play the fictional player "Kai Mercer" across the card, the clips and the grid.

| Mixkit ID | Title on Mixkit | Used for | Segment used |
|---|---|---|---|
| 44448 | Skilled basketball player shooting baskets training alone | Card photo (`assets/card_photo.jpg`, side angle, mid-air) and the start frame of the AI bridge; grid tile 8 | card still at 3.8 s; tile still at 0.3 s |
| 44455 | Skilled basketball player training dunk | Whole take shown in STEP 05 (`assets/full/`); dunk clip and replay in the output and finale (`assets/dunk/`); end frame of the AI bridge (the break point) | full take 0–7.25 s at 20 fps; dunk 3.9–6.3 s at 0.8×; break point 4.5 s |
| 44444 | Talented basketball player practicing shots | Grid tiles 4 and 6 | stills |
| 44447 | Basketball player shooting baskets while training alone | Grid tile 0 | still at 1.3 s |
| 44465 | Talented basketball player scores against a player | Grid tile 2 | still at 3.4 s |
| 44462 | One vs one between two basketball players | Grid tile 3 | still at 4.0 s |
| 44471 | Two professional basketball players playing a match | Grid tiles 5 and 9 | stills at 2.5 s and 1.9 s |
| 44446 | Portrait of a basketball player practicing shots | Grid tile 7 | still at 5.8 s |
| 44464 | Pair of good basketball players playing a match | Grid tile 10 | still at 2.5 s |
| 44445 | Experienced basketball player making a basket | Grid tile 11 | still at 0.8 s |

- **Source pages:** `https://mixkit.co/free-stock-video/<title-slug>-<ID>/`, for example https://mixkit.co/free-stock-video/skilled-basketball-player-training-dunk-44455/
- **Files:** downloaded as 1920×1080 MP4 from `https://assets.mixkit.co/videos/<ID>/<ID>-1080.mp4`.
- **Edits:** cropped to 9:16, scaled, with a light contrast and saturation grade.
- **Licence:** Mixkit Stock Video Free License (https://mixkit.co/license/#videoFree). Each item page says the clip is a "free stock video clip for commercial or personal use, under the Mixkit Stock Video Free License". No attribution is required. The video credits Mixkit on the end card anyway.

## Music and sound effects (all Mixkit)

| Mixkit ID | Title | Licence | Used for |
|---|---|---|---|
| Music 369 | "Need for speed" by Arulo (trap, 92 BPM) | Mixkit Stock Music Free License | Soundtrack, from 9.35 s in the track so the beat drops on STEP 01 |
| SFX 1492 | Cinematic whoosh fast transition | Sound Effects Free License | Scene transitions |
| SFX 1490 | Fast whoosh transition | Sound Effects Free License | Clip changes |
| SFX 166 | Fast small sweep transition | Sound Effects Free License | Scan sweeps |
| SFX 2350 | Magic sparkle whoosh | Sound Effects Free License | Card swings in (opening and finale) |
| SFX 2633 | Sweeping sparkle presentation intro | Sound Effects Free License | Card reveal on the phone |
| SFX 3082 | Magic sparkle poof hit | Sound Effects Free License | Card outro on the phone |
| SFX 1432 | Camera digital shutter | Sound Effects Free License | Snap |
| SFX 1107 | Page forward single chime | Sound Effects Free License | Identified, check passed, rendered |
| SFX 2568 | Cool interface click tone | Sound Effects Free License | Field boxes and checklist ticks |
| SFX 2573 | Interface option select | Sound Effects Free License | Clip grid, share sheet |
| SFX 2537 | Laptop keyboard typing sequence | Sound Effects Free License | Shot list typing |
| SFX 2093 | Basketball ball, hard hit | Sound Effects Free License | Dunk slam |
| SFX 462 | Huge crowd cheering victory | Sound Effects Free License | Crowd after the dunk and the finale |
| SFX 2908 | Movie trailer epic impact | Sound Effects Free License | Finale freeze on the card |

- **Files:** music from `https://assets.mixkit.co/music/369/369.mp3`; effects from `https://assets.mixkit.co/active_storage/sfx/<ID>/<ID>.wav`. Copies are in `audio/`.
- **Licences:** the music page's metadata names the "Mixkit Stock Music Free License" (https://mixkit.co/license/#musicFree). Every sound-effects listing page I used is under the "Sound Effects Free License" (https://mixkit.co/license/#sfxFree).
- **Before publishing:** save PDFs of those two licence pages too. If a platform flags the music, the saved licence page is your proof of licence.

## AI transition frames (generated, illustrative)

- **What:** `assets/bridge/` holds 60 frames (2.0 s), stored card-first and played in reverse in the video. They take the camera from the clip's break point (44455 at 4.51 s, under the rim) to the card moment (44448 at 3.8 s, side angle from behind).
- **3D gym:** both shots were calibrated into one room from the vanishing points of the walls (OpenCV line detector and RANSAC). Floor, far wall, end wall, ceiling and backboard are planes whose textures are baked from the two frames, with the player removed using other frames of the same clips. Parts neither camera saw are filled by repeating the wall's own bays, or by inpainting. The rim, net and the board's mount are modelled. It is rendered in WebGL with three.js (MIT), with 6 sub-frames per frame for real motion blur (`tools/atlas.py`, `tools/endwall.py`, `tools/room.html`, `tools/path9.py`).
- **Player:** cut out of 44455 at 4.51–4.6 s (front) and of 44448 at 3.4–3.8 s (back) with Mask R-CNN and Keypoint R-CNN (torchvision, COCO weights). Wall pixels trapped between his raised arms are removed by differencing against the clean plate. Each cut-out becomes a relief shell (inflated from its silhouette) placed at his real position and size in the room. He turns 196° in the air from the clip's pose into the card's pose. Slow-motion in-betweens use RAFT optical flow (torchvision, BSD-3-Clause). His jersey and shorts are graded from the clip's colours to the card's (`tools/player9.py`, `tools/trans9.py`).
- **Rights:** every pixel comes from the two licensed Mixkit clips above. The models are only used as tools.
- **Labelling:** the transition carries an "AI transition" badge on screen.
- **Limits:** the two takes are different jumps, so the turn itself is synthesised. For about two frames while he turns side-on the two shells are blended, hidden by the motion blur of the camera swing. A first-and-last-frame video model could generate a truer turn, so the end card still calls the frames illustrative. The keyframes for that model are in `gen/`.

## Generative test (not used in the video)

- `gen/wan_test/` holds 17 frames (320×576) from Wan 2.1 VACE 1.3B (Apache-2.0, https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B), given only the two keyframes in `gen/` and a text prompt. It took about 2 hours on this machine's CPU. `gen/compare_v9_vs_wan.mp4` shows it next to the v9 transition.

## Not taken from stock

- **Card design, UI and logo:** made in code (`scene.html`).
- **Names:** "Kai Mercer", "Harbor City Comets", "Prime" and "CardPlay" are invented placeholders.
- **No marks:** no NBA, team or league marks, and no real pro athletes.

## Before external use, check these

1. **Save the licence text.** I couldn't capture the full Mixkit licence text automatically, because the page loads it in a pop-up. Save a PDF of https://mixkit.co/license/#videoFree with today's date, as a record of the terms you used.
2. **The model is a real person under a made-up name.** Stock licences usually forbid using a recognisable model in a way that suggests they personally endorse a product, or in a misleading or offensive context. A clearly labelled concept demo is a normal stock use. Do not present him as a real athlete or as a spokesperson for the brand. If the video becomes paid advertising, re-read the Mixkit terms for this point.
3. **AI-generated frames of a real person.** Here the person is a stock model, and the bridge is a short interpolation between two licensed shots of him. In the real product the frames would show NBA players, so the NBPA (PLYRS UNTD) and NBA licences must explicitly cover AI-generated frames of the athlete, not just reuse of footage. Keep generated frames labelled and attach C2PA content credentials to every output.
4. **The ball shows a manufacturer's printing.** It's small and not readable at video size. That's fine for a demo, but crop or blur it if a licensor asks.

## Sources checked and not used

- **Pexels and Pixabay:** reachable, but searching needs an API key, and Pexels has paused new keys.
- **Wikimedia Commons:** API rate-limited from the cloud network. Most dunk photos there show named college or pro players, which raises personality and league-rights issues.
- **StockSnap (CC0, via Openverse):** good stills of courts, hoops and balls, but no game action. Kept as a fallback for backgrounds.
