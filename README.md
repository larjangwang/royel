# วีดีโอจำลองการเดินถวายดอกไม้จันทน์ (แถวตอนเรียง 4)

4-Abreast Ceremonial Procession Illustration — an illustrated walkthrough video and
presentation page showing the walking route for the sandalwood-flower offering ceremony:
entry screening, seated waiting in dome tents, forming 4-abreast rows along the tent
fronts, splitting 2+2 around the central fountain, offering at the trays in front of the
pavilion, then exiting directly left (north in the plan — never backtracking into the
approach lane) around the stage edge and along the front boundary to the left venue exit.

## Run locally

- **XAMPP (Windows):** place/clone at `C:/xampp/htdocs/royal` with Apache running,
  then open <http://localhost/royal/>
- **Any static server:** from inside the cloned repo run
  `python -m http.server 8090` and open <http://localhost:8090>

## Usage

`index.html` is the presentation page: the video player supports play/pause,
chapter jump buttons (0:00 / 0:10 / 0:18 / 0:28), playback speed, fullscreen
presentation mode, and a download link for the MP4.

## Video outputs

Two identical MP4 files (Thai-named copy + English-named copy), each
40 seconds, 1920x1080, 30 fps (H.264):

- `simulation_video.mp4`
- `วีดีโอจำลองการเดินถวายดอกไม้จันทน์.mp4`

The walking route follows the annotated reference image `pic/walking_route_new.jpg`
(also viewable as a thumbnail on the page).

## Regenerating the video (optional)

`generate_video.py` renders all frames with Pillow and pipes them to FFmpeg.
People are procedural shaded-3D sprites drawn by `human_renderer.py` (analytic
ellipsoid ray-casting, Pillow/numpy only — not scanned or photographed people).
Requirements on Windows:

- Python with `Pillow` and `numpy`
- FFmpeg (path is hardcoded in `main()` — adjust for your machine)
- Windows Tahoma fonts (`C:\Windows\Fonts\tahoma*.ttf`)
- `BASE_DIR` is hardcoded to `C:/xampp/htdocs/royal` — adjust if placed elsewhere

Run: `python generate_video.py` (renders ~1200 frames; writes a pending MP4 first and
only replaces the published videos on success).

The axial (middle-walkway) view also shows completed-offering participants
returning down the right-hand corridor in 4-abreast rows after the initial
left departure.

Note: the illustration is a route/presentation aid — people counts, timing, and flow
are illustrative, not a throughput measurement.
