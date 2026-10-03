"""Render frames: preview PNGs or final video segments (one ffmpeg per worker)."""
import argparse
import os
import subprocess
import sys
import time
import multiprocessing as mp
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_R = None


def _init(snap, W, H):
    global _R
    import render
    _R = render.Renderer(render.Scene(snap), W, H)


def _png(args):
    f, out = args
    t0 = time.time()
    img = _R.render(f)
    cv2.imwrite(os.path.join(out, f'f{f:04d}.png'), img)
    return f, time.time() - t0


def _segment(args):
    frames, path, W, H, fps, crf = args
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
           '-profile:v', 'high', '-x264-params', 'keyint=60:min-keyint=60:scenecut=0', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in frames:
        img = _R.render(f)
        p.stdin.write(np.ascontiguousarray(img).tobytes())
        if f % 10 == 0:
            print(f'frame {f} ({time.time() - t0:.0f}s)', flush=True)
    p.stdin.close()
    p.wait()
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('snap')
    ap.add_argument('out')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--frames', default='0:900:1')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--crf', type=int, default=16)
    a = ap.parse_args()
    W = a.width
    H = W * 9 // 16
    os.makedirs(a.out, exist_ok=True)
    s0, s1, st = (int(x) for x in a.frames.split(':'))
    frames = list(range(s0, s1, st))
    with mp.Pool(a.workers, initializer=_init, initargs=(a.snap, W, H)) as pool:
        if not a.video:
            for f, dt in pool.imap_unordered(_png, [(f, a.out) for f in frames]):
                print(f'frame {f} {dt:.1f}s', flush=True)
        else:
            nseg = max(a.workers * 3, 1)
            chunks = [c for c in np.array_split(frames, nseg) if len(c)]
            jobs = [(list(c), os.path.join(a.out, f'seg{i:03d}.mp4'), W, H, 30, a.crf) for i, c in enumerate(chunks)]
            segs = pool.map(_segment, jobs, chunksize=1)
            with open(os.path.join(a.out, 'list.txt'), 'w') as fh:
                for sgm in segs:
                    fh.write(f"file '{os.path.abspath(sgm)}'\n")
