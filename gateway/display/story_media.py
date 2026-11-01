"""Read-only allowlisted public story-demo media; no user media lookup."""
from pathlib import Path
import re

MEDIA = {'story-garden-demo-v1.webm': 'video/webm',
         'story-garden-demo-v1.mp4': 'video/mp4', 'story-garden-demo-v1.jpg': 'image/jpeg'}
MAX_BYTES = 5 * 1024 * 1024


def media_slice(path, range_header=None, *, directory=None):
    name = path.removeprefix('/story-media/')
    if not path.startswith('/story-media/') or name not in MEDIA:
        raise FileNotFoundError('unknown public story media')
    root = Path(directory) if directory is not None else Path(__file__).with_name('static') / 'story-media'
    target = root / name
    if target.is_symlink() or target.parent.resolve() != target.parent or not target.is_file():
        raise FileNotFoundError('public story media unavailable')
    size = target.stat().st_size
    if not 0 < size <= MAX_BYTES:
        raise FileNotFoundError('invalid public story media')
    headers = {'Content-Type': MEDIA[name], 'Accept-Ranges': 'bytes',
               'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'private, max-age=3600'}
    start, end, status = 0, size - 1, 200
    if range_header is not None:
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header)
        try:
            if not match or not any(match.groups()):
                raise ValueError
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
                if start >= size or end < start:
                    raise ValueError
            else:
                suffix = int(right)
                if suffix <= 0:
                    raise ValueError
                start = max(0, size - suffix)
            status = 206
            headers['Content-Range'] = f'bytes {start}-{end}/{size}'
        except ValueError:
            return 416, {**headers, 'Content-Range': f'bytes */{size}', 'Content-Length': '0'}, target, 0, 0
    length = end - start + 1
    return status, {**headers, 'Content-Length': str(length)}, target, start, length
