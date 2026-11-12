#!/usr/bin/env python3
"""photo-geolocator-01 entry: model-free photo geolocation.

Pure Python 3.13 standard library, no model calls, no network beyond the
trial env. For each photo it decodes the JPEG's DC coefficients only
(per-8x8-block average YCbCr, converted to approximate RGB), extracts
aggregate regional colour features, places the photo at the nearest of
three aggregate regional anchors fitted on the 75 public practice photos,
heavily shrunk toward the best global fixed prior, with two validated
single-feature rules (yellow arid ground shifts the guess south, a blue
mid-band shifts it west). Every photo is always answered; any per-photo
failure falls back to the fixed prior.

Deterministic: no randomness, no wall clock; answers are a pure function
of the photo bytes.
"""

import json
import os
import struct
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Fitted constants (from the 75 public practice photos only: aggregate
# colour centroids and the geographic medoid of each cluster's photos).
# No per-photo data, hashes, ids or coordinates are stored.
# Feature order: white, green, tan, sky, bright, sat, grey, warm.
FEAT_MU = (0.097, 0.119, 0.0712, 0.3198, 0.5413, 0.2731, 0.0869, -0.0837)
FEAT_SD = (0.1132, 0.148, 0.1343, 0.2278, 0.1107, 0.1327, 0.1182, 0.1085)
# (standardised centroid, medoid lat, medoid lon, cluster size)
ANCHORS = (
    ((1.059, -0.501, 0.335, -0.941, -0.1, -0.871, 0.903, 0.92), 48.18, 40.61, 23),
    ((-0.689, -0.439, 0.007, 1.425, 1.019, 1.284, -0.52, -1.566), 43.28, 3.31, 14),
    ((-0.387, 0.465, -0.205, 0.045, -0.315, 0.054, -0.355, 0.02), 63.55, 28.38, 38),
)
FALLBACK = (56.0, 26.0)  # best fixed point over the practice photos
ANCHOR_W = 0.2           # weight of the nearest anchor vs the fallback

# Yellow arid-ground rule: yellow fraction above 0.06 shifts the guess
# south, up to -60 degrees at yellow 0.15.
YELLOW_THR = 0.06
YELLOW_CAP = 0.15
YELLOW_SHIFT = -60.0
# Blue mid-band rule: mean blue of the middle third above 0.45 shifts
# the guess west, up to -40 degrees at mid-blue 0.65.
BLUE_THR = 0.45
BLUE_CAP = 0.65
BLUE_SHIFT = -40.0


# ---------------------------------------------------------------------------
# Minimal baseline-JPEG DC decoder (stdlib only).
# ---------------------------------------------------------------------------

class BitReader:
    __slots__ = ("data", "pos", "buf", "nbits")

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.buf = 0
        self.nbits = 0

    def _fill(self):
        d = self.data
        p = self.pos
        n = len(d) - p
        if n > 4:
            n = 4
        v = 0
        for i in range(n):
            v = (v << 8) | d[p + i]
        self.buf = (self.buf << (8 * n)) | v
        self.nbits += 8 * n
        self.pos += n

    def peek(self, n):
        while self.nbits < n:
            before = self.pos
            self._fill()
            if self.pos == before:
                raise ValueError("bitstream exhausted")
        if self.nbits >= n:
            return (self.buf >> (self.nbits - n)) & ((1 << n) - 1)
        return (self.buf & ((1 << self.nbits) - 1)) << (n - self.nbits)

    def drop(self, n):
        self.nbits -= n
        if self.nbits > 0:
            self.buf &= (1 << self.nbits) - 1
        else:
            self.buf = 0

    def get(self, n):
        v = self.peek(n)
        self.drop(n)
        return v


class Huff:
    """Canonical Huffman table; decode by walking bits."""

    __slots__ = ("table",)

    def __init__(self, counts, symbols):
        table = {}
        code = 0
        idx = 0
        for ln in range(1, 17):
            for _ in range(counts[ln - 1]):
                table[(ln, code)] = symbols[idx]
                idx += 1
                code += 1
            code <<= 1
        self.table = table

    def decode(self, br):
        code = 0
        table = self.table
        for ln in range(1, 17):
            code = (code << 1) | br.get(1)
            sym = table.get((ln, code))
            if sym is not None:
                return sym
        raise ValueError("bad huffman code")


def _extend(v, s):
    if s == 0:
        return 0
    if v < (1 << (s - 1)):
        return v - (1 << s) + 1
    return v


def decode_dc(data):
    """Decode DC coefficients of a baseline JPEG.

    Returns (blocks_w, blocks_h, Y, Cb, Cr, chroma_w, chroma_h) where the
    grids hold per-8x8-block average sample values (0..255).
    Raises ValueError on anything this minimal decoder cannot handle.
    """
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("not a jpeg")
    i = 2
    qt = {}
    ht = {}
    sof = None
    dri = 0
    sos = None
    scan_start = None
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xD9:
            break
        if m == 0x01 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + ln]
        if m == 0xDB:
            j = 0
            while j < len(seg):
                pq = seg[j] >> 4
                tid = seg[j] & 15
                j += 1
                if pq == 0:
                    qt[tid] = list(seg[j:j + 64])
                    j += 64
                else:
                    qt[tid] = [struct.unpack(">H", seg[j + 2 * k:j + 2 * k + 2])[0]
                               for k in range(64)]
                    j += 128
        elif m == 0xC4:
            j = 0
            while j < len(seg):
                tc = seg[j] >> 4
                tid = seg[j] & 15
                j += 1
                counts = list(seg[j:j + 16])
                j += 16
                n = sum(counts)
                symbols = list(seg[j:j + n])
                j += n
                ht[(tc, tid)] = Huff(counts, symbols)
        elif m == 0xC0:
            h, w = struct.unpack(">HH", seg[1:5])
            n = seg[5]
            comps = []
            for k in range(n):
                cid = seg[6 + 3 * k]
                hv = seg[6 + 3 * k + 1]
                tq = seg[6 + 3 * k + 2]
                comps.append((cid, hv >> 4, hv & 15, tq))
            sof = (w, h, comps)
        elif m == 0xDD:
            dri = struct.unpack(">H", seg[0:2])[0]
        elif m == 0xDA:
            sos = seg
            scan_start = i + 2 + ln
            break
        i += 2 + ln
    if sof is None or sos is None:
        raise ValueError("no SOF0/SOS")
    w, h, comps = sof
    ns = sos[0]
    scan = []
    for k in range(ns):
        cid = sos[1 + 2 * k]
        td = sos[2 + 2 * k]
        scan.append((cid, td >> 4, td & 15))
    # Entropy data: drop stuffed 0x00 bytes after 0xFF; stop at the next
    # real marker (EOI). Restart markers are skipped (their positions are
    # irrelevant: with no DRI there are none; with DRI the predictor reset
    # below re-aligns the bit reader).
    raw = data[scan_start:]
    cleaned = bytearray()
    j = 0
    while j < len(raw):
        b = raw[j]
        if b == 0xFF:
            nb = raw[j + 1] if j + 1 < len(raw) else 0
            if nb == 0x00:
                cleaned.append(0xFF)
                j += 2
                continue
            if 0xD0 <= nb <= 0xD7:
                j += 2
                continue
            break
        cleaned.append(b)
        j += 1
    br = BitReader(bytes(cleaned))
    max_h = max(c[1] for c in comps)
    max_v = max(c[2] for c in comps)
    mcux = (w + 8 * max_h - 1) // (8 * max_h)
    mcuy = (h + 8 * max_v - 1) // (8 * max_v)
    grids = {}
    for (cid, hs, vs, _tq) in comps:
        bw = (w * hs + 8 * max_h - 1) // (8 * max_h)
        bh = (h * vs + 8 * max_v - 1) // (8 * max_v)
        grids[cid] = ([0.0] * (bw * bh), bw, bh)
    comp_by_id = {c[0]: c for c in comps}
    scan_comps = []
    for (cid, td, ta) in scan:
        c = comp_by_id[cid]
        scan_comps.append((cid, ht[(0, td)], ht[(1, ta)], c[1], c[2], qt[c[3]][0]))
    dc_pred = {cid: 0 for cid, _, _, _, _, _ in scan_comps}
    mcu_count = 0
    for _my in range(mcuy):
        for _mx in range(mcux):
            if dri and mcu_count % dri == 0 and mcu_count > 0:
                dc_pred = {cid: 0 for cid in dc_pred}
                br.drop(br.nbits % 8)
            for (cid, hdc, hac, hs, vs, q0) in scan_comps:
                g, gbw, gbh = grids[cid]
                for by in range(vs):
                    for bx in range(hs):
                        sym = hdc.decode(br)
                        diff = _extend(br.get(sym), sym) if sym else 0
                        dc_pred[cid] += diff
                        # AC: consume until EOB, or until coefficient 63
                        # is filled (a block ending exactly at 63 emits
                        # no EOB).
                        k = 1
                        while k < 64:
                            s = hac.decode(br)
                            if s == 0x00:
                                break
                            run = s >> 4
                            size = s & 15
                            if s == 0xF0:
                                k += 16
                                continue
                            if size:
                                br.get(size)
                            k += run + 1
                        gx = _mx * hs + bx
                        gy = _my * vs + by
                        if gx < gbw and gy < gbh:
                            g[gy * gbw + gx] = dc_pred[cid] * q0 / 8.0 + 128.0
            mcu_count += 1
    y_grid, bw, bh = grids[comps[0][0]]
    if len(comps) == 3:
        cb_grid, cbw, cbh = grids[comps[1][0]]
        cr_grid, _crw, _crh = grids[comps[2][0]]
    else:
        cb_grid, cr_grid = y_grid, y_grid
        cbw, cbh = bw, bh
    return bw, bh, y_grid, cb_grid, cr_grid, cbw, cbh


def block_rgb(data):
    """Per-8x8-block average RGB (0..1) lists plus the block grid size."""
    bw, bh, y_grid, cb_grid, cr_grid, cbw, cbh = decode_dc(data)
    n = bw * bh
    r_list = [0.0] * n
    g_list = [0.0] * n
    b_list = [0.0] * n
    for gy in range(bh):
        cy = (gy * cbh) // bh
        base_y = gy * bw
        base_c = cy * cbw
        for gx in range(bw):
            cx = (gx * cbw) // bw
            i = base_y + gx
            j = base_c + cx
            y = y_grid[i]
            cb = cb_grid[j] - 128.0
            cr = cr_grid[j] - 128.0
            r = y + 1.402 * cr
            g = y - 0.34414 * cb - 0.71414 * cr
            b = y + 1.772 * cb
            r_list[i] = (0.0 if r < 0.0 else (255.0 if r > 255.0 else r)) / 255.0
            g_list[i] = (0.0 if g < 0.0 else (255.0 if g > 255.0 else g)) / 255.0
            b_list[i] = (0.0 if b < 0.0 else (255.0 if b > 255.0 else b)) / 255.0
    return bw, bh, r_list, g_list, b_list


# ---------------------------------------------------------------------------
# Colour features and placement.
# ---------------------------------------------------------------------------

def features(bw, bh, r_list, g_list, b_list):
    """Aggregate colour features of the photo (see module docstring)."""
    n = bw * bh
    mid0 = bh // 3
    mid1 = (2 * bh) // 3
    sum_r = sum_g = sum_b = 0.0
    sum_val = sum_sat = sum_warm = 0.0
    cnt_green = cnt_sky = cnt_white = cnt_tan = cnt_grey = cnt_yellow = 0
    sum_mid_b = 0.0
    cnt_mid = 0
    for idx in range(n):
        r = r_list[idx]
        g = g_list[idx]
        b = b_list[idx]
        sum_r += r
        sum_g += g
        sum_b += b
        mx = r if r >= g else g
        if b > mx:
            mx = b
        mn = r if r <= g else g
        if b < mn:
            mn = b
        val = mx
        sat = (mx - mn) / mx if mx > 0.0 else 0.0
        sum_val += val
        sum_sat += sat
        sum_warm += r - b
        if g > r + 0.03 and g > b + 0.02:
            cnt_green += 1
        if b > r + 0.05 and b > g + 0.02 and val > 0.35:
            cnt_sky += 1
        if val > 0.75 and sat < 0.12:
            cnt_white += 1
        if r > g + 0.03 and g >= b and 0.08 < sat < 0.45:
            cnt_tan += 1
        if sat < 0.08 and 0.15 < val < 0.75:
            cnt_grey += 1
        if r > 0.4 and g > 0.35 and b < r - 0.15 and b < g - 0.08:
            cnt_yellow += 1
        row = idx // bw
        if mid0 <= row < mid1:
            sum_mid_b += b
            cnt_mid += 1
    inv = 1.0 / n
    feat = {
        "white": cnt_white * inv,
        "green": cnt_green * inv,
        "tan": cnt_tan * inv,
        "sky": cnt_sky * inv,
        "bright": sum_val * inv,
        "sat": sum_sat * inv,
        "grey": cnt_grey * inv,
        "warm": sum_warm * inv,
        "yellow": cnt_yellow * inv,
        "midB": (sum_mid_b / cnt_mid) if cnt_mid else sum_b * inv,
    }
    return feat


def place(feat):
    """Latitude/longitude for one photo's feature dict."""
    vals = (feat["white"], feat["green"], feat["tan"], feat["sky"],
            feat["bright"], feat["sat"], feat["grey"], feat["warm"])
    best_j = 0
    best_d = None
    for j, (cent, _la, _lo, _n) in enumerate(ANCHORS):
        d = 0.0
        for k in range(8):
            z = (vals[k] - FEAT_MU[k]) / FEAT_SD[k] - cent[k]
            d += z * z
        if best_d is None or d < best_d:
            best_d = d
            best_j = j
    _cent, a_lat, a_lon, _n = ANCHORS[best_j]
    lat = ANCHOR_W * a_lat + (1.0 - ANCHOR_W) * FALLBACK[0]
    lon = ANCHOR_W * a_lon + (1.0 - ANCHOR_W) * FALLBACK[1]
    yellow = feat["yellow"]
    if yellow > YELLOW_THR:
        lat += YELLOW_SHIFT * (min(yellow, YELLOW_CAP) - YELLOW_THR) / (YELLOW_CAP - YELLOW_THR)
    mid_b = feat["midB"]
    if mid_b > BLUE_THR:
        lon += BLUE_SHIFT * (min(mid_b, BLUE_CAP) - BLUE_THR) / (BLUE_CAP - BLUE_THR)
    if lat > 90.0:
        lat = 90.0
    elif lat < -90.0:
        lat = -90.0
    if lon > 180.0:
        lon = 180.0
    elif lon < -180.0:
        lon = -180.0
    return round(lat, 6), round(lon, 6)


def place_photo(data):
    bw, bh, r_list, g_list, b_list = block_rgb(data)
    return place(features(bw, bh, r_list, g_list, b_list))


# ---------------------------------------------------------------------------
# Env client.
# ---------------------------------------------------------------------------

def _opener():
    # No proxies: the sandbox reaches only the env.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(opener, url):
    req = urllib.request.Request(url, method="GET")
    with opener.open(req, timeout=60) as resp:
        return resp.read()


def _post(opener, url, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"content-type": "application/json"})
    with opener.open(req, timeout=60) as resp:
        return resp.status, resp.read()


def main():
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        sys.stderr.write("main.py: MAKEFUN_ENV_URL is not set\n")
        return 1
    opener = _opener()
    try:
        listed = json.loads(_get(opener, env_url + "/photos").decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - nothing to post without the list
        sys.stderr.write("main.py: cannot list photos: %r\n" % (exc,))
        return 1
    photos = listed.get("photos", [])
    answers = []
    for photo in photos:
        pid = photo.get("id")
        if not pid or photo.get("answered"):
            continue
        url = photo.get("url") or ("/photos/" + pid + ".jpg")
        if not url.startswith("http"):
            url = env_url + url
        try:
            data = _get(opener, url)
            lat, lon = place_photo(data)
        except Exception as exc:  # noqa: BLE001 - fall back, never skip
            sys.stderr.write("main.py: %s: %r; using fallback\n" % (pid, exc))
            lat, lon = FALLBACK
        answers.append({"id": pid, "lat": lat, "lon": lon})
    if not answers:
        sys.stdout.write("main.py: nothing to answer\n")
        return 0
    last_exc = None
    for _attempt in range(3):
        try:
            status, body = _post(opener, env_url + "/answers", {"answers": answers})
            sys.stdout.write("main.py: answered %d photos: HTTP %d %s\n"
                             % (len(answers), status, body.decode("utf-8", "replace")[:200]))
            return 0 if 200 <= status < 300 else 1
        except urllib.error.HTTPError as exc:
            sys.stderr.write("main.py: answers refused: HTTP %d\n" % exc.code)
            return 1
        except Exception as exc:  # noqa: BLE001 - retry a failed post once
            last_exc = exc
    sys.stderr.write("main.py: answers post failed: %r\n" % (last_exc,))
    return 1


if __name__ == "__main__":
    sys.exit(main())
