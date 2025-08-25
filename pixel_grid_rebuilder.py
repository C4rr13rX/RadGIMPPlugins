#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtualize Grid — color-faithful pixel-grid rebuilder (GIMP 3)
(unchanged algorithms; async multithreaded preview with progress bar)
Fix: if Finalize is on and Expand canvas is off, paint a canvas-sized uniform grid
Default k(px) = 23
"""

import sys, math, statistics, traceback, os, threading, concurrent.futures
import gi
gi.require_version("Gimp", "3.0")
gi.require_version("Gegl", "0.4")
gi.require_version("Gtk",  "3.0")
gi.require_version("GdkPixbuf", "2.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gimp, Gegl, Gtk, GdkPixbuf, GimpUi, GLib, Gdk, cairo

PROC_NAME = "python-fu-virtualize-grid-radiate"

# ---------------- small utils ----------------
def idx(x,y,w): return ((y*w)+x)*4
def clamp(v,lo,hi): return lo if v<lo else (hi if v>hi else v)
def normalize(v):
    m = max(v) if v else 0.0
    return [0.0]*len(v) if m<=1e-12 else [x/m for x in v]
def smooth_ma(sig, r):
    if r<=0: return sig[:]
    n=len(sig); out=[0.0]*n; pref=[0.0]*(n+1)
    for i,x in enumerate(sig): pref[i+1]=pref[i]+x
    for i in range(n):
        lo=max(0,i-r); hi=min(n-1,i+r)
        out[i]=(pref[hi+1]-pref[lo])/(hi-lo+1)
    return out
def q5(r,g,b):  # 5-bit per channel key
    return ((r>>3)<<10) | ((g>>3)<<5) | (b>>3)

# ----------- Lab for ΔE*ab -------------
def srgb_to_lin(c):
    cs=c/255.0
    return cs/12.92 if cs<=0.04045 else ((cs+0.055)/1.055)**2.4
def rgb_to_lab(r,g,b):
    rl,gl,bl = srgb_to_lin(r), srgb_to_lin(g), srgb_to_lin(b)
    X = rl*0.4124564 + gl*0.3575761 + bl*0.1804375
    Y = rl*0.2126729 + gl*0.7151522 + bl*0.0721750
    Z = rl*0.0193339 + gl*0.1191920 + bl*0.9503041
    X/=0.95047; Z/=1.08883
    def f(t): return t**(1/3) if t>0.008856 else 7.787*t + 16/116
    fx,fy,fz = f(X), f(Y), f(Z)
    return (116*fy-16, 500*(fx-fy), 200*(fy-fz))
def deltaE(c1,c2):
    L1,a1,b1=rgb_to_lab(*c1); L2,a2,b2=rgb_to_lab(*c2)
    return math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)

# ---------- k estimation ----------
def best_offset(sig, k):
    n=len(sig); besto=0; beste=-1.0
    for o in range(max(1,k)):
        s=c=0; x=o
        while x<n:
            s+=sig[x]; c+=1; x+=k
        if c and s/c>beste: beste=s/c; besto=o
    return besto,beste
def score_k(col,row,k):
    ox,ev=best_offset(col,k); oy,eh=best_offset(row,k)
    return ox,oy,(0.5*(ev+eh))/max(1.0, math.sqrt(k))
def sweep_k(col,row,kmin,kmax):
    best=(kmin,0,0); bestv=-1e9
    for k in range(kmin,kmax+1):
        ox,oy,s = score_k(col,row,k)
        if s>bestv: best=(k,ox,oy); bestv=s
    return best
def shrink_divisor(col,row,k,keep=0.94,divs=(2,3,4)):
    ox,oy,s0 = score_k(col,row,k); out=(k,ox,oy)
    for d in divs:
        kd=max(2,k//d)
        if kd==k: continue
        oxd,oyd,sd = score_k(col,row,kd)
        if sd >= keep*s0: out=(kd,oxd,oyd)
    return out

def estimate_k_runlength(src,w,h,maxk=128, step=2):
    hist=[0]*(maxk+1)
    # rows
    for y in range(0,h,step):
        off=idx(0,y,w)
        prev=q5(src[off],src[off+1],src[off+2]); run=1
        for x in range(1,w):
            j=idx(x,y,w); q=q5(src[j],src[j+1],src[j+2])
            if q==prev: run+=1
            else:
                if 2<=run<=maxk: hist[run]+=1
                prev=q; run=1
        if 2<=run<=maxk: hist[run]+=1
    # cols
    for x in range(0,w,step):
        j=idx(x,0,w)
        prev=q5(src[j],src[j+1],src[j+2]); run=1
        for y in range(1,h):
            i=idx(x,y,w); q=q5(src[i],src[i+1],src[i+2])
            if q==prev: run+=1
            else:
                if 2<=run<=maxk: hist[run]+=1
                prev=q; run=1
        if 2<=run<=maxk: hist[run]+=1
    return max(range(2,maxk+1), key=lambda r: hist[r]) if any(hist[2:]) else 2

def build_edge_energies(src,w,h):
    # simple Sobel over luma + adj RGB diffs, then normalize
    L=[0]*(w*h)
    for y in range(h):
        o=idx(0,y,w); base=y*w
        for x in range(w):
            r,g,b = src[o],src[o+1],src[o+2]
            L[base+x] = (54*r + 183*g + 19*b) >> 8
            o+=4
    Ex=[0.0]*w; Ey=[0.0]*h
    for y in range(1,h-1):
        row=y*w
        for x in range(1,w-1):
            p00=L[row-w+x-1]; p01=L[row-w+x]; p02=L[row-w+x+1]
            p10=L[row  +x-1];                  p12=L[row  +x+1]
            p20=L[row+w+x-1]; p21=L[row+w+x]; p22=L[row+w+x+1]
            gx=(p02+2*p12+p22)-(p00+2*p10+p20)
            gy=(p20+2*p21+p22)-(p00+2*p01+p02)
            Ex[x]+=abs(gx); Ey[y]+=abs(gy)
    Ex_adj=[0.0]*w; Ey_adj=[0.0]*h
    for x in range(1,w):
        s=0
        for y in range(h):
            a=idx(x-1,y,w); b=a+4
            s+=abs(src[a]-src[b])+abs(src[a+1]-src[b+1])+abs(src[a+2]-src[b+2])
        Ex_adj[x]=s
    for y in range(1,h):
        s=0; base=idx(0,y,w); prev=base-(w*4)
        for x in range(w):
            a=prev+x*4; b=base+x*4
            s+=abs(src[a]-src[b])+abs(src[a+1]-src[b+1])+abs(src[a+2]-src[b+2])
        Ey_adj[y]=s
    return normalize(Ex), normalize(Ey), normalize(Ex_adj), normalize(Ey_adj)

def _fallback_lines(L, k0):
    k0=max(2,int(k0))
    n=max(2, int(round(L/float(k0))))
    lines=[0]
    pos=k0
    for _ in range(n-1):
        lines.append(clamp(int(round(pos)), lines[-1]+1, L-1))
        pos += k0
    lines.append(L)
    return lines

def pick_gridlines_greedy(E, L, k0, slack=0.2):
    try:
        k0=max(2,int(k0)); slack=max(0.05,min(0.5,float(slack)))
        Nc=max(2, int(round(L/float(k0)))); M=Nc-1
        smin=max(3, int(round(k0*(1.0-slack))))
        last_idx = max(0, len(E)-1)

        lines=[0]; jprev=0; pos=float(k0)
        for m in range(M):
            hi_stop = min(L-(M-m)*smin, last_idx)
            lo = clamp(int(round(pos-k0*slack)), jprev+smin, hi_stop)
            hi = clamp(int(round(pos+k0*slack)), lo,           hi_stop)
            best=lo; val=-1e9
            for t in range(lo, hi+1):
                v=E[t]
                if v>val: val=v; best=t
            lines.append(best); jprev=best; pos=best+k0

        lines.append(L)
        return lines
    except Exception:
        return _fallback_lines(L, k0)

# ---------- sampling (alpha-aware modal) ----------
def core_box(x0,y0,x1,y1,k,coh):
    m = int(round(k*(0.35 - 0.25*(coh/100.0))))
    X0=clamp(x0+m, x0, x1-1); Y0=clamp(y0+m, y0, y1-1)
    X1=clamp(x1-m, X0+1, x1);  Y1=clamp(y1-m, Y0+1, y1)
    return X0,Y0,X1,Y1

def sample_tile_modal_alpha(src,w, x0,y0,x1,y1, k_local, cohesion):
    X0,Y0,X1,Y1 = core_box(x0,y0,x1,y1, k_local, cohesion)
    bins = {}   # key -> [w, wr, wg, wb]
    weight_sum=0.0
    for y in range(Y0,Y1):
        o=idx(X0,y,w)
        for x in range(X0,X1):
            r,g,b,a = src[o],src[o+1],src[o+2],src[o+3]
            if a<8: o+=4; continue
            wgt = (a/255.0)**2
            key = q5(r,g,b)
            ent = bins.get(key)
            if ent is None: bins[key]=[wgt, wgt*r, wgt*g, wgt*b]
            else:
                ent[0]+=wgt; ent[1]+=wgt*r; ent[2]+=wgt*g; ent[3]+=wgt*b
            weight_sum += wgt
            o+=4
    if not bins or weight_sum<=1e-9:
        return (0,0,0), 0.0
    key = max(bins.keys(), key=lambda k: bins[k][0])
    wsum, wr, wg, wb = bins[key]
    color = (int(round(wr/wsum)), int(round(wg/wsum)), int(round(wb/wsum)))
    coverage = wsum/weight_sum
    return color, coverage

# ---------- analysis (multithreaded rows) ----------
def analyze_tiles_parallel(src,w,h,Gx,Gy,coh, workers, cancel_event, progress_cb=None):
    rows=len(Gy)-1; cols=len(Gx)-1
    colors=[[ (0,0,0) for _ in range(cols)] for _ in range(rows)]
    if rows<=0 or cols<=0:
        return colors

    def work_row(r):
        if cancel_event.is_set(): return None
        y0,y1=Gy[r],Gy[r+1]; ky=y1-y0
        row_cols=[]
        for c in range(cols):
            if cancel_event.is_set(): return None
            x0,x1=Gx[c],Gx[c+1]; k_local=min(ky, x1-x0)
            col,_ = sample_tile_modal_alpha(src,w,x0,y0,x1,y1,k_local,coh)
            row_cols.append(col)
        return (r,row_cols)

    done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work_row, r): r for r in range(rows)}
        for fut in concurrent.futures.as_completed(futs):
            if cancel_event.is_set(): break
            res = fut.result()
            if res is None: continue
            r,row_cols = res
            colors[r] = row_cols
            done += 1
            if progress_cb:
                progress_cb(0.30 + 0.50*(done/rows), "Sampling tiles…")
    return colors

# ---------- conservative anti-halo ----------
def snap_edges(colors, cleanup):
    H=len(colors); W=len(colors[0]) if H else 0
    out=[row[:] for row in colors]
    dE_thr = 12 if cleanup>=50 else 10
    for r in range(H):
        for c in range(W):
            me = colors[r][c]
            nbrs=[]
            if c-1>=0: nbrs.append(colors[r][c-1])
            if c+1<W:  nbrs.append(colors[r][c+1])
            if r-1>=0: nbrs.append(colors[r-1][c])
            if r+1<H:  nbrs.append(colors[r+1][c])
            nearest=None; best=1e9
            for nb in nbrs:
                d=deltaE(me, nb)
                if d<best: best=d; nearest=nb
            agree=sum(1 for nb in nbrs if nearest and deltaE(nearest, nb)<=dE_thr)
            if nearest is not None and best<=dE_thr and agree>=2:
                out[r][c]=nearest
    return out

# ---------- paint ----------
def repaint_irregular(colors,Gx,Gy,w,h):
    out=bytearray(w*h*4)
    cols=len(Gx)-1; rows=len(Gy)-1
    if cols<=0 or rows<=0: return out
    widths=[Gx[i+1]-Gx[i] for i in range(cols)]
    for r in range(rows):
        y0,y1=Gy[r],Gy[r+1]
        band=bytearray(w*4); off=0
        for (R,G,B),seg in zip(colors[r], widths):
            if seg<=0: continue
            block=bytes((R,G,B,255))*seg
            band[off:off+seg*4]=block; off+=seg*4
        for y in range(y0,y1):
            dst=idx(0,y,w); out[dst:dst+w*4]=band
    return out

def repaint_uniform(colors, kf):
    rows=len(colors); cols=len(colors[0]) if rows else 0
    W=cols*kf; H=rows*kf
    out=bytearray(W*H*4)
    for r in range(rows):
        band=bytearray(W*4); off=0
        for c in range(cols):
            R,G,B = colors[r][c]
            seg=bytes((R,G,B,255))*kf
            band[off:off+kf*4]=seg; off+=kf*4
        for y in range(kf):
            dst=((r*kf+y)*W)*4; out[dst:dst+W*4]=band
    return out,W,H

def repaint_uniform_to_canvas(colors, w, h):
    """
    Paint a uniform grid that exactly fits the current canvas (w,h),
    regardless of the nominal kf. Uses equal partitions per row/col.
    """
    rows=len(colors); cols=len(colors[0]) if rows else 0
    out=bytearray(w*h*4)
    if rows<=0 or cols<=0 or w<=0 or h<=0:
        return out
    # integer partitions (sum of widths/heights == w/h)
    x_edges=[(i*w)//cols for i in range(cols+1)]
    y_edges=[(j*h)//rows for j in range(rows+1)]
    widths=[x_edges[i+1]-x_edges[i] for i in range(cols)]
    heights=[y_edges[j+1]-y_edges[j] for j in range(rows)]
    for r in range(rows):
        band=bytearray(w*4); off=0
        for c in range(cols):
            ww = widths[c]
            if ww<=0: continue
            R,G,B = colors[r][c]
            block=bytes((R,G,B,255))*ww
            band[off:off+ww*4]=block; off+=ww*4
        y0=y_edges[r]; y1=y_edges[r+1]
        for y in range(y0,y1):
            dst=idx(0,y,w); out[dst:dst+w*4]=band
    return out

# ---------- defaults ----------
def auto_defaults(src,w,h):
    kmax=min(128, max(8, min(w,h)//4))
    k_rl = estimate_k_runlength(src,w,h,maxk=kmax,step=2)
    Ex_s,Ey_s,Ex_a,Ey_a = build_edge_energies(src,w,h)
    col = normalize([0.6*xs+0.4*xa for xs,xa in zip(Ex_s,Ex_a)])
    row = normalize([0.6*ys+0.4*ya for ys,ya in zip(Ey_s,Ey_a)])
    k_sw,_,_ = sweep_k(col,row,2,kmax)
    k0 = int(round((k_rl+k_sw)/2)) if abs(k_rl-k_sw)<=max(2,(k_rl+k_sw)//12) else min(k_rl,k_sw)
    k0,_,_ = shrink_divisor(col,row,k0,keep=0.94)
    return max(2,k0), (Ex_s,Ey_s,Ex_a,Ey_a)

# ---------- preview dialog ----------
class Preview(Gtk.Dialog):
    def __init__(self, src,w,h,k0,energies):
        super().__init__(title="Virtualize Grid — Preview",
                         flags=Gtk.DialogFlags.MODAL,
                         buttons=(Gtk.STOCK_CANCEL,Gtk.ResponseType.CANCEL,
                                  Gtk.STOCK_APPLY, Gtk.ResponseType.OK))
        try:
            screen=Gdk.Screen.get_default(); sw,sh=screen.get_width(),screen.get_height()
        except Exception:
            sw,sh=1600,900
        self.set_default_size(min(1100,sw-180), min(720,sh-180))

        self.src=src; self.w=w; self.h=h
        self.k=int(k0); self.coh=80; self.cln=70; self.slack=20
        self.finalize=True; self.expand=True
        self.Ex_s,self.Ey_s,self.Ex_a,self.Ey_a=energies
        self.pb=None
        self._deb=None
        self._first_paint_done=False

        # async job management
        self._job_id = 0
        self._cancel_event = threading.Event()
        self._worker = None

        box=self.get_content_area()
        v=Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8); v.set_border_width(10); box.add(v)
        row=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8); v.pack_start(row, False, False, 0)

        # k(px): Entry + iconic buttons
        row.pack_start(Gtk.Label(label="k (px):"), False, False, 0)
        self.k_entry = Gtk.Entry(); self.k_entry.set_width_chars(5); self.k_entry.set_text(str(self.k))
        row.pack_start(self.k_entry, False, False, 0)
        def set_k(val):
            try: v=int(val)
            except: v=self.k
            self.k=clamp(v, 2, max(2,min(self.w,self.h)//2))
            self.k_entry.set_text(str(self.k))
            self._debounce()
        for label,cb in (("–", lambda: set_k(self.k-1)),
                         ("+",  lambda: set_k(self.k+1)),
                         ("÷2", lambda: set_k(max(2,self.k//2))),
                         ("×2", lambda: set_k(min(max(2,min(self.w,self.h)//2), self.k*2)))):
            b=Gtk.Button(label=label); b.connect("clicked", lambda _b, f=cb: f()); row.pack_start(b, False, False, 0)
        def apply_entry(*_): set_k(self.k_entry.get_text()); return False
        self.k_entry.connect("activate", apply_entry)
        self.k_entry.connect("focus-out-event", apply_entry)

        # Cohesion / Cleanup
        def spin(lo,hi,val,wc=64):
            adj=Gtk.Adjustment(val, lo, hi, 1, 5, 0)
            sb=Gtk.SpinButton(adjustment=adj, climb_rate=1, digits=0); sb.set_width_chars(5); sb.set_size_request(wc,-1)
            return sb,adj
        row.pack_start(Gtk.Label(label="Cohesion (%):"), False, False, 8)
        self.sbC,self.adjC=spin(0,100,self.coh); row.pack_start(self.sbC, False, False, 0)
        row.pack_start(Gtk.Label(label="Cleanup (%):"), False, False, 8)
        self.sbE,self.adjE=spin(0,100,self.cln); row.pack_start(self.sbE, False, False, 0)

        self.chkF=Gtk.CheckButton(label="Finalize to uniform squares"); self.chkF.set_active(True)
        self.chkX=Gtk.CheckButton(label="Expand canvas on Apply");      self.chkX.set_active(True)
        row.pack_start(self.chkF, False, False, 10); row.pack_start(self.chkX, False, False, 6)

        # Progress bar
        self.pbar = Gtk.ProgressBar()
        self.pbar.set_show_text(True)
        v.pack_start(self.pbar, False, False, 0)

        # Preview area: DrawingArea
        frame=Gtk.Frame(label="Rebuilt Preview (fit to view)")
        frame.set_hexpand(True); frame.set_vexpand(True)
        self.da = Gtk.DrawingArea()
        self.da.set_hexpand(True); self.da.set_vexpand(True)
        frame.add(self.da); v.pack_start(frame, True, True, 0)

        # signals
        self.adjC.connect("value-changed", self._on_change, "coh")
        self.adjE.connect("value-changed", self._on_change, "cln")
        self.chkF.connect("toggled", lambda *_: self._toggle("finalize", self.chkF.get_active()))
        self.chkX.connect("toggled", lambda *_: self._toggle("expand", self.chkX.get_active()))
        self.da.connect("draw", self._on_draw)
        self.da.connect("size-allocate", lambda *_: self.da.queue_draw())
        self.connect("delete-event", self._on_delete)

        self.connect("map",     lambda *_: GLib.idle_add(self._first_paint))
        self.connect("realize", lambda *_: GLib.idle_add(self._first_paint))

        self.show_all()
        GLib.idle_add(self._first_paint)

    # ----- progress + busy helpers -----
    def _set_progress_ui(self, frac, text):
        try:
            self.pbar.set_fraction(clamp(frac,0.0,1.0))
            self.pbar.set_text(text or "")
        except Exception:
            pass
        return False
    def _progress(self, frac, text):
        GLib.idle_add(self._set_progress_ui, frac, text)

    def _busy(self, on=True):
        try:
            win = self.get_window()
            if not win: return
            display = Gdk.Display.get_default()
            if on:
                cur = Gdk.Cursor.new_from_name(display, "watch") or Gdk.Cursor.new_from_name(display, "wait")
                win.set_cursor(cur)
            else:
                win.set_cursor(None)
        except Exception:
            pass

    # ----- async job control -----
    def _cancel_current(self):
        self._cancel_event.set()
        self._worker = None

    def _start_render(self):
        self._cancel_current()
        self._cancel_event = threading.Event()
        job_id = self._job_id = self._job_id + 1

        self._progress(0.0, "Preparing…")
        self._busy(True)

        def worker():
            try:
                if self._cancel_event.is_set(): return
                self._progress(0.05, "Fusing energy…")
                rad=max(1,int(round(self.k*0.25)))
                Ex = normalize([0.65*s+0.35*a for s,a in zip(self.Ex_s,self.Ex_a)])
                Ey = normalize([0.65*s+0.35*a for s,a in zip(self.Ey_s,self.Ey_a)])
                Ex = smooth_ma(Ex,rad); Ey = smooth_ma(Ey,rad)
                if self._cancel_event.is_set(): return

                self._progress(0.15, "Picking grid…")
                Gx=pick_gridlines_greedy(Ex,self.w,self.k,0.20)
                Gy=pick_gridlines_greedy(Ey,self.h,self.k,0.20)
                if Gx[0]!=0: Gx=[0]+Gx
                if Gx[-1]!=self.w: Gx=Gx+[self.w]
                if Gy[0]!=0: Gy=[0]+Gy
                if Gy[-1]!=self.h: Gy=Gy+[self.h]
                if self._cancel_event.is_set(): return

                workers = max(2, min(4, (os.cpu_count() or 2)))
                colors = analyze_tiles_parallel(self.src,self.w,self.h,Gx,Gy,self.coh,
                                                workers, self._cancel_event, self._progress)
                if self._cancel_event.is_set(): return

                self._progress(0.85, "Edge cleanup…")
                colors = snap_edges(colors, self.cln)
                if self._cancel_event.is_set(): return

                self._progress(0.92, "Painting…")
                dx=[Gx[i+1]-Gx[i] for i in range(len(Gx)-1)]
                dy=[Gy[i+1]-Gy[i] for i in range(len(Gy)-1)]
                kf=max(2,int(round((statistics.median(dx)+statistics.median(dy))/2.0)))
                rb,Wf,Hf = repaint_uniform(colors,kf)  # preview uses uniform true size
                pix = GdkPixbuf.Pixbuf.new_from_data(bytes(rb), GdkPixbuf.Colorspace.RGB, True,8,Wf,Hf,Wf*4)

                if self._cancel_event.is_set(): return
                def deliver():
                    if self._job_id != job_id: return False
                    self.pb = pix
                    self.da.queue_draw()
                    self._progress(1.0, "Done")
                    self._busy(False)
                    return False
                GLib.idle_add(deliver)
            except Exception as e:
                def report():
                    self._busy(False)
                    Gimp.message(f"Preview error: {type(e).__name__}: {e}")
                    return False
                GLib.idle_add(report)

        t = threading.Thread(target=worker, daemon=True)
        self._worker = t
        t.start()

    # ----- preview helpers -----
    def _first_paint(self):
        if self._first_paint_done: return False
        self._first_paint_done=True
        self._start_render()
        return False

    def _toggle(self, attr, val):
        setattr(self,attr,bool(val)); self._debounce()
    def _on_change(self, adj, attr):
        setattr(self,attr,int(adj.get_value())); self._debounce()
    def _debounce(self):
        if self._deb: GLib.source_remove(self._deb)
        self._deb=GLib.timeout_add(80, self._debounced_fire)
    def _debounced_fire(self):
        self._deb=None
        self._start_render()
        return False

    def _on_draw(self, da, cr: cairo.Context):
        if not self.pb: return False
        w = da.get_allocated_width()
        h = da.get_allocated_height()
        if w <= 1 or h <= 1: return False

        pw = self.pb.get_width()
        ph = self.pb.get_height()
        scale = min(w / pw, h / ph)
        sw = pw * scale
        sh = ph * scale
        offx = (w - sw) * 0.5
        offy = (h - sh) * 0.5

        cr.save()
        cr.translate(offx, offy)
        cr.scale(scale, scale)
        Gdk.cairo_set_source_pixbuf(cr, self.pb, 0, 0)
        cr.paint()
        cr.restore()
        return False

    def _on_delete(self, *args):
        self._cancel_current()
        return False

    def params(self):
        return dict(k=self.k, cohesion=self.coh, cleanup=self.cln, finalize=self.finalize, expand=self.expand)

# -------------- main proc -----------------
def run(proc, run_mode, image, drawables, config, data):
    try:
        if len(drawables)!=1 or not isinstance(drawables[0], Gimp.Layer):
            return proc.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, GLib.Error("Select exactly one layer."))
        layer=drawables[0]; w,h=layer.get_width(), layer.get_height()
        Gegl.init(None); GimpUi.init("virtualize-grid")

        # STRAIGHT-ALPHA read
        buf=layer.get_buffer()
        rect=Gegl.Rectangle(); rect.x=0; rect.y=0; rect.width=w; rect.height=h
        raw = buf.get(rect, 1.0, "R'G'B'A u8", Gegl.AbyssPolicy.CLAMP)
        src = memoryview(bytearray(raw))

        # energies kept for consistency; initial k forced to 23 by request
        _k_auto, energies = auto_defaults(src,w,h)
        k0 = 23  # preferred default

        dlg=Preview(src,w,h,k0,energies)
        resp=dlg.run(); p=dlg.params(); dlg.destroy()
        if resp!=Gtk.ResponseType.OK:
            return proc.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

        # Apply
        Ex_s,Ey_s,Ex_a,Ey_a = energies
        rad=max(1,int(round(p["k"]*0.25)))
        Ex=smooth_ma(normalize([0.65*s+0.35*a for s,a in zip(Ex_s,Ex_a)]),rad)
        Ey=smooth_ma(normalize([0.65*s+0.35*a for s,a in zip(Ey_s,Ey_a)]),rad)
        Gx=pick_gridlines_greedy(Ex,w,p["k"],0.20)
        Gy=pick_gridlines_greedy(Ey,h,p["k"],0.20)
        if Gx[0]!=0: Gx=[0]+Gx
        if Gx[-1]!=w: Gx=Gx+[w]
        if Gy[0]!=0: Gy=[0]+Gy
        if Gy[-1]!=h: Gy=Gy+[h]

        colors = analyze_tiles_parallel(src,w,h,Gx,Gy,p["cohesion"],
                                        workers=max(2, min(4, (os.cpu_count() or 2))),
                                        cancel_event=threading.Event(),
                                        progress_cb=None)
        colors = snap_edges(colors, p["cleanup"])

        image.undo_group_start()
        try:
            if p["finalize"]:
                dx=[Gx[i+1]-Gx[i] for i in range(len(Gx)-1)]
                dy=[Gy[i+1]-Gy[i] for i in range(len(Gy)-1)]
                kf=max(2,int(round((statistics.median(dx)+statistics.median(dy))/2.0)))
                if p["expand"]:
                    out,Wf,Hf = repaint_uniform(colors,kf)
                    image.resize(Wf,Hf,0,0); w,h=Wf,Hf
                else:
                    # canvas-sized uniform repaint to avoid blank layer
                    out = repaint_uniform_to_canvas(colors, w, h)
            else:
                out = repaint_irregular(colors,Gx,Gy,w,h)

            pos=image.get_item_position(layer)
            out_layer=Gimp.Layer.new(image, "Virtual Pixels", w,h, Gimp.ImageType.RGBA_IMAGE, 100.0, Gimp.LayerMode.NORMAL)
            image.insert_layer(out_layer, layer.get_parent(), pos)
            obuf=out_layer.get_buffer()
            rect=Gegl.Rectangle(); rect.x=0; rect.y=0; rect.width=w; rect.height=h
            obuf.set(rect, "R'G'B'A u8", bytes(out))
            obuf.flush(); out_layer.update(0,0,w,h)
            layer.set_visible(False)
        finally:
            image.undo_group_end()

        return proc.new_return_values(Gimp.PDBStatusType.SUCCESS, None)
    except Exception as e:
        Gimp.message("Virtualize Grid error:\n"+ "".join(traceback.format_exception_only(type(e),e)))
        return proc.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error(str(e)))

class Plugin(Gimp.PlugIn):
    def do_query_procedures(self): return [PROC_NAME]
    def do_create_procedure(self, name):
        if name!=PROC_NAME: return None
        p=Gimp.ImageProcedure.new(self,name,Gimp.PDBProcType.PLUGIN, run, None)
        p.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        p.set_menu_label("Virtualize Grid…"); p.add_menu_path("<Image>/Filters/Pixel Art")
        p.set_documentation("Rebuild blown-up pixel art on an adaptive lattice with palette-preserving colors.", None, None)
        p.set_attribution("Adam + GPT-5 Thinking","MIT","2025")
        return p

Gimp.main(Plugin.__gtype__, sys.argv)
