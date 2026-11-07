import cv2, numpy as np, subprocess, sys, math
from collections import deque
from skimage.morphology import skeletonize
from objs import load

W,H,FPS = 1920,1080,30
RED  = np.array([38,32,205],np.float32)/255   # BGR marker red
GRAY = np.array([78,76,74],np.float32)/255
OIL  = np.array([18,20,26],np.float32)/255
rng = np.random.default_rng(7)

def ease(t): t=min(max(t,0),1); return t*t*(3-2*t)
def eob(t):  # ease out back
    t=min(max(t,0),1); c=1.7; return 1+(c+1)*(t-1)**3+c*(t-1)**2

def geo_order(mask):
    """draw-on order: geodesic distance per connected component, components left->right"""
    h,w=mask.shape; order=np.full((h,w),-1.0,np.float32)
    nl,lab=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
    comps=[]
    for i in range(1,nl):
        ys,xs=np.nonzero(lab==i); comps.append((xs.min(),i,ys,xs))
    comps.sort(); base=0.0
    for _,i,ys,xs in comps:
        k=np.argmin(xs*1.0+ys*0.3); sy,sx=ys[k],xs[k]
        dist=np.full((h,w),-1,np.int32); dist[sy,sx]=0; q=deque([(sy,sx)]); m=lab==i
        while q:
            y,x=q.popleft(); d=dist[y,x]+1
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    yy,xx=y+dy,x+dx
                    if 0<=yy<h and 0<=xx<w and m[yy,xx] and dist[yy,xx]<0:
                        dist[yy,xx]=d; q.append((yy,xx))
        sel=dist>=0; order[sel]=base+dist[sel]; base+=dist.max()+8
    order[order>=0]/=max(base,1)
    return order

class Sprite:
    def __init__(s, red, dark, bbox, S, off, motion='boil', phase=0.0, t0=0, dur=1):
        x,y,w,h=bbox; p=6; x-=p;y-=p;w+=2*p;h+=2*p
        r=red[y:y+h,x:x+w]; d=dark[y:y+h,x:x+w]; m=r|d
        o=geo_order(m)
        sw,sh=int(round(w*S)),int(round(h*S))
        up=lambda a:cv2.resize(a.astype(np.float32),(sw,sh),interpolation=cv2.INTER_LINEAR)
        ra=cv2.GaussianBlur(up(r),(0,0),0.9); da=cv2.GaussianBlur(up(d),(0,0),0.9)
        a=np.clip(np.maximum(ra,da)*1.25-0.1,0,1)
        col=(RED[None,None]*ra[...,None]+GRAY[None,None]*da[...,None])/np.maximum(ra+da,1e-4)[...,None]
        # subtle marker texture
        tex=1+0.06*cv2.GaussianBlur(rng.standard_normal((sh,sw)).astype(np.float32),(0,0),1.2)
        s.col=np.clip(col*tex[...,None],0,1); s.a=a
        oo=cv2.resize(o,(sw,sh),interpolation=cv2.INTER_NEAREST)
        oo=cv2.dilate(oo,np.ones((3,3),np.uint8))  # fill edges of upscaled mask
        oo[oo<0]=0; s.order=oo
        s.pos=np.array([off[0]+x*S, off[1]+y*S],np.float32)   # canvas top-left
        s.size=(sw,sh); s.motion=motion; s.phase=phase; s.t0=t0; s.dur=dur
        s.anchor=np.array([sw/2,sh*0.92],np.float32)  # bottom-center pivot
        s.cx=s.pos[0]+sw/2
        s.seed=rng.integers(1e6)

    def transform(s,t,tm):
        """returns (dx,dy,rot_deg,sx,sy) ; tm = time since motion start"""
        ph=s.phase; M=s.motion; k=max(0,min(1,tm/0.8))  # fade-in of motion
        dx=dy=rot=0.0; sx=sy=1.0
        if M=='walk':
            w=2*math.pi*1.6*tm+ph
            dy=-abs(math.sin(w))*16*k; rot=math.sin(w)*7*k; dx=tm*14*k
        elif M=='arrow':
            c=((tm*0.9+ph)%1.0); dx=k*(ease(c)*34-6); sx=1+0.08*k*math.sin(c*math.pi)
        elif M=='jump':
            c=(tm*0.75+ph)%1.0
            if c<0.18: sq=math.sin(c/0.18*math.pi); sy=1-0.22*sq*k; sx=1+0.18*sq*k
            elif c<0.7: u=(c-0.18)/0.52; dy=-4*u*(1-u)*120*k; sy=1+0.08*k*math.sin(u*math.pi); sx=2-sy
            elif c<0.82: sq=math.sin((c-0.7)/0.12*math.pi); sy=1-0.15*sq*k; sx=1+0.12*sq*k
        elif M=='wiggle':
            rot=math.sin(2*math.pi*0.9*tm+ph)*6*k; sx=sy=1+0.04*k*math.sin(2*math.pi*1.8*tm+ph)
        elif M=='sway':
            rot=math.sin(2*math.pi*0.55*tm+ph)*3.5*k; dx=math.sin(2*math.pi*0.55*tm+ph+1)*6*k
        elif M=='dance':
            w=2*math.pi*1.1*tm+ph
            rot=math.sin(w)*12*k; dy=-abs(math.sin(w))*18*k; sy=1+0.06*k*math.cos(2*w); sx=2-sy
        elif M=='rock':
            w=2*math.pi*0.45*tm+ph; rot=math.sin(w)*6*k; dy=math.sin(2*w)*6*k
        elif M=='hop':
            c=(tm*1.3+ph)%1.0; dy=-abs(math.sin(c*math.pi))*40*k; rot=math.sin(c*2*math.pi)*8*k
        elif M=='shake':
            rot=math.sin(2*math.pi*2.3*tm+ph)*3*k; sy=1+0.05*k*math.sin(2*math.pi*1.15*tm+ph)
        # hand-drawn "boil": new random pose every 4 frames
        f=int(t*FPS)//4; r=np.random.default_rng(s.seed+f)
        bx,by,br=r.uniform(-1.3,1.3),r.uniform(-1.3,1.3),r.uniform(-0.5,0.5)
        return dx+bx,dy+by,rot+br,sx,sy

    def draw(s,canvas,t,tm,reveal,cam):
        if reveal<=0: return None
        dx,dy,rot,sx,sy=s.transform(t,tm)
        ax,ay=s.anchor
        M=cv2.getRotationMatrix2D((float(ax),float(ay)),rot,1.0)
        Sm=np.array([[sx,0,ax*(1-sx)],[0,sy,ay*(1-sy)],[0,0,1]],np.float32)
        A=np.vstack([M,[0,0,1]])@Sm
        A[0,2]+=s.pos[0]+dx; A[1,2]+=s.pos[1]+dy
        A=cam@A
        a=s.a
        if reveal<1:
            soft=np.clip((reveal-s.order)/0.015+1,0,1); a=a*soft
        A2=A[:2]
        corners=np.array([[0,0,1],[s.size[0],0,1],[0,s.size[1],1],[s.size[0],s.size[1],1]],np.float32)@A2.T
        x0=int(max(0,corners[:,0].min()-2)); x1=int(min(W,corners[:,0].max()+3))
        y0=int(max(0,corners[:,1].min()-2)); y1=int(min(H,corners[:,1].max()+3))
        if x1<=x0 or y1<=y0: return None
        A3=A2.copy(); A3[0,2]-=x0; A3[1,2]-=y0
        rgba=np.dstack([s.col,a])
        wp=cv2.warpAffine(rgba,A3,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR,borderValue=0)
        al=wp[...,3:4]; reg=canvas[y0:y1,x0:x1]
        # multiply-ish marker blend
        reg[:]=reg*(1-al)+(wp[...,:3]*reg)*al*0.15+wp[...,:3]*al*0.85
        if reveal<1:
            band=(np.abs(s.order-reveal)<0.012)&(s.a>0.3)
            if band.any():
                ys,xs=np.nonzero(band); p=np.array([xs.mean(),ys.mean(),1],np.float32)@A2.T
                return p
        return None

def paper():
    base=np.ones((H,W,3),np.float32)*np.array([0.93,0.95,0.965],np.float32)
    n=cv2.GaussianBlur(rng.standard_normal((H,W)).astype(np.float32),(0,0),1.0)*0.012
    n+=cv2.GaussianBlur(rng.standard_normal((H,W)).astype(np.float32),(0,0),25)*0.25
    base+=n[...,None]
    yy,xx=np.mgrid[0:H,0:W]; r=((xx-W/2)/(W*0.75))**2+((yy-H/2)/(H*0.75))**2
    base*=(1-0.22*r)[...,None]
    return np.clip(base,0,1)

def draw_pen(img,p):
    x,y=int(p[0]),int(p[1]); tipx,tipy=x+70,y-110
    cv2.line(img,(x+14,y-22),(tipx,tipy),(40,40,40),24,cv2.LINE_AA)
    cv2.line(img,(x+14,y-22),(tipx,tipy),(60,60,215),18,cv2.LINE_AA)
    cv2.line(img,(x+40,y-63),(tipx,tipy),(235,235,235),18,cv2.LINE_AA)
    cv2.line(img,(x+3,y-5),(x+14,y-22),(40,40,190),10,cv2.LINE_AA)

def build_scene(name, motions, split, layout_box):
    red,dark,objs=load(name)
    x0,y0,x1,y1=layout_box
    S=min((W-220)/(x1-x0),(H-200)/(y1-y0))
    off=(W/2-(x0+x1)/2*S, H/2-(y0+y1)/2*S+20)
    sprites=[]
    for k,o in enumerate(objs):
        mot=motions.get(k,'sway')
        if split.get(k)=='color':
            m=o['mask']; parts=[]
            for src,lbl in ((red,'r'),(dark,'d')):
                pm0=src&m; nl,lab,st,_=cv2.connectedComponentsWithStats(cv2.dilate(pm0.astype(np.uint8),np.ones((5,5),np.uint8)))
                for i in range(1,nl):
                    pm=(lab==i)&pm0
                    if pm.sum()<40: continue
                    parts.append((lbl,int(pm.sum()),pm,st[i,:4]))
            big=max(p[1] for p in parts if p[0]=='r')
            for lbl,ar,pm,bb in parts:
                mm='boil' if (lbl=='d' or ar==big) else 'shake'
                print(name,k,lbl,bb,ar,mm)
                sprites.append(Sprite(red&pm,dark&pm,bb,S,off,mm,phase=bb[0]*0.01))
        elif k in split:
            m=o['mask']; nl,lab,st,_=cv2.connectedComponentsWithStats(cv2.dilate(m.astype(np.uint8),np.ones((9,9),np.uint8)))
            parts=[]
            for i in range(1,nl):
                pm=(lab==i)&m
                if pm.sum()<60: continue
                parts.append((st[i,0],pm,st[i,:4]))
            parts.sort(key=lambda z:z[0])
            for j,(_,pm,bb) in enumerate(parts):
                print(name,k,'part',j,bb,int(pm.sum()))
                mm=split[k][j] if isinstance(split[k],list) and j<len(split[k]) else mot
                sprites.append(Sprite(red&pm,dark&pm,bb,S,off,mm,phase=j*0.37))
        else:
            sprites.append(Sprite(red&o['mask'],dark&o['mask'],o['bbox'],S,off,mot,phase=k*0.9))
    sprites.sort(key=lambda s:s.cx)
    return sprites,S,off,red

# ---------- scenes ----------
print('building...',flush=True)
sc1,S1,off1,red1=build_scene('s1',{0:'walk',1:'arrow',2:'walk',3:'arrow',4:'jump',5:'wiggle',6:'sway',7:'boil'},
                     {0:'walk',2:['walk','walk','walk','hop'],3:'arrow',7:'color'},
                     (430,575,1740,1115))
sc2,S2,off2,_=build_scene('s2',{0:'dance',1:'dance',2:'dance',3:'dance',4:'shake',5:'hop',6:'wiggle',7:'rock'},{},(275,590,1645,1120))
for s in sc2:
    if s.motion=='rock': s.anchor=np.array([s.size[0]/2,s.size[1]*0.75],np.float32)

# oil line path (scene 1): skeleton of the long red line
nl,lab,st,_=cv2.connectedComponentsWithStats(red1.astype(np.uint8))
cand=[(((lab==i)[560:800,1450:1700]).sum(),i) for i in range(1,nl)]
line=lab==max(cand)[1]
pts=[]
for xv in range(1395,1716,3):
    col=np.nonzero(line[:,xv-1:xv+2].any(1))[0]
    col=col[col<930]
    if len(col): pts.append((xv,float(np.median(col))))
pts=np.array(pts,np.float32)
pts[:,1]=np.convolve(np.pad(pts[:,1],3,mode='edge'),np.ones(7)/7,'valid')
pts_c=pts*S1+np.array(off1,np.float32)
seg=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts_c,axis=0),axis=1))]
def along(u):
    d=u*seg[-1]; i=np.searchsorted(seg,d)-1; i=np.clip(i,0,len(seg)-2)
    f=(d-seg[i])/(seg[i+1]-seg[i]+1e-6); return pts_c[i]*(1-f)+pts_c[i+1]*f
_,dark1,_=load('s1')
bm=np.zeros(dark1.shape,np.uint8)
for yv in range(815,960):
    row=np.nonzero(dark1[yv,1590:1735])[0]+1590
    L=row[row<1660]; R=row[row>1665]
    if len(L) and len(R) and R.min()-L.max()>6: bm[yv,L.max():R.min()+1]=255
bm=cv2.dilate(bm,np.ones((5,5),np.uint8))
bucket_mask=cv2.resize(bm,None,fx=S1,fy=S1,interpolation=cv2.INTER_LINEAR)
bucket_off=np.array(off1,np.float32)
ys_,xs_=np.nonzero(bucket_mask); BTOP,BBOT=ys_.min(),ys_.max()

PAPER=paper()
# timeline (seconds)
T_D1=(0.6,7.0); T_M1=7.0; T_FLOW=8.0; T_END1=17.0; T_TR=1.0
T_D2=(T_END1+T_TR+0.3, T_END1+T_TR+5.8); T_M2=T_D2[1]; T_END2=T_M2+8.5; T_OUT=1.2
TOTAL=T_END2+T_OUT
NF=int(TOTAL*FPS)

def schedule(sprites,t0,t1):
    n=len(sprites); span=t1-t0
    for i,s in enumerate(sprites):
        st=t0+span*i/n*0.92; s.t0=st; s.dur=max(0.5,span/n*1.25)

schedule(sc1,*T_D1); schedule(sc2,*T_D2)

def render_scene(sprites,t,tmot,cam,extra=None):
    img=PAPER.copy() if cam is None else cv2.warpAffine(PAPER,cam[:2],(W,H),borderMode=cv2.BORDER_REFLECT)
    camM=np.eye(3,dtype=np.float32) if cam is None else cam
    if extra: extra(img,t,camM,'under')
    pen=None
    for s in sprites:
        rv=(t-s.t0)/s.dur
        p=s.draw(img,t,max(0,t-tmot),min(rv,1.0) if rv<1 else 1.0,camM)
        if p is not None: pen=p
    if extra: extra(img,t,camM,'over')
    out=(np.clip(img,0,1)*255).astype(np.uint8)
    if pen is not None: draw_pen(out,pen)
    return out

def oil_fx(img,t,cam,layer):
    tf=t-T_FLOW
    if tf<0: return
    if layer=='under':
        lvl=ease(tf/7.5)
        if lvl>0:
            m=np.zeros((H,W),np.uint8)
            Mb=cam[:2].copy(); Mb[:,2]+=cam[:2,:2]@bucket_off
            m=cv2.warpAffine(bucket_mask,Mb,(W,H))
            ys_,xs_=np.nonzero(m)
            if len(ys_)==0: return
            ytop,ybot=ys_.min(),ys_.max(); yl=ybot-(ybot-ytop)*lvl*0.85
            xs=np.arange(W); wave=yl+3*np.sin(xs*0.08+t*5)
            yy=np.arange(H)[:,None]; m[(yy<wave[None,:])]=0
            bc=np.c_[xs_,ys_]
            a=cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),1.2)[...,None]
            img[:]=img*(1-a)+OIL*a
            # glossy highlight
            hl=np.zeros((H,W),np.float32); x0=int(bc[:,0].min()+10); x1=int(bc[:,0].max()-14)
            cv2.line(hl,(x0,int(yl+6)),(x1,int(yl+5)),1.0,3,cv2.LINE_AA)
            hl=cv2.GaussianBlur(hl,(0,0),1.5)*a[...,0]*0.5; img[:]+=hl[...,None]
    else:
        # drops flowing along line
        for j in range(6):
            u=(tf*0.45+j/6)%1.0
            if tf*0.45+j/6<1: continue  # stagger start
            p=np.r_[along(u),1]@cam[:2].T
            r=11*(0.8+0.3*math.sin(u*math.pi)); a_=min(1,(1-u)*8)
            ov=img.copy()
            cv2.circle(ov,(int(p[0]),int(p[1])),int(r),OIL.tolist(),-1,cv2.LINE_AA)
            cv2.circle(ov,(int(p[0]-r*0.35),int(p[1]-r*0.35)),max(2,int(r*0.3)),(0.6,0.6,0.65),-1,cv2.LINE_AA)
            img[:]=img*(1-a_)+ov*a_

def zoom(t,ta,tb,z0,z1,cx=W/2,cy=H/2):
    z=z0+(z1-z0)*ease((t-ta)/(tb-ta))
    return np.array([[z,0,cx*(1-z)],[0,z,cy*(1-z)],[0,0,1]],np.float32)

def title(img,text,alpha,y=90,size=1.6):
    if alpha<=0: return
    ov=img.copy()
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,size,3)
    cv2.putText(ov,text,((W-tw)//2,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,size,(60,50,170),3,cv2.LINE_AA)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

only=sys.argv[2:]
ff=None if only else subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
    '-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',sys.argv[1] if len(sys.argv)>1 else 'out.mp4'],stdin=subprocess.PIPE)
only=sys.argv[2:]  # optional frame list for previews
frames=[int(x) for x in only] if only else range(NF)
for fi in frames:
    t=fi/FPS
    if t<T_END1:
        cam=zoom(t,0,T_END1,1.0,1.08)
        fr=render_scene(sc1,t,T_M1,cam,oil_fx)
        fr=(fr.astype(np.float32)*min(1,t/0.6)+255*PAPER.mean()*(1-min(1,t/0.6))).astype(np.uint8)
    elif t<T_END1+T_TR:
        u=ease((t-T_END1)/T_TR)
        a=render_scene(sc1,t,T_M1,zoom(T_END1,0,T_END1,1.0,1.08),oil_fx)
        b=render_scene(sc2,t,T_M2,None)
        sh=int(u*W)
        fr=np.empty_like(a); fr[:,:W-sh]=a[:,sh:]; fr[:,W-sh:]=b[:,:sh]
        if 0<sh<W:  # page edge shadow
            x=W-sh; lo=max(0,x-30); fr[:,lo:x]=(fr[:,lo:x]*np.linspace(1,0.75,x-lo)[None,:,None]).astype(np.uint8)
    else:
        cam=zoom(t,T_END1+T_TR,T_END2,1.0,1.07)
        fr=render_scene(sc2,t,T_M2,cam)
        if t>T_END2:
            u=ease((t-T_END2)/T_OUT); fr=(fr*(1-u)+np.array([235,240,243])*u).astype(np.uint8)
    if only: cv2.imwrite(f'prev_{fi:04d}.jpg',fr); continue
    ff.stdin.write(fr.tobytes())
    if fi%60==0: print(fi,'/',NF,flush=True)
if ff: ff.stdin.close(); ff.wait()
print('done',NF/FPS,'s')
