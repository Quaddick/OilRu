import cv2, numpy as np
BOX={'s1':(435,560,1800,1120),'s2':(250,580,1700,1120)}
def load(n):
    red=cv2.imread(n+'_red.png',0)>0; dark=cv2.imread(n+'_dark.png',0)>0
    x0,y0,x1,y1=BOX[n]; keep=np.zeros_like(red); keep[y0:y1,x0:x1]=True
    red&=keep; dark&=keep
    # dark: drop huge blobs
    nl,lab,st,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    for i in range(1,nl):
        if st[i,4]>8000 or st[i,4]<60: dark[lab==i]=False
    ink=(red|dark).astype(np.uint8)
    grp=cv2.dilate(ink,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
    nl,lab,st,cen=cv2.connectedComponentsWithStats(grp)
    objs=[]
    for i in range(1,nl):
        m=(lab==i)&(ink>0)
        if m.sum()<150: continue
        x,y,w,h=st[i,:4]; objs.append(dict(bbox=(x,y,w,h),mask=m,cx=x+w/2,cy=y+h/2,area=int(m.sum())))
    objs.sort(key=lambda o:o['cx'])
    return red,dark,objs
if __name__=='__main__':
    for n in ['s1','s2']:
        red,dark,objs=load(n)
        vis=np.full(red.shape+(3,),255,np.uint8); vis[dark]=(80,80,80); vis[red]=(40,40,220)
        for k,o in enumerate(objs):
            x,y,w,h=o['bbox']; cv2.rectangle(vis,(x,y),(x+w,y+h),(0,160,0),2)
            cv2.putText(vis,str(k),(x,y-4),0,1.0,(200,0,0),2)
            print(n,k,o['bbox'],o['area'])
        x0,y0,x1,y1=BOX[n]; cv2.imwrite(n+'_objs.jpg',vis[y0:y1,x0:x1])
