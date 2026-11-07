import cv2, numpy as np
for n in ['s1','s2']:
    im = cv2.imread({'s1':'sheet1.webp','s2':'sheet2.webp'}[n]); im = cv2.resize(im,(2000,1500),interpolation=cv2.INTER_AREA)
    f = im.astype(np.float32); b,g,r = f[...,0],f[...,1],f[...,2]
    redness = r-np.maximum(g,b)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bg = cv2.dilate(gray, np.ones((25,25),np.uint8)); bg = cv2.GaussianBlur(bg,(0,0),15).astype(np.float32)
    ratio = gray/(bg+1)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    paper = ((hsv[...,2]>95)&(hsv[...,1]<55)).astype(np.uint8)
    nl,lab,st,_ = cv2.connectedComponentsWithStats(paper)
    big = 1+np.argmax(st[1:,4]); paper=(lab==big).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((61,61),np.uint8))
    cs,_=cv2.findContours(paper,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    paper=np.zeros_like(paper); cv2.drawContours(paper,cs,-1,1,-1)
    paperD = cv2.erode(paper, np.ones((90,90),np.uint8))
    paper = cv2.erode(paper, np.ones((9,9),np.uint8))
    red = ((redness>20)&(paper>0)).astype(np.uint8)*255
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    dark = ((ratio<0.70)&(redness<12)&(paper>0)).astype(np.uint8)*255
    dark[cv2.dilate(red,np.ones((5,5),np.uint8))>0]=0
    for m in (red,dark):
        nl,lab,st,_=cv2.connectedComponentsWithStats(m)
        for i in range(1,nl):
            if st[i,4]<25: m[lab==i]=0
    cv2.imwrite(n+'_red.png', red); cv2.imwrite(n+'_dark.png', dark); cv2.imwrite(n+'_small.png', im)
    vis = np.full_like(im,255); vis[dark>0]=(80,80,80); vis[red>0]=(30,30,220)
    cv2.imwrite(n+'_vis.jpg', vis)
