"""Regenerated task plates from coursework concepts; presentation added for portfolio."""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .generate import read, resource, show, save, ACCENT
from . import processing, filters, edges, features, geometry, pca
from .course_utils.a5_utils import read_matches

def grid(name,panels,cols=3):
    fig,axs=plt.subplots((len(panels)+cols-1)//cols,cols,figsize=(12,3.6*((len(panels)+cols-1)//cols)),squeeze=False)
    for ax,(img,title) in zip(axs.flat,panels):show(ax,img,title)
    for ax in list(axs.flat)[len(panels):]:ax.axis('off')
    fig.tight_layout(pad=2);save(fig,name)

def assignment1():
    im=read(1,'images/umbrellas.jpg')/255.;gray=im.mean(2);inv=im.copy();inv[130:260,240:450]=1-inv[130:260,240:450]
    fig,axs=plt.subplots(2,3,figsize=(12,8))
    for ax,x,t in zip(axs.flat,[im,gray,im[130:260,240:450,1],inv,gray,gray*.3],['RGB image','Grayscale · channel mean','Green-channel crop','Invert a rectangular region','Original intensity range','Reduce intensity to [0, 0.3]']):show(ax,x,t,vmin=0,vmax=1)
    fig.tight_layout(pad=2);save(fig,'01-pixels.png')
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for j,(x,title) in enumerate([(gray*.5,'Darkened ×0.5 (derived)'),(gray,'Original'),(np.clip(gray*1.5,0,1),'Brightened ×1.5 (derived)')]):
        show(axs[0,j],x,title,vmin=0,vmax=1)
        for bins in [20,100]:axs[1,j].plot(np.linspace(0,1,bins),processing.myhist(x,bins),label=f'{bins} bins')
        axs[1,j].set(xlabel='Intensity',ylabel='Normalized frequency');axs[1,j].legend()
    fig.tight_layout(pad=2);save(fig,'01-histograms.png')
    bird=read(1,'images/bird.jpg',gray=True)/255.;mask=(bird>.3).astype(np.uint8);kernel=np.ones((5,5),np.uint8)
    coins=read(1,'images/coins.jpg');cm=(cv2.cvtColor(coins,cv2.COLOR_RGB2GRAY)/255.<.9).astype(np.uint8);count,lab,stats,_=cv2.connectedComponentsWithStats(cm);small=np.isin(lab,[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]<=700]);result=coins.copy();result[~small]=255
    grid('01-morphology.png',[(mask,'Threshold mask · 0.3'),(cv2.erode(mask,kernel),'Erosion · 5 × 5'),(cv2.dilate(mask,kernel),'Dilation · 5 × 5'),(coins,'Coin image'),(cm,'Dark foreground · threshold 0.9'),(result,'Retain components ≤ 700 px')])

def assignment2():
    signal=np.array([0,1,1,1,0,.7,.5,.2,0,0,1,0]);kernel=np.array([.5,1,.3]);fig,ax=plt.subplots(1,3,figsize=(13,4))
    ax[0].plot(signal,'o-',label='Signal');ax[0].plot(filters.simple_convolution_fix(signal,kernel),'o-',label='Convolution');ax[0].legend();ax[0].set_title('Padded 1D convolution')
    for sigma in [.5,1,2,3,4]:
        g=filters.gauss(sigma);ax[1].plot(np.arange(len(g))-len(g)//2,g,label=f'σ={sigma}')
    ax[1].legend();ax[1].set_title('Normalized Gaussian kernels');g=filters.gauss(2);k=np.array([.1,.6,.4]);a=np.convolve(np.convolve(signal,g),k);b=np.convolve(signal,np.convolve(g,k));ax[2].plot(a,label='(signal ∗ g) ∗ k');ax[2].plot(b,'--',label='signal ∗ (g ∗ k)');ax[2].legend();ax[2].set_title('Associativity · full convolution');fig.tight_layout(pad=2);save(fig,'02-convolution.png')
    fox=read(2,'images/fox.jpg',gray=True)/255.;grid('02-sharpening.png',[(fox,'Original'),(filters.gaussfilter(fox,1),'Gaussian smoothing'),(np.clip(2*fox-cv2.filter2D(fox,-1,np.ones((3,3))/9),0,1),'Unsharp filter · 2I − box(I)')])
    files=sorted(resource(2,'dataset').glob('*.png'));H=np.array([filters.myhist3(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)/255.,8).ravel() for p in files]);q=next(i for i,p in enumerate(files) if p.name=='object_05_4.png');weighted=H*np.exp(-5*H.sum(0)/H.sum());weighted/=weighted.sum(1,keepdims=True);fig,axs=plt.subplots(2,2,figsize=(12,7))
    for ax,method in zip(axs.flat,['L2','Chi2','Intersection','Hellinger']):
        for values,label in [(H,'Original histograms'),(weighted,'Frequency-weighted')]:ax.plot(sorted(filters.compare_histograms(values[q],v,method) for v in values),label=label)
        ax.set(title=method,xlabel='Images sorted by distance',ylabel='Distance');ax.legend()
    fig.tight_layout(pad=2);save(fig,'02-distances.png')

def assignment3():
    im=read(3,'images/museum.jpg',gray=True,max_side=550)/255.;ix,iy=edges.compute_derivatives(im,2);xx,yy,xy=edges.compute_second_derivatives(im,2);mag,direction=edges.gradient_magnitude(im,2)
    grid('03-derivatives.png',[(im,'Museum · input'),(ix,'First derivative Ix'),(iy,'First derivative Iy'),(xx,'Second derivative Ixx'),(xy,'Mixed derivative Ixy'),(mag,'Gradient magnitude')]);thin=edges.non_max_suppression(mag,direction);grid('03-thinning.png',[(mag>.1,'Threshold magnitude > 0.1'),(thin>.1,'Directional non-max suppression'),(edges.hysteresis(thin,.04,.1),'Hysteresis · low 0.04 / high 0.1')])
    eclipse=read(3,'images/eclipse.jpg');edge=cv2.Canny(cv2.cvtColor(eclipse,cv2.COLOR_RGB2GRAY),50,150);acc=np.zeros(edge.shape,np.int32);ys,xs=np.nonzero(edge);theta=np.deg2rad(np.arange(360));radius=48
    for x,y in zip(xs,ys):
        xx=np.rint(x-radius*np.cos(theta)).astype(int);yy=np.rint(y-radius*np.sin(theta)).astype(int);ok=(xx>=0)&(yy>=0)&(xx<edge.shape[1])&(yy<edge.shape[0]);np.add.at(acc,(yy[ok],xx[ok]),1)
    y,x=np.unravel_index(acc.argmax(),acc.shape);fig,ax=plt.subplots(1,3,figsize=(12,4));show(ax[0],edge,'Canny edges');show(ax[1],acc,'Circle-center votes');show(ax[2],eclipse,'Strongest center · fixed radius 48 px');ax[2].add_patch(plt.Circle((x,y),radius,fill=False,color='#ff7048',lw=2));fig.tight_layout(pad=2);save(fig,'03-circles.png')

def assignment4():
    a=read(4,'data/newyork/newyork_a.jpg');b=read(4,'data/newyork/newyork_b.jpg');ga=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)/255.;gb=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)/255.;fig,axs=plt.subplots(2,3,figsize=(12,7))
    for row,(name,detector) in enumerate([('Hessian',features.hessian_points),('Harris',features.harris_points)]):
        for col,sigma in enumerate([3,6,9]):
            response,x,y=detector(ga,sigma,.4);show(axs[row,col],a,f'{name} · σ={sigma} · {len(x)} points');axs[row,col].scatter(x,y,s=12,facecolors='none',edgecolors='#ff663e',linewidths=.8)
    fig.tight_layout(pad=2);save(fig,'04-detectors.png')
    _,xa,ya=features.harris_points(ga,1,.4);_,xb,yb=features.harris_points(gb,1,.4);pa=np.c_[xa,ya];pb=np.c_[xb,yb];da=features.simple_descriptors(ga,ya,xa,n_bins=16,window_size=20,sigma=1);db=features.simple_descriptors(gb,yb,xb,n_bins=16,window_size=20,sigma=1);one=features.find_correspondences(da,db);mutual,pa,pb=features.find_matches(ga,gb,1,.4);fig,axs=plt.subplots(2,1,figsize=(12,8))
    for ax,m,title in [(axs[0],one,'Nearest descriptor · one-way'),(axs[1],mutual,'Mutual nearest neighbors')]:
        show(ax,np.concatenate([a,b],1),f'{title} ({len(m)} matches)')
        for i,j in m:
            p=pa[i];q=pb[j]+[a.shape[1],0];ax.plot([p[0],q[0]],[p[1],q[1]],lw=.7,color=ACCENT,alpha=.7)
    fig.tight_layout(pad=2);save(fig,'04-matches.png')

def assignment5():
    left=read(5,'data/disparity/office_left.png',gray=True,max_side=120)/255.;right=read(5,'data/disparity/office_right.png',gray=True,max_side=120)/255.;disp=geometry.compute_disparity_ncc(left,right,patch_size=6,max_disp=24);fig,ax=plt.subplots(1,3,figsize=(12,4));show(ax[0],left,'Left image · resized to 120 px');show(ax[1],right,'Right image');show(ax[2],disp,'NCC disparity · 6 × 6 patches',cmap='viridis');fig.tight_layout(pad=2);save(fig,'05-disparity.png')
    matches=read_matches(resource(5,'data/house/2D/house.nview-corners'),7,8);a=np.loadtxt(resource(5,'data/house/2D/house.007.corners'))[matches[:,0]];b=np.loadtxt(resource(5,'data/house/2D/house.008.corners'))[matches[:,1]];F=geometry.fundamental_matrix(a,b);fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,img,points,lines in [(axs[0],read(5,'data/house/images/house.007.png'),a,(F.T@np.c_[b,np.ones(len(b))].T).T),(axs[1],read(5,'data/house/images/house.008.png'),b,(F@np.c_[a,np.ones(len(a))].T).T)]:
        show(ax,img,'Points and corresponding epipolar lines');h,w=img.shape[:2]
        for i in np.linspace(0,len(points)-1,8,dtype=int):
            aa,bb,cc=lines[i];color=plt.cm.tab10(i%10);ax.scatter(*points[i],c=[color],s=20)
            if abs(bb)>1e-10:ax.plot([0,w],[-cc/bb,-(cc+aa*w)/bb],color=color,lw=.8)
        ax.set(xlim=(0,w),ylim=(h,0))
    fig.tight_layout(pad=2);save(fig,'05-epipolar.png')

def assignment6():
    X=np.loadtxt(resource(6,'data/points.txt'));mean=X.mean(0);D=X-mean;values,U=np.linalg.eigh(np.cov(D.T));U=U[:,::-1];values=values[::-1];projected=(D@U[:,:1])@U[:,:1].T+mean;high=np.loadtxt(resource(6,'data/points_50D.txt'));s=np.linalg.eigvalsh(np.cov(high.T))[::-1];cum=np.cumsum(s)/s.sum();k=np.searchsorted(cum,.8)+1;fig,axs=plt.subplots(1,3,figsize=(13,4))
    axs[0].scatter(*X.T,label='Points');axs[0].scatter(*mean,label='Mean')
    for i in range(2):axs[0].quiver(*mean,*(U[:,i]*np.sqrt(values[i])),angles='xy',scale_units='xy',scale=1,color=ACCENT)
    axs[0].set_title('Covariance & principal directions');axs[0].axis('equal');axs[1].scatter(*X.T,label='Original');axs[1].scatter(*projected.T,label='Rank-1 reconstruction')
    for a,b in zip(X,projected):axs[1].plot([a[0],b[0]],[a[1],b[1]],color='gray',lw=.5)
    axs[1].legend();axs[1].set_title('Project onto the first component');axs[1].axis('equal');axs[2].plot(np.arange(1,len(s)+1),cum*100);axs[2].axhline(80,ls='--',color=ACCENT);axs[2].set(title=f'50D data · {k} components retain 80%',xlabel='Components',ylabel='Variance retained (%)');fig.tight_layout(pad=2);save(fig,'06-point-pca.png')
    dual,sv,mu=pca.dual_pca(X.T);recon=(dual@(dual.T@(X.T-mu))+mu).T;fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].bar(np.arange(len(values))-.15,values,width=.3,label='Direct covariance');ax[0].bar(np.arange(len(sv))+.15,sv,width=.3,label='Dual covariance');ax[0].legend();ax[0].set(title='Nonzero eigenvalues agree',xlabel='Component');ax[1].scatter(*X.T,s=70,label='Original');ax[1].scatter(*recon.T,s=80,facecolors='none',edgecolors=ACCENT,label='Dual PCA reconstruction');ax[1].legend();ax[1].set_title(f'Full reconstruction · max error {np.max(abs(recon-X)):.1e}');fig.tight_layout(pad=2);save(fig,'06-dual-pca.png')

def generate_walkthrough():
    for n,fn in enumerate([assignment1,assignment2,assignment3,assignment4,assignment5,assignment6],1):print(f'Generating assignment {n} task plates',flush=True);fn()

if __name__=='__main__':generate_walkthrough()
