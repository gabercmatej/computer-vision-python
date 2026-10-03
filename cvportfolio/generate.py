"""Generate every portfolio figure and measured result: python -m cvportfolio.generate."""
from pathlib import Path
import argparse
import json
import platform
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import processing, filters, edges, features, geometry, pca
from .course_utils.a5_utils import read_matches

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'coursework'
OUT = ROOT / 'docs/assets'
INK, MUTED, BG, ACCENT = '#172821', '#62736b', '#f4f6f1', '#287c58'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,
    'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
    'axes.edgecolor':'#d7ded6','axes.titleweight':'medium','axes.titlesize':12,
    'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG})


def resource(n, relative):
    base = DATA / f'assignment{n}'
    if n != 1: base /= f'assignment{n}'
    p = base / relative
    if not p.exists(): raise FileNotFoundError(f'Missing course material: {p.relative_to(ROOT)}')
    return p


def read(n, relative, gray=False, max_side=None):
    img = cv2.imread(str(resource(n, relative)))
    if img is None: raise ValueError(f'Cannot decode {relative}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB)
    if max_side and max(img.shape[:2]) > max_side:
        factor = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return img


def save_image(name, array):
    if array.dtype != np.uint8: array = np.rint(np.clip(array,0,1)*255).astype(np.uint8)
    if array.ndim == 3: array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(OUT/name),array): raise IOError(name)


def show(ax, array, title, cmap='gray', **kwargs):
    ax.imshow(array, cmap=cmap, **kwargs); ax.set_title(title,loc='left',pad=12)
    ax.axis('off')


def save(fig, name):
    fig.savefig(OUT/name,dpi=170,bbox_inches='tight',pad_inches=.3)
    plt.close(fig)


def processing_demo():
    image = read(1,'images/bird.jpg'); gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)/255.
    threshold = processing.otsu_threshold(gray)
    mask = (gray > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    masked = processing.immask(image,mask)
    fig, ax=plt.subplots(2,2,figsize=(12,8))
    show(ax[0,0],image,'Source image'); show(ax[0,1],mask,f'Otsu mask · threshold {threshold:.3f}')
    show(ax[1,0],masked,'Foreground after morphological closing')
    hist=processing.myhist(gray,64)
    ax[1,1].bar(np.linspace(0,1,64),hist,width=1/64,color=ACCENT)
    ax[1,1].axvline(threshold,color=INK,ls='--',lw=1)
    ax[1,1].set(title='Normalized intensity histogram',xlabel='Intensity',ylabel='Pixel proportion')
    ax[1,1].spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=2); save(fig,'01-processing.png')
    save_image('bird-source.jpg',image); save_image('bird-mask.png',masked)
    return {'threshold':round(threshold,4),'foreground_fraction':round(float(mask.mean()),4)}


def filtering_demo():
    image=read(2,'images/fox.jpg',gray=True,max_side=500)/255.
    rng=np.random.default_rng(42); noisy=image.copy(); noise=rng.random(image.shape)
    noisy[noise<.04]=0; noisy[noise>.96]=1
    median=filters.medianfilter(noisy,3); gaussian=filters.gaussfilter(noisy,1)
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for a,x,title in zip(ax.flat,[image,noisy,gaussian,median],['Source','Salt & pepper noise · 8% probability','Gaussian filter · σ = 1','Median filter · 3 × 3']): show(a,x,title,vmin=0,vmax=1)
    fig.tight_layout(pad=2);save(fig,'02-filtering.png')
    files=sorted(resource(2,'dataset').glob('*.png')); images=[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in files]
    hist=np.array([filters.myhist3(x/255.,8).ravel() for x in images]); q=next(i for i,p in enumerate(files) if p.name=='object_05_4.png')
    distances=np.array([filters.compare_histograms(hist[q],h,'Hellinger') for h in hist]); distances[q]=np.inf
    nearest=np.argsort(distances)[:5]
    fig,ax=plt.subplots(1,6,figsize=(14,3))
    show(ax[0],images[q],'Query image')
    for a,i in zip(ax[1:],nearest):show(a,images[i],f'Distance {distances[i]:.3f}')
    fig.tight_layout(pad=1);save(fig,'02-retrieval.png')
    return {'noisy_mse':float(np.mean((noisy-image)**2)),'median_mse':float(np.mean((median-image)**2)), 'gaussian_mse':float(np.mean((gaussian-image)**2)), 'retrieval_images':len(files),'query':files[q].name,'nearest':[files[i].name for i in nearest]}


def edge_demo():
    image=read(3,'images/building.jpg',max_side=650); gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)/255.
    magnitude,direction=edges.gradient_magnitude(gray,1.2)
    thin=edges.non_max_suppression(magnitude,direction)
    counts=[]
    for high in [.05,.1,.15,.2]:
        binary=edges.hysteresis(thin,high*.4,high)
        save_image(f'edges-{high:.2f}.png',binary.astype(float));counts.append(int(binary.sum()))
    binary=edges.hysteresis(thin,.04,.1)
    lines,acc,rhos,thetas=edges.hough_find_lines(binary,bins_rho=360,bins_theta=360,n_lines=12,nms_size=15)
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    show(ax[0,0],image,'Source image');show(ax[0,1],binary,'Edges · non-maximum suppression + hysteresis')
    show(ax[1,0],acc,'Hough accumulator',cmap='Greens',aspect='auto')
    show(ax[1,1],image,'12 strongest line hypotheses')
    h,w=gray.shape
    for rho,theta in lines:
        c,s=np.cos(theta),np.sin(theta)
        if abs(s)>.1: ax[1,1].plot([0,w],[rho/s,(rho-w*c)/s],color='#d85032',lw=1.2,alpha=.8)
        else: ax[1,1].plot([rho/c,(rho-h*s)/c],[0,h],color='#d85032',lw=1.2,alpha=.8)
    ax[1,1].set(xlim=(0,w),ylim=(h,0));fig.tight_layout(pad=2);save(fig,'03-edges.png')
    save_image('edges-source.jpg',image)
    return {'thresholds':[.05,.1,.15,.2],'edge_pixels':counts,'line_hypotheses':len(lines),'sigma':1.2}


def feature_demo():
    a=read(4,'data/newyork/newyork_a.jpg');b=read(4,'data/newyork/newyork_b.jpg')
    ga=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)/255.;gb=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)/255.
    matches,points1,points2=features.find_matches(ga,gb,sigma=1,thresh=.4)
    H,inliers=features.ransac_homography(points1,points2,matches,threshold=2,k=1000,seed=42)
    errors=features.reprojection_error(H,points1[inliers[:,0]],points2[inliers[:,1]])
    warped=cv2.warpPerspective(a,H,(b.shape[1],b.shape[0]))
    mask=cv2.warpPerspective(np.ones(ga.shape,dtype=np.uint8),H,(b.shape[1],b.shape[0]))>0
    overlay=b.copy();overlay[mask]=np.rint(.5*b[mask]+.5*warped[mask]).astype(np.uint8)
    fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(2,2)
    ax=fig.add_subplot(gs[0,:]);canvas=np.concatenate([a,b],axis=1);show(ax,canvas,'Harris features → symmetric matches → RANSAC inliers')
    rng=np.random.default_rng(3)
    for i,j in inliers:
        color=plt.cm.viridis(rng.uniform(.2,.9));p=points1[i];q=points2[j]+[a.shape[1],0]
        ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=.8,alpha=.85);ax.scatter([p[0],q[0]],[p[1],q[1]],s=8,color=color)
    show(fig.add_subplot(gs[1,0]),warped,'Source warped into the target frame')
    show(fig.add_subplot(gs[1,1]),overlay,'Alignment overlay · 50% blend')
    fig.tight_layout(pad=2);save(fig,'04-homography.png')
    save_image('alignment-target.jpg',b);save_image('alignment-warped.jpg',warped)
    return {'candidate_matches':len(matches),'inliers':len(inliers),'median_reprojection_px':float(np.median(errors)),'threshold_px':2,'iterations':1000,'H':H.tolist()}


def stereo_demo():
    p1=np.loadtxt(resource(5,'data/house/2D/house.007.corners'));p2=np.loadtxt(resource(5,'data/house/2D/house.008.corners'))
    matches=read_matches(resource(5,'data/house/2D/house.nview-corners'),7,8)
    a,b=p1[matches[:,0]],p2[matches[:,1]]
    F=geometry.fundamental_matrix(a,b)
    P1=np.loadtxt(resource(5,'data/house/3D/house.007.P'));P2=np.loadtxt(resource(5,'data/house/3D/house.008.P'))
    X=geometry.triangulate(a,b,P1,P2);T=np.array([[-1,0,0],[0,0,-1],[0,1,0]]);display=X@T.T
    image1=read(5,'data/house/images/house.007.png');image2=read(5,'data/house/images/house.008.png')
    fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(2,2)
    colors=plt.cm.viridis(np.linspace(.1,.95,len(a)))
    for cell,img,p in [(gs[0,0],image1,a),(gs[0,1],image2,b)]:
        ax=fig.add_subplot(cell);show(ax,img,'Supplied point correspondences');ax.scatter(p[:,0],p[:,1],s=12,c=colors)
    ax=fig.add_subplot(gs[1,:],projection='3d');ax.scatter(*display.T,c=colors,s=15,depthshade=False)
    ax.set_title('3D structure recovered by linear triangulation',loc='left');ax.view_init(elev=18,azim=-62);ax.set_box_aspect(np.ptp(display,axis=0));ax.set_axis_off()
    fig.tight_layout(pad=2);save(fig,'05-stereo.png')
    # A separate rotating scientific view, not a simulated reconstruction.
    fig=plt.figure(figsize=(6,4));ax=fig.add_subplot(projection='3d')
    ax.scatter(*display.T,c=colors,s=22);ax.set_box_aspect(np.ptp(display,axis=0));ax.set_axis_off()
    frames=[]
    from PIL import Image
    for angle in range(0,360,12):
        ax.view_init(elev=18,azim=angle);fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
    frames[0].save(OUT/'house-rotation.gif',save_all=True,append_images=frames[1:],duration=110,loop=0)
    plt.close(fig)
    q1=(P1@np.c_[X,np.ones(len(X))].T).T;q2=(P2@np.c_[X,np.ones(len(X))].T).T
    error1=np.linalg.norm(q1[:,:2]/q1[:,2:]-a,axis=1);error2=np.linalg.norm(q2[:,:2]/q2[:,2:]-b,axis=1)
    return {'points':len(X),'mean_symmetric_epipolar_px':float(geometry.reprojection_errors(F,a,b).mean()),'mean_triangulation_reprojection_px':float(np.r_[error1,error2].mean()),'F':F.tolist()}


def pca_demo():
    files=sorted(resource(6,'data/faces/1').glob('*.png'))
    if not files: files=sorted(resource(6,'data/faces/1').glob('*.jpg'))
    images=[cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) for p in files];shape=images[0].shape
    X=np.array([i.ravel() for i in images],dtype=float).T
    U,S,mean=pca.dual_pca(X);x=X[:,0:1];coeff=U.T@(x-mean)
    components=[1,2,4,8,16,32];mse=[]
    fig,ax=plt.subplots(2,4,figsize=(12,8))
    show(ax[0,0],x.reshape(shape),'Original training image',vmin=0,vmax=255)
    save_image('face-original.png',images[0])
    for a,k in zip(list(ax.flat)[1:7],components):
        reconstructed=(U[:,:k]@coeff[:k]+mean).reshape(shape)
        mse.append(float(np.mean((reconstructed-x.reshape(shape))**2)))
        show(a,reconstructed,f'{k} components',vmin=0,vmax=255)
        save_image(f'face-{k}.png',np.clip(reconstructed/255.,0,1))
    cumulative=np.cumsum(S)/S.sum()
    ax[1,3].plot(np.arange(1,len(S)+1),cumulative*100,color=ACCENT,lw=2)
    ax[1,3].set(xlabel='Components',ylabel='Variance retained (%)',ylim=(0,102),title='Explained variance')
    ax[1,3].spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=2);save(fig,'06-pca.png')
    fig,ax=plt.subplots(1,6,figsize=(12,3))
    for j,a in enumerate(ax):show(a,U[:,j].reshape(shape),f'Eigenface {j+1}')
    fig.tight_layout();save(fig,'06-eigenfaces.png')
    return {'training_images':len(files),'image_shape':list(shape),'nonzero_components':len(S),'components':components,'reconstruction_mse':mse,'variance_retained':[float(cumulative[min(k,len(S))-1]) for k in components],'evaluation':'Reconstruction of a training image, not held-out recognition accuracy.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=['1','2','3','4','5','6'],help='Generate one experiment (full run creates the manifest).')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    demos=[processing_demo,filtering_demo,edge_demo,feature_demo,stereo_demo,pca_demo]
    results={}
    for i,demo in enumerate(demos,1):
        if args.only and args.only!=str(i):continue
        print(f'Generating experiment {i}: {demo.__name__}',flush=True)
        start=time.perf_counter();results[str(i)]=demo()
        print(f'  completed in {time.perf_counter()-start:.1f}s',flush=True)
    if not args.only:
        payload={'seed':42,'versions':{'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,'matplotlib':matplotlib.__version__},'experiments':results}
        (OUT/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
        (OUT/'metrics.js').write_text('window.PORTFOLIO_METRICS = '+json.dumps(payload,indent=2)+';\n')
        print('All six experiments exported.',flush=True)

if __name__=='__main__':main()
