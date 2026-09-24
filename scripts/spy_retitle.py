import numpy as np, sys
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
src, dst = sys.argv[1], sys.argv[2]
im = np.array(Image.open(src).convert('RGB')).astype(int)
H,W,_ = im.shape
dark = (im.sum(axis=2) < 300)
# frame lines: columns/rows where a long run of dark pixels exists
colfrac = dark.mean(axis=0); rowfrac = dark.mean(axis=1)
def runs(mask):
    r=[];s=None
    for i,v in enumerate(mask):
        if v and s is None: s=i
        if not v and s is not None: r.append((s,i-1)); s=None
    if s is not None: r.append((s,len(mask)-1))
    return r
cols = runs(colfrac>0.7); rows = runs(rowfrac>0.7)
print('cols',cols); print('rows',rows)
assert len(cols)==8 and len(rows)==6, 'frame detection failed'
xs=[(cols[2*i][0],cols[2*i+1][1]) for i in range(4)]
ys=[(rows[2*i][0],rows[2*i+1][1]) for i in range(3)]
labels=[['Original','RCM','AMD','Rabbit'],['GROOT','Gray','Metis','PaToH'],['SlashBurn','DTC-LSH','Degree','Random']]
fig,axes=plt.subplots(3,4,figsize=(14,11.2))
for r,(y0,y1) in enumerate(ys):
    for c,(x0,x1) in enumerate(xs):
        # title strip above the panel
        ty0 = max(0,y0-110); strip = im[ty0:y0-5, x0:x1]
        nonwhite = strip[(strip.sum(axis=2) < 700)]
        color = tuple((np.median(nonwhite,axis=0)/255).tolist()) if len(nonwhite) else (0,0,0)
        panel = im[y0:y1+1, x0:x1+1].astype(np.uint8)
        ax=axes[r][c]; ax.imshow(panel); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.8)
        ax.set_title(labels[r][c], fontsize=22, fontweight='bold', color=color, pad=8, family='serif')
plt.subplots_adjust(left=0.01,right=0.99,top=0.96,bottom=0.01,hspace=0.16,wspace=0.04)
plt.savefig(dst, dpi=300, bbox_inches='tight', pad_inches=0.05)
print('saved', dst)
