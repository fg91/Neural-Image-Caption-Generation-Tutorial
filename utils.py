from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patheffects import Stroke, Normal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
    
def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)

def draw_outline(matplt_plot_obj, lw):
    matplt_plot_obj.set_path_effects([Stroke(linewidth=lw, foreground='black'), Normal()])

def show_img(im, figsize=None, ax=None, alpha=1, cmap=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha, cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def visualize_attention(im, pred, alphas, denorm, vocab, att_size=14, sz=224):
    cap_len = len(pred)
    alphas = alphas.view(-1,1,  att_size, att_size).cpu().data.numpy()
    alphas = np.maximum(0, alphas)
    alphas -= alphas.min()
    alphas /= alphas.max()
    
    #figure, axes = plt.subplots(cap_len//3 + 1,3, figsize=(10,8))
    figure, axes = plt.subplots(4,3, figsize=(10,8))

    for i, ax in enumerate(axes.flat):
        if i < cap_len:
            ax = show_img(denorm(im), ax=ax)
            if i > 0:
                mask = np.array(Image.fromarray(alphas[i - 1,0]).resize((sz,sz)))
                blurred_mask = gaussian_filter(mask, sigma=10)
                show_img(blurred_mask, ax=ax, alpha=0.5, cmap='afmhot')
                draw_text(ax, (0,0), vocab.itos[pred[i - 1]])
            else:
                ax.axis('off')
    plt.tight_layout()
