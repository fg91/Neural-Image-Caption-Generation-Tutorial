from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patheffects import Stroke, Normal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# the functions fig2data and fig2img are taken from 
# http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
# Deprecation errors have been fixed

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
    
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

def visualize_attention(im, pred, alphas, denorm, vocab, att_size=7, thresh=0., sz=224, return_fig_as_PIL_image=False):
    cap_len = len(pred)
    alphas = alphas.view(-1,1,  att_size, att_size).cpu().data.numpy()
    alphas = np.maximum(thresh, alphas)
    alphas -= alphas.min()
    alphas /= alphas.max()
 
    figure, axes = plt.subplots(cap_len//5 + 1,5, figsize=(12,8))
 
    for i, ax in enumerate(axes.flat):
        if i <= cap_len:
            ax = show_img(denorm(im), ax=ax)
            if i > 0:
                mask = np.array(Image.fromarray(alphas[i - 1,0]).resize((sz,sz)))
                blurred_mask = gaussian_filter(mask, sigma=8)
                show_img(blurred_mask, ax=ax, alpha=0.5, cmap='afmhot')
                draw_text(ax, (0,0), vocab.itos[pred[i - 1]])
        else:
            ax.axis('off')
    plt.tight_layout()

    if return_fig_as_PIL_image:
        return fig2img(figure)


