import pandas as pd
from PIL import Image
from PIL.Image import LANCZOS
import requests
from pathlib import Path
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("datafile", help='tsv file containing image captions and urls', type=str)
parser.add_argument("startIdx", type=int)
parser.add_argument("endIdx", type=int)
parser.add_argument("maxSize", type=int, nargs='?', default=None)
args = parser.parse_args()

captions_and_links = pd.read_csv(args.datafile, sep="\t",header=None)

def calc_new_size(img, max_sz):
    width, height = img.size
    smaller_side, resize_ratio = (width, max_sz/width) if width < height else (height, max_sz/height)
    if smaller_side <= max_sz:
        return None
    else:
        return (int(width * resize_ratio), int(height * resize_ratio))

def get_image_w_caption(df, PATH, idx):
    try:
        caption, link = df.iloc(0)[idx]
        img = Image.open(requests.get(link, stream=True, timeout=10).raw)
        if args.maxSize is not None:
            new_size = calc_new_size(img, args.maxSize)
            if new_size is not None:
                img = img.resize(new_size, resample=LANCZOS)
        img.save(PATH/(str(idx)+".png"), format='png')
    except:
        return None
    return str(str(idx)+".png"), caption

PATH = Path('data/')
SUB_PATH = PATH/'downloadedPics'
PATH.mkdir(exist_ok=True)
SUB_PATH.mkdir(exist_ok=True)

images = {}
for i in range(args.startIdx, args.endIdx + 1, 1):
    result = get_image_w_caption(captions_and_links, SUB_PATH, i)
    if result is not None:
        images[i] = result
    if i % 1000 == 1:
        pickle.dump(images,
                    (PATH/('dict_' + str(args.startIdx)
                           + '-' + str(args.endIdx) + '.pkl')).open('wb'))
        print('saved')
    if i % 100 == 0:
        print(i,'/', args.endIdx - args.startIdx)

pickle.dump(images,
            (PATH/('dict_' + str(args.startIdx) + '-' + str(args.endIdx) + '.pkl')).open('wb'))
