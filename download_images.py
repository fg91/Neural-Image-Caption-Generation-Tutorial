import pandas as pd
from PIL import Image
import requests
from pathlib import Path
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("datafile", help='tsv file containing image captions and urls', type=str)
parser.add_argument("startIdx", type=int)
parser.add_argument("endIdx", type=int)
args = parser.parse_args()

captions_and_links = pd.read_csv(args.datafile, sep="\t",header=None)

def get_image_w_caption(df, PATH, idx):
    try:
        caption, link = df.iloc(0)[idx]
        img = Image.open(requests.get(link, stream=True, timeout=10).raw)
        img.save(PATH/(str(idx)+".png"), format='png')
    except:
        return None
    return str(str(idx)+".png"), caption

PATH = Path('data/')
SUB_PATH = PATH/'downloaded'
PATH.mkdir(exist_ok=True)
SUB_PATH.mkdir(exist_ok=True)

images = {}
for i in range(args.startIdx, args.endIdx + 1, 1):
    result = get_image_w_caption(captions_and_links, SUB_PATH, i)
    if result is not None:
        images[i] = result
    if i % 1000 == 0:
        pickle.dump(images,
                    (PATH/('dict_' + str(args.startIdx)
                           + '-' + str(args.endIdx) + '.pkl')).open('wb'))
        print('saved')
    if i % 100 == 0:
        print(i,'/', args.endIdx - args.startIdx)

pickle.dump(images,
            (PATH/('dict_' + str(args.startIdx) + '-' + str(args.endIdx) + '.pkl')).open('wb'))
