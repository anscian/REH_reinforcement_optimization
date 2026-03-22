from PIL import Image

box_downsample = lambda im, sz : im.resize(sz, resample=Image.Resampling.BOX)

nearest_upsample = lambda im, sz : im.resize(sz, resample=Image.Resampling.NEAREST)