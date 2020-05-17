import os
import glob
import torch
from torchvision import transforms
import numpy as np
import itertools, operator
import pydicom
import nibabel as nib
from scipy.misc.pilutil import toimage
import argparse


def get_study(path):
    def load_dcm(x):
        try:
            return pydicom.read_file(x)
        except:
            return None

    def sort_dicoms(loaded_dcm): return sorted(loaded_dcm, key=lambda y: float(y[('0020', '0032')][2]))

    files = glob.glob(os.path.join(path, '*'))
    dcms = [load_dcm(x) for x in files]
    dcms = [x for x in dcms if x is not None]
    assert len(dcms) > 0

    dcms = sort_dicoms(dcms)

    intercept = float(dcms[0][('0028', '1052')].value)
    slope = float(dcms[0][('0028', '1053')].value)

    xy_spacing = [float(x) for x in dcms[0][('0028', '0030')].value]
    z_spacing = abs(dcms[0][('0020', '1041')].value - dcms[1][('0020', '1041')].value)

    pixel_spacing = [z_spacing] + xy_spacing

    out = np.stack([x.pixel_array for x in dcms])

    out_array = out.astype('float32') * slope + intercept

    return out_array, pixel_spacing


class FakeMultiChannel(object):
    def __init__(self):
        pass

    def __call__(self, x):
        x = x.astype(np.float32) / 255
        x = np.stack([x] * 3, axis=2)
        return toimage(x, channel_axis=2)


class toTensor(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.tensor(x)


def get_abdomen(preds):
    abdomen = max((list(y) for (x, y) in itertools.groupby((enumerate(preds)), operator.itemgetter(1)) if x == 0),
                key=len)
    bottom = min(abdomen)[0]
    top = max(abdomen)[0]
    return slice(bottom, top +1)


def run(in_dir, out_file):
    ct_array, spacing = get_study(in_dir)

    m = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch.pth'))
    m = m.eval()

    tfs = transforms.Compose([FakeMultiChannel(),
                              transforms.Resize(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
    )

    preds = []
    for i in range(len(ct_array)):
        p = m(tfs(ct_array[i]).unsqueeze(0).cuda())
        preds.append(torch.argmax(p).cpu().item())


    interior = get_abdomen(preds)
    interior_images = ct_array[interior]

    nii_img = nib.Nifti1Image(interior_images, np.eye(4))
    nib.save(nii_img, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dicom_directory', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    assert os.path.isdir(args.dicom_directory), 'Dicom directory not found'
    run(args.dicom_directory, args.output_file)
