"""Official CIFAR-10 preparation; imported only by training and offline evaluation."""
from pathlib import Path
import hashlib
import json
import pickle
import tarfile
import urllib.request
import numpy as np
import torch
from .model import patchify

URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
MD5 = 'c58f30108f718f92721af3b95e74349a'
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def prepare(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    archive = root/'cifar-10-python.tar.gz'
    if not archive.exists():
        tmp = archive.with_suffix('.download')
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != MD5:
        raise ValueError('Official archive MD5 mismatch')
    def load_batch(tar, name):
        # Only named regular files from the checksum-verified official archive; never extract paths.
        f = tar.extractfile('cifar-10-batches-py/'+name)
        obj = pickle.load(f, encoding='bytes')
        return obj[b'data'].reshape(-1,3,32,32), np.asarray(obj[b'labels'], dtype=np.int64)
    with tarfile.open(archive) as tar:
        batches = [load_batch(tar, 'data_batch_'+str(i)) for i in range(1,6)]
        images = np.concatenate([b[0] for b in batches]); labels = np.concatenate([b[1] for b in batches])
        test_images, test_labels = load_batch(tar, 'test_batch')
    rng = np.random.default_rng(61023)
    train, val, selection = [], [], []
    for cls in range(10):
        indices = rng.permutation(np.flatnonzero(labels == cls))
        val.extend(indices[:500]); selection.extend(indices[:100]); train.extend(indices[500:])
    train, val, selection = (np.asarray(sorted(v), dtype=np.int64) for v in (train,val,selection))
    arrays = dict(train_images=images[train], train_labels=labels[train], val_images=images[val], val_labels=labels[val], test_images=test_images, test_labels=test_labels)
    for name, arr in arrays.items():
        np.save(root/(name+'.npy'), arr)
    np.savez(root/'splits.npz', train=train, val=val, selection_global=selection,
             selection_val=np.searchsorted(val, selection), test=np.arange(10000))
    manifest = dict(source=URL, official_page='https://cave.cs.toronto.edu/kriz/cifar.html', archive=str(archive.resolve()),
                    md5=digest, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), split_seed=61023,
                    sizes={k:len(v) for k,v in arrays.items()}, train_counts=np.bincount(arrays['train_labels']).tolist(),
                    val_counts=np.bincount(arrays['val_labels']).tolist(), augmentation=False,
                    preprocessing='RGB BCHW uint8; y=2*((pixel+U[0,1))/256)-1; validation RNG CPU seed=91023, batch=64, order fixed',
                    classes=CLASSES, files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.npy')})
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


def load_split(root, split):
    root = Path(root)
    return torch.from_numpy(np.load(root/(split+'_images.npy'))), torch.from_numpy(np.load(root/(split+'_labels.npy')))


def dequantize(images, generator):
    noise = torch.rand(images.shape, generator=generator)
    return patchify(2*((images.float()+noise)/256)-1)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--out', default='data/p1/cifar10')
    prepare(p.parse_args().out)
