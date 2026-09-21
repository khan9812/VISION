"""Resolve executable settings and identify inputs to reusable computations."""
from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NOISE2SR_DEFAULTS = {
    'lr': 1e-4, 'epoch': 1500, 'batch_size': 12, 'patch_size': 128,
    's': 2, 'M': 50, 'gpu': 0, 'seed': 42,
    'num_workers': 0, 'persistent_workers': False, 'loader_generator_seed': None,
}


def resolve_noise2sr_config(overrides=None, *, parameters=None, epochs=None, image_shape=None):
    """YAML defaults < explicit JSON settings < explicit epoch argument.

    The returned dictionary is also the network's execution record. A small
    image can reduce the training patch; that resolved size enters the cache.
    """
    if parameters is None:
        parameters = yaml.safe_load((ROOT/'configs/sam2.1/default_parameters.yaml').read_text(encoding='utf-8'))
    cfg = dict(NOISE2SR_DEFAULTS)
    source = dict(parameters.get('noise2sr', {}))
    aliases = {'scale_factor': 's', 'inference_averages': 'M'}
    for key, value in source.items():
        key = aliases.get(key, key)
        if key in cfg:
            cfg[key] = value
    if overrides:
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError(f'Unknown Noise2SR settings: {sorted(unknown)}')
        cfg.update(overrides)
    if epochs is not None:
        cfg['epoch'] = epochs
    for key in ('epoch', 'batch_size', 'patch_size', 's', 'M'):
        if int(cfg[key]) != cfg[key] or int(cfg[key]) < 1:
            raise ValueError(f'Noise2SR {key} must be a positive integer')
        cfg[key] = int(cfg[key])
    if cfg['s'] != 2:
        raise ValueError('The released Noise2SR architecture uses stride s=2')
    if float(cfg['lr']) <= 0:
        raise ValueError('Noise2SR learning rate must be positive')
    if int(cfg['num_workers']) < 0:
        raise ValueError('Noise2SR num_workers must be non-negative')
    cfg['num_workers'] = int(cfg['num_workers'])
    cfg['seed'] = int(cfg['seed'])
    cfg['persistent_workers'] = bool(cfg['persistent_workers']) and cfg['num_workers'] > 0
    if image_shape is not None:
        shortest = min(image_shape[:2])
        if shortest < 2:
            raise ValueError('Noise2SR requires both image dimensions to be at least 2 pixels')
        cfg['patch_size'] = min(cfg['patch_size'], shortest)
    cfg['patch_size'] -= cfg['patch_size'] % cfg['s']
    if cfg['patch_size'] < 2:
        raise ValueError('Noise2SR patch size must be at least 2 pixels')
    return cfg


def runtime_signature():
    """Invalidate cached outputs when source, YAML, libraries or weights change."""
    digest = sha256()
    for folder, pattern in (('modules','*.py'), ('app','*.py'), ('analysis','*.py'), ('configs','*.yaml'), ('configs','*.json')):
        for path in sorted((ROOT/folder).rglob(pattern)):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    packages = {}
    for name in ('torch','torchvision','sam-2','openai-clip','numpy','scipy','pandas','bm3d','shapely'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    weights = {}
    for path in (ROOT/'checkpoints/sam2.1_hiera_large.pt', Path.home()/'.cache/clip/ViT-L-14-336px.pt'):
        stat = path.stat() if path.is_file() else None
        weights[path.name] = {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns} if stat else None
    return {'source_sha256':digest.hexdigest(),'packages':packages,'weight_file_identity':weights}


def preprocessing_record(image_shape, epochs=None):
    """Execution settings required to reuse a validation preprocessing image."""
    return {'runtime': runtime_signature(),
            'noise2sr': resolve_noise2sr_config(epochs=epochs, image_shape=image_shape)}


def preprocessing_record_matches(path, image_shape, epochs=None):
    """Legacy images without full execution metadata are not fresh-run caches."""
    try:
        record = json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
        return record.get('release_preprocessing') == preprocessing_record(image_shape, epochs)
    except (OSError, ValueError, TypeError):
        return False
