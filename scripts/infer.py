import hashlib
import json
import sys
from collections.abc import MutableMapping
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import torch
import yaml
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/src'))
from climax.arch import ClimaX

URL = 'https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
VARS = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
INDICES = [0, 120, 1460, 5840, 20000, 40000, 60000, 80000]
MEAN = np.array([280., 0., 0.], dtype=np.float32)[:, None, None]
STD = np.array([20., 10., 10.], dtype=np.float32)[:, None, None]
CHECKSUM = '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43'


class RemoteStore(MutableMapping):
    def __getitem__(self, key):
        try:
            with urlopen(f'{URL}/{key}', timeout=90) as response:
                return response.read()
        except HTTPError as error:
            if error.code == 404:
                raise KeyError(key) from error
            raise

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise TypeError('read-only')

    def __delitem__(self, key):
        raise TypeError('read-only')


def read_data():
    store = RemoteStore()
    arrays = [zarr.open_array(store=store, path=name, mode='r') for name in VARS]
    latitude = np.asarray(zarr.open_array(store=store, path='latitude', mode='r')[:], dtype=np.float64)
    assert latitude.shape == (32,)
    assert all(array.shape[0] > max(INDICES) + 1 for array in arrays)
    samples = []
    for index in INDICES:
        x = np.stack([np.asarray(a[index], dtype=np.float32).T for a in arrays])
        y = np.stack([np.asarray(a[index + 1], dtype=np.float32).T for a in arrays])
        assert x.shape == y.shape == (3, 32, 64)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        samples.append((x, y))
    return latitude, samples


def load_model():
    config = yaml.safe_load((ROOT / 'upstream/configs/pretrain_climax.yaml').read_text())
    model = ClimaX(**config['model']['net']['init_args'])
    checkpoint = ROOT / 'checkpoints/5.625deg.ckpt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKSUM
    state = torch.load(checkpoint, map_location='cpu')['state_dict']
    renamed = {}
    for name, tensor in state.items():
        assert name.startswith('net.')
        name = name[4:]
        for old, new in [('channel_embed', 'var_embed'), ('channel_query', 'var_query'), ('channel_agg.', 'var_agg.')]:
            if name.startswith(old):
                name = new + name[len(old):]
                break
        renamed[name] = tensor
    model.load_state_dict(renamed, strict=True)
    return model.cuda().eval(), len(renamed)


def predict(model, x):
    normalized = (x - MEAN) / STD
    inputs = torch.from_numpy(normalized[None]).cuda()
    lead_time = torch.tensor([0.06], device='cuda')
    with torch.no_grad():
        _, output = model(inputs, None, lead_time, VARS, VARS, None, None)
    prediction = output[0].cpu().numpy() * STD + MEAN
    assert prediction.shape == x.shape and np.isfinite(prediction).all()
    return prediction


def scores(prediction, target, weights):
    diff = prediction.astype(np.float64) - target.astype(np.float64)
    mae = np.average(np.abs(diff), weights=weights)
    rmse = np.sqrt(np.average(diff ** 2, weights=weights))
    return {'mae': float(mae), 'rmse': float(rmse)}


def main():
    latitude, samples = read_data()
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None], (32, 64)).copy()
    assert np.isfinite(weights).all() and np.all(weights > 0)
    model, tensor_count = load_model()
    rows, inputs, targets, predictions = [], [], [], []
    for index, (x, target) in zip(INDICES, samples):
        prediction = predict(model, x)
        inputs.append(x)
        targets.append(target)
        predictions.append(prediction)
        sample_results = {}
        for i, name in enumerate(VARS):
            sample_results[name] = {
                'model': scores(prediction[i], target[i], weights),
                'persistence': scores(x[i], target[i], weights),
            }
        rows.append({'index': index, 'results': sample_results})

    repeated = predict(model, samples[0][0])
    difference = float(np.max(np.abs(repeated.astype(np.float64) - predictions[0].astype(np.float64))))
    assert np.allclose(repeated, predictions[0], rtol=1e-5, atol=1e-5)

    x_all = np.stack(inputs)
    y_all = np.stack(targets)
    p_all = np.stack(predictions)
    all_weights = np.broadcast_to(weights, (len(INDICES), 32, 64))
    aggregate = {}
    for i, name in enumerate(VARS):
        aggregate[name] = {
            'model': scores(p_all[:, i], y_all[:, i], all_weights),
            'persistence': scores(x_all[:, i], y_all[:, i], all_weights),
        }
    report = {
        'scope': 'Diagnostic only: CMIP6 pretrained weights, assumed normalization, no ERA5 forecast fine-tuning',
        'data': URL,
        'sample_indices': INDICES,
        'lead_hours_ground_truth': 6,
        'model_lead_time_argument': 0.06,
        'variables': VARS,
        'assumed_mean': MEAN[:, 0, 0].tolist(),
        'assumed_std': STD[:, 0, 0].tolist(),
        'checkpoint_sha256': CHECKSUM,
        'checkpoint_tensors': tensor_count,
        'source_commit': '6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c',
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'metric': 'latitude-cosine-area-weighted MAE and RMSE, unnormalized physical units',
        'repeat_max_abs_difference': difference,
        'aggregate': aggregate,
        'per_sample': rows,
    }
    destination = ROOT / 'results/era5'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(destination / 'prediction.npz', input=x_all, target=y_all, prediction=p_all, latitude=latitude, indices=INDICES)
    print(json.dumps({key: report[key] for key in ('scope', 'sample_indices', 'repeat_max_abs_difference', 'aggregate')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
