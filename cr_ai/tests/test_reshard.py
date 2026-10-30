"""BC 分片打散（tools/reshard_bc.py）：样本一个不少、一个不多，每个样本的单位帧原样带过去，同一局的样本被分散开。"""
import glob
import os
import shutil

import numpy as np
import pytest

from cr_ai.config import ROOT
from cr_ai.tools.reshard_bc import load, reshard


def _rows(files):
    """每个样本的指纹：(局号, 玩家, tick, 选项, 落点, 这一帧单位的校验和)。"""
    out = []
    for fn in files:
        d = load(fn)
        off = d['frame_off']
        for i in range(len(d['frame'])):
            f = d['frame'][i]
            u = d['units'][off[f]:off[f + 1]]
            out.append((int(d['game'][i]), int(d['player'][i]), int(d['feat_tick'][i]), int(d['sel'][i]), int(d['cell'][i]),
                        hash(u.tobytes())))
    return sorted(out)


def test_reshard_preserves_samples(tmp_path):
    src_all = sorted(glob.glob(os.path.join(ROOT, 'data_bc', 'il_replay', 'shard_*.npz')))[:4]
    if len(src_all) < 4:
        pytest.skip('没有 BC 分片')
    src = tmp_path / 'src'
    src.mkdir()
    for f in src_all:
        shutil.copy(f, src)
    dst = tmp_path / 'dst'
    reshard(str(src), str(dst), groups=3, chunk=2, workers=2, seed=1)
    out = sorted(glob.glob(str(dst / 'shard_*.npz')))
    assert not os.path.exists(dst / '_parts')
    assert _rows(sorted(glob.glob(str(src / 'shard_*.npz')))) == _rows(out)
    # 同一局的样本分散在多个输出分片里
    d = load(out[0])
    games = np.unique(d['game'])
    assert len(games) > 100                                           # 4 片 × 50 局几乎都出现在每个输出分片里
