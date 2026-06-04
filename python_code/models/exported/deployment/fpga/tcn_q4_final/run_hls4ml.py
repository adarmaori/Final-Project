import json
from pathlib import Path

import hls4ml

pkg_path = Path(__file__).resolve().parent / 'hls4ml_package.json'
pkg = json.loads(pkg_path.read_text(encoding='utf-8'))
cfg = hls4ml.utils.config_from_onnx_model(pkg['onnx_model'], granularity='name')
cfg.update(pkg['hls_config'])
prj = hls4ml.converters.convert_from_onnx_model(
    pkg['onnx_model'],
    hls_config=cfg,
    output_dir=str((Path(__file__).resolve().parent / 'hls4ml_prj')),
    backend=pkg['backend'],
    io_type=pkg['io_type'],
)
prj.compile()
print('hls4ml project generated at', Path(__file__).resolve().parent / 'hls4ml_prj')