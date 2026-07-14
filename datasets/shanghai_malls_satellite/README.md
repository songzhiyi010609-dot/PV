# Full Shanghai mall satellite dataset

This directory stores the full Shanghai experiment outputs.

## Structure

- `data/`: geocoded mall index and imagery index.
- `images/`: exported satellite images for each Shanghai mall.
- `models/`: downloaded open-source model helper files.
- `results/`: model prediction CSVs and summary reports.
- `masks/`: predicted PV masks for positive/possible samples.
- `overlays/`: visual overlays for positive/possible samples.

The full run is based on `raw/上海市.csv`.

## Pipeline

```powershell
python .\satellite_experiment\scripts\build_shanghai_full_dataset.py --stage all
python .\satellite_experiment\scripts\run_bdappv_inference.py
python .\satellite_experiment\scripts\summarize_bdappv_results.py
```

The imagery uses Esri World Imagery export around each geocoded mall point.
The PV recognition experiment uses the open BDAPPV model weights from
`gabrielkasmi/bdappv-models` on Hugging Face.
