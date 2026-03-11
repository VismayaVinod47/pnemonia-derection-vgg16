# Flask Frontend for Federated VGG

This starter app serves a simple web UI for image upload and model inference.

## 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Add your trained model

Place your trained weights at:

- `federated_vgg.pt`

## 3) Update model + labels

- Edit `model_definition.py` so the architecture exactly matches training.
- Edit `CLASS_NAMES` in `app.py` to match your training label order.
- If needed, update `transform` in `app.py` to match training preprocessing.

## 4) Run

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).
