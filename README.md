# Text Simplification with T5

A text simplification project using T5 transformer model. This project provides a complete pipeline for training and evaluating text simplification models.

## Features

- **T5-based Model**: Utilizes the T5 transformer architecture for text simplification
- **Flexible Configuration**: YAML-based configuration for easy model and training customization
- **Multiple Metrics**: Evaluates using BLEU, ROUGE, and BERTScore
- **Efficient Data Processing**: Caches processed datasets for faster training
- **Comprehensive Logging**: Detailed logging and result tracking

## Project Structure

```bash
text_simplification/
├── configs/
│   ├── t5_config.yaml      # Training configuration
│   └── eval_config.yaml    # Evaluation configuration
├── scripts/
│   └── src/
│       ├── data/           # Data processing modules
│       ├── evaluation/     # Evaluation modules
│       ├── models/         # Model implementation
│       ├── training/       # Training modules
│       └── utils/          # Utility functions
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ngocthien2306/t5-text-simplication
cd t5-text-simplication
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv t5
source t5/bin/activate  # Linux/Mac
# or
.\t5\Scripts\activate  # Windows
```

```
conda create --name t5 python=3.10
conda activate t5
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```


## Downloading Pre-trained Models and Datasets

### Pre-trained Model
You can download our pre-trained model and datasets from Google Drive:

1. **Checkpoint Model**: Download the fine-tuned model checkpoint from [Google Drive](https://drive.google.com/drive/folders/1e45NuCzrmH-DpVydOdb0msu8sSHZL4Ob)
   - Download the `checkpoint-xxxx` folder
   - Place it in the root directory of the project

2. **Datasets**: In the same Google Drive link
   - Download the `datasets` folder
   - Place it in the root directory of the project

Your directory structure should look like this after downloading:
```bash
text_simplification/
├── checkpoint-26470/        # Downloaded model checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── datasets/               # Downloaded datasets
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── src/
├── scripts/
└── ...
```

## Overall Performance

| Metric | Score |
|--------|-------|
| BLEU | 0.3510 |
| ROUGE-L | 0.5904 |
| BERTScore | 0.8237 |


## Configuration

The project uses two main configuration files:

### Training Configuration (`configs/t5_config.yaml`)

```yaml
data:
  root_dir: "../datasets/"
  train_file: "train.jsonl"
  val_file: "valid.jsonl"
  test_file: "test.jsonl"

model:
  name: "t5-base"
  save_dir: "./t5_simplification"
  max_input_length: 512
  max_target_length: 128

training:
  gpu_devices: "2,4,5,7"
  learning_rate: 5e-5
  train_batch_size: 16
  eval_batch_size: 8
```

### Evaluation Configuration (`configs/eval_config.yaml`)

```yaml
model:
  checkpoint_path: "./t5_simplification"
  gpu_device: "7"

generation:
  max_length: 128
  num_beams: 4
  length_penalty: 2.0

evaluation:
  metrics:
    bleu: true
    rouge: true
    bert_score: true
```

## Usage

### Training

To train the model:

```bash
python train.py --config configs/t5_config.yaml
```

### Evaluation

To evaluate the model:

```bash
python evaluate.py --config configs/eval_config.yaml --batch_size 16
```

## Data Format

The training data should be in JSONL format with the following structure:

```json
{"src": "complex sentence", "trg": "simplified sentence"}
```

## Metrics

The evaluation pipeline includes multiple metrics:

- **BLEU**: Machine translation evaluation metric
- **ROUGE**: Set of metrics for evaluating text summarization
- **BERTScore**: Contextual similarity metric using BERT embeddings

## Results

Results are saved in the specified output directory with the following structure:

```bash
results/
└── evaluation_{timestamp}/
    ├── metrics.json        # Detailed metrics
    └── predictions.txt     # Model predictions
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{text_simplification_t5,
  author = {Thien Nguyen, Andre, Dicky, Robit},
  title = {Text Simplification with T5},
  year = {2024},
  url = {https://github.com/ngocthien2306/t5-text-simplication}
}
```

## Acknowledgments

- Hugging Face Team for the Transformers library
- Google Research for the T5 model