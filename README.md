# Lightweight Object Detection for African Penguin Population Monitoring

Code, results and documents for my BSc (Honours) Computer Science thesis at Rhodes University (2026), supervised by Prof. Dane Brown.

**Project website:** https://sites.google.com/rucis.co.za/23x6327/
**Thesis:** [Docs/Hluma_Final_Thesis_2026.pdf](Docs/Hluma_Final_Thesis_2026.pdf)

## About the project

The African Penguin is Critically Endangered, and monitoring it depends on accurate population counts. Published penguin detectors have mostly been tested on dedicated GPUs, which field teams rarely have. This project tests whether lightweight YOLO detectors can count African Penguins accurately on a standard CPU.

The nano and medium versions of YOLOv11 and YOLOv26 were trained on a merged dataset of 37,487 penguin images, labelled with SAM 3 and checked by hand. Each model was evaluated on detection accuracy, counting accuracy and CPU latency, using Intel OpenVINO and native PyTorch on an Intel Core Ultra 5 235.

## Main results

Grouped-split test set of 3,383 photographs. Latency is the mean per image with OpenVINO FP32 on the CPU.

| Model | mAP@50 | mAP@50:95 | Total count error | Latency (ms) | Images per second |
|---|---|---|---|---|---|
| **YOLOv11n** (recommended) | 0.862 | 0.657 | -5.4% | 15.9 | 62.9 |
| YOLOv26n | 0.854 | 0.657 | -11.1% | 14.8 | 67.7 |
| YOLOv11m | 0.900 | 0.728 | -1.1% | 111.0 | 9.0 |
| YOLOv26m | 0.901 | 0.738 | -1.0% | 111.5 | 9.0 |

- The nano models stayed within 5 points of the medium models on mAP@50.
- OpenVINO was 1.34 to 1.56 times faster than PyTorch for every model.
- YOLOv26n was only 7% faster than YOLOv11n but undercounted twice as much, so YOLOv11n with OpenVINO is recommended for counting on a standard CPU.

## Repository layout

| Folder | Contents |
|---|---|
| `Original_Split/` | `original_split.ipynb`: the first dataset build, training and evaluation (Experiments 1.1 and 4.1) |
| `Grouped_Split/` | `grouped_split.ipynb`: the grouped 70/20/10 dataset, training and Experiments 1.2 to 5. `make_thesis_figures.py` draws the figures |
| `Tools/` | Supporting scripts: hyperparameter tuning, SAM 3 annotation, label conversion and an OpenVINO device check |
| `Results/` | Result files (CSV) for both splits |
| `Extra_Experiments/` | A speed-only benchmark on the SAMA-COCO validation set, not reported in the thesis |
| `Docs/` | The thesis and seminar slides |

## Experiments

| Experiment | What it tests | Where |
|---|---|---|
| 1.1 | Detection and counting accuracy, original split | `Original_Split/original_split.ipynb` |
| 1.2 | Detection and counting accuracy, grouped split | `Grouped_Split/grouped_split.ipynb`, sections 3 and 4 |
| 2 | CPU latency, OpenVINO against PyTorch | `grouped_split.ipynb`, section 5 |
| 3 | OpenVINO FP32 against FP16 | `grouped_split.ipynb`, section 6 |
| 4.1, 4.2 | Hyperparameter tuning of YOLOv11n | both notebooks, using `Tools/yolo_any_tune.py` |
| 5 | Scale augmentation of YOLOv11n | `grouped_split.ipynb`, section 8 |

## Running the code

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
   For GPU training, install the CUDA build of PyTorch first, following pytorch.org.
2. Change the paths at the top of each notebook to point to your copy of the data.
3. Run the notebook cells in order. Each training cell takes several hours on an RTX 4070, and the latency timing should run with nothing else using the machine.

| Software | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0 (CUDA 13.0) |
| Ultralytics | 8.4.41 |
| OpenVINO | 2026.2.0 |

Training ran on an NVIDIA RTX 4070 (12 GB). Benchmarking ran on an Intel Core Ultra 5 235 CPU with 14 threads.

## Data

The images are not included, because of their size and licences. They come from:

- African, Magellanic, Humboldt and Galápagos penguin photographs on iNaturalist, including the [Penguins of South Africa](https://www.inaturalist.org/projects/penguins-of-south-africa) project
- The GEORa camera of the Penguin Watch project, from the [Oxford penguin counting dataset](https://www.robots.ox.ac.uk/~vgg/data/penguins/)
- Seven penguin datasets hosted on Roboflow Universe (listed in Table 3.1 of the thesis)

All images were labelled with SAM 3 and checked by hand in Roboflow.

## Notes

- The original split's OpenVINO results include rows for the `intel:gpu.0` device. The Intel GPU runtime was not installed on the machine at the time, so these timings most likely ran on the CPU. They are not used in the thesis.
- `Tools/yolo_any_tune.py`, `Tools/bench_schema.py` and `Tools/sam3_coco_annotator.py` were written by Prof. Dane Brown and are included unchanged.

## Acknowledgement

This work was undertaken in the Distributed Multimedia CoE at Rhodes University, with financial support from Telkom SA. The authors acknowledge that opinions, findings and conclusions or recommendations expressed here are those of the author(s) and that the above-mentioned sponsor accepts no liability whatsoever in this regard.

## Citation

```
Xako, H. (2026). Lightweight Object Detection for African Penguin Population Monitoring.
BSc (Honours) thesis, Department of Computer Science, Rhodes University.
```
