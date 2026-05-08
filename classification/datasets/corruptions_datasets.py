
import os
import csv
import json
import torch
import logging
from typing import Optional, Sequence
from functools import lru_cache
from pathlib import Path

from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder, CustomCifarDataset, default_loader

logger = logging.getLogger(__name__)


CLASSIFICATION_DIR = Path(__file__).resolve().parents[1]
ROBUSTBENCH_DATA_DIR = CLASSIFICATION_DIR / "robustbench" / "data"
IMAGENET_CLASS_TO_ID_MAP_PATH = ROBUSTBENCH_DATA_DIR / "imagenet_class_to_id_map.json"
IMAGENET_TEST_IDS_PATH = ROBUSTBENCH_DATA_DIR / "imagenet_test_image_ids.txt"
IMAGENET_VAL_IDS_50K_PATH = Path(__file__).resolve().parent / "imagenet_list" / "imagenet_val_ids_50k.txt"


@lru_cache(maxsize=None)
def load_imagenet_class_to_id_map():
    with IMAGENET_CLASS_TO_ID_MAP_PATH.open("r") as f:
        return json.load(f)


def _list_subdirectories(root: Path):
    return sorted([path.name for path in root.iterdir() if path.is_dir()])


@lru_cache(maxsize=None)
def get_imagenet_subset_wnids(data_dir: str, expected_num_classes: Optional[int] = None):
    data_root = Path(data_dir)

    for corruption in CORRUPTIONS:
        corruption_dir = data_root / corruption
        if not corruption_dir.is_dir():
            continue

        for severity in range(1, 6):
            severity_dir = corruption_dir / str(severity)
            if not severity_dir.is_dir():
                continue

            wnids = _list_subdirectories(severity_dir)
            if not wnids:
                continue

            unknown_wnids = sorted(set(wnids) - set(load_imagenet_class_to_id_map().keys()))
            if unknown_wnids:
                raise ValueError(f"Found unknown ImageNet WNIDs in '{severity_dir}': {unknown_wnids[:5]}")

            if expected_num_classes is not None and len(wnids) != expected_num_classes:
                raise ValueError(
                    f"Expected {expected_num_classes} classes in '{severity_dir}', but found {len(wnids)}. "
                    "Please verify the ImageNet100-C directory structure."
                )

            return tuple(wnids)

    raise FileNotFoundError(
        f"Could not find any corruption/severity/class folders under '{data_dir}'. "
        "Expected a layout like <root>/<corruption>/<severity>/<wnid>/*.JPEG"
    )


@lru_cache(maxsize=None)
def get_imagenet_subset_indices(data_dir: str, expected_num_classes: Optional[int] = None):
    class_to_idx = load_imagenet_class_to_id_map()
    return tuple(class_to_idx[wnid] for wnid in get_imagenet_subset_wnids(data_dir, expected_num_classes))


def resolve_stream_image_path(raw_path: str, data_dir: str, corruption: Optional[str] = None,
                              severity: Optional[str] = None, label: Optional[str] = None):
    raw_path = Path(raw_path)
    data_root = Path(data_dir)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(data_root / raw_path)

    if corruption is not None and severity is not None and label is not None:
        candidates.append(data_root / str(corruption) / str(severity) / str(label) / raw_path.name)

    root_name = data_root.name
    if root_name in raw_path.parts:
        idx = len(raw_path.parts) - 1 - raw_path.parts[::-1].index(root_name)
        candidates.append(data_root.joinpath(*raw_path.parts[idx + 1:]))

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not resolve stream image path '{raw_path}' under DATA_DIR='{data_dir}'. "
        "If you generated the CSV on a different machine, either regenerate it there or keep the same dataset root layout."
    )


class ImageNet100CStreamDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, data_dir: str, transform=None, max_samples: int = -1, expected_num_classes: int = 100):
        self.csv_path = Path(csv_path)
        self.data_dir = data_dir
        self.transform = transform

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Stream CSV not found: {self.csv_path}")

        subset_wnids = list(get_imagenet_subset_wnids(data_dir, expected_num_classes=expected_num_classes))
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(subset_wnids)}

        with self.csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        required_columns = {"position", "path", "label", "corruption", "severity", "is_shift"}
        if not rows:
            raise RuntimeError(f"Stream CSV is empty: {self.csv_path}")
        missing_columns = required_columns - set(rows[0].keys())
        if missing_columns:
            raise ValueError(f"Stream CSV '{self.csv_path}' is missing columns: {sorted(missing_columns)}")

        rows.sort(key=lambda row: int(row["position"]))
        if max_samples != -1:
            rows = rows[:max_samples]

        self.rows = []
        for row in rows:
            label = row["label"]
            if label not in self.class_to_idx:
                raise ValueError(
                    f"Label '{label}' from stream CSV '{self.csv_path}' is not part of the detected ImageNet100 subset."
                )

            resolved_path = resolve_stream_image_path(
                raw_path=row["path"],
                data_dir=data_dir,
                corruption=row.get("corruption"),
                severity=row.get("severity"),
                label=label,
            )

            self.rows.append({
                "path": resolved_path,
                "label_idx": self.class_to_idx[label],
                "label_str": label,
                "corruption": row["corruption"],
                "severity": int(row["severity"]),
                "is_shift": int(row["is_shift"]),
                "position": int(row["position"]),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = default_loader(row["path"])
        if self.transform is not None:
            sample = self.transform(sample)

        return (
            sample,
            torch.tensor(row["label_idx"], dtype=torch.long),
            row["corruption"],
            torch.tensor(row["severity"], dtype=torch.long),
            torch.tensor(row["is_shift"], dtype=torch.long),
            torch.tensor(row["position"], dtype=torch.long),
            row["path"],
        )


def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)

def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual',
    class_to_idx: Optional[dict] = None,
    file_list_path: Optional[str] = None):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting or "correlated" in setting or n_examples != -1 or "continual" in setting or "reset_each_shift" in setting:
    # if "mixed_domains" in setting or "correlated" in setting or n_examples != -1:
        # load imagenet class to id mapping from robustbench
        class_to_idx = class_to_idx or load_imagenet_class_to_id_map()

        if n_examples != -1 or "correlated" in setting or "continual" in setting or "mixed_domains" in setting or "reset_each_shift" in setting:
        # if n_examples != -1 or "correlated" in setting:
            # create file path of file containing all 50k image ids
            file_path = file_list_path or str(IMAGENET_VAL_IDS_50K_PATH)
        else:
            # create file path of default test list from robustbench
            file_path = file_list_path or str(IMAGENET_TEST_IDS_PATH)

        # load file containing file ids
        with open(file_path, 'r') as f:
            fnames = f.readlines()

        item_list = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            item_list += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
        dataset_test.samples = item_list

    return dataset_test
