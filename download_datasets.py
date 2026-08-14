from src.automl.datasets import EmotionsDataset, FlowersDataset, FashionDataset, SkinCancerDataset

DATASETS = [EmotionsDataset, FlowersDataset, FashionDataset, SkinCancerDataset]

if __name__ == "__main__":
    for cls in DATASETS:
        print(f"Downloading {cls._dataset_name}...")
        cls(root="data", split="train", download=True)
        print(f"Done: {cls._dataset_name}\n")

    print("All datasets downloaded.")