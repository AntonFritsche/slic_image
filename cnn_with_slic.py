import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchviz import make_dot
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
from tqdm import tqdm
from pathlib import Path
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.segmentation import slic


def resnet50_with_slic(img_dir: str, num_classes: int, num_segments: int):
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)

    summary(model, (3, 224, 224))

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    summary(model, (3, 224, 224))

class Model(nn.Module):
    def __init__(self, num_blocks: int, num_classes: int, num_segment_features: int) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.num_segment_features = num_segment_features
        self.cnn_out_dim = 256
        self.mlp_out_dim = 256

        # model part 1
        self.cnn_blocks = nn.ModuleList()
        for i in range(1, num_blocks+1):
            in_dim = 3 ** i
            out_dim = 3 ** (i + 1)
            block = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_dim)
            )
            self.cnn_blocks.append(block)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_classifier = nn.Linear(3 ** (num_blocks + 1), self.cnn_out_dim)

        # model part 2
        self.mlp = nn.Sequential(
            nn.Linear(num_segment_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, self.mlp_out_dim),
            nn.ReLU()
        )

        # Fusion Head
        combined_dim = self.mlp_out_dim + self.cnn_out_dim
        self.fusion_head_classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        for block in self.cnn_blocks:
            x = block(x)
            print(x.shape)
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        x_img = self.cnn_classifier(x) # Shape: (B, self.cnn_out_dim)

        x_tab = self.mlp(tabular_features).view(x_img.shape[0], -1)  # Shape: (B, self.mlp_out_dim)

        # Konkatenation
        print("shapes: ", x_img.shape, x_tab.shape)
        x_combined = torch.cat((x_img, x_tab), dim=1)

        logits = self.fusion_head_classifier(x_combined)
        return logits

def cnn_with_slic(img_dir: str, num_classes: int, num_segments: int, num_segment_features: int):
    model = Model(num_blocks=5, num_classes=num_classes, num_segment_features=num_segment_features)

    dummy_image = torch.randn(1, 3, 120, 120, requires_grad=True)
    dummy_segment_features = torch.randn(1, num_segment_features, requires_grad=True)
    output = model(dummy_image, dummy_segment_features)
    make_dot(
        output,
        params=dict(list(model.named_parameters()))).render("model_view/fusion_model", format="png"
    )

    #summary(model, ((3, 120, 120), (1, num_segment_features)))

def extract_features_from_dataset(
    csv_path: str | Path,
    images_dir: str | Path,
    output_csv_path: str | Path,
    num_segments: int = 100,
    sigma: float = 5.0,
    compactness: float = 10.0,
    filename_col: str = "Name",
    label_col: str = "Type1"
) -> pd.DataFrame:
    df_meta = pd.read_csv(csv_path)
    img_dir = Path(images_dir)
    all_superpixel_data = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Verarbeite Bilder"):
        img_name = str(row[filename_col]) + ".png"
        img_label = row[label_col]
        img_path = img_dir / img_name

        if not img_path.is_file():
            print(f"Übersprungen (nicht gefunden): {img_name}")
            continue

        try:
            image = io.imread(img_path)

            # Transparenzkanal verwerfen, falls vorhanden
            if image.ndim == 3 and image.shape[-1] == 4:
                image = image[..., :3]

            # Graustufenbild für GLCM-Texturfeatures vorbereiten
            if image.ndim == 3:
                gray = color.rgb2gray(image)
            else:
                gray = image.astype(float) / 255.0
            gray_ubyte = (gray * 255).astype(np.uint8)

            # SLIC-Segmentierung
            segments = slic(
                image,
                n_segments=num_segments,
                sigma=sigma,
                compactness=compactness,
                start_label=1
            )

            # Regionseigenschaften bestimmen
            props = regionprops(segments, intensity_image=gray)

            for prop in props:
                label_id = prop.label
                mask = (segments == label_id)

                # 1. Farbmerkmale (Mittelwert und Standardabweichung je Kanal)
                if image.ndim == 3:
                    mean_r = float(np.mean(image[mask, 0]))
                    mean_g = float(np.mean(image[mask, 1]))
                    mean_b = float(np.mean(image[mask, 2]))
                    std_r  = float(np.std(image[mask, 0]))
                    std_g  = float(np.std(image[mask, 1]))
                    std_b  = float(np.std(image[mask, 2]))
                else:
                    mean_r = mean_g = mean_b = float(np.mean(image[mask]))
                    std_r = std_g = std_b = float(np.std(image[mask]))

                # 2. Formmerkmale
                area = prop.area
                centroid_y, centroid_x = prop.centroid
                eccentricity = prop.eccentricity

                # 3. Texturmerkmale via GLCM (Gray-Level Co-occurrence Matrix)
                minr, minc, maxr, maxc = prop.bbox
                sub_gray = gray_ubyte[minr:maxr, minc:maxc]
                sub_mask = mask[minr:maxr, minc:maxc]
                masked_patch = np.where(sub_mask, sub_gray, 0)

                glcm = graycomatrix(
                    masked_patch,
                    distances=[1],
                    angles=[0],
                    levels=256,
                    symmetric=True,
                    normed=True
                )
                contrast = float(graycoprops(glcm, "contrast")[0, 0])
                homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
                energy = float(graycoprops(glcm, "energy")[0, 0])

                # Zeile zusammenstellen
                all_superpixel_data.append({
                    "image_name": img_name,
                    "superpixel_id": label_id,
                    "target_label": img_label,
                    "area": area,
                    "centroid_y": centroid_y,
                    "centroid_x": centroid_x,
                    "eccentricity": eccentricity,
                    "mean_r": mean_r,
                    "mean_g": mean_g,
                    "mean_b": mean_b,
                    "std_r": std_r,
                    "std_g": std_g,
                    "std_b": std_b,
                    "contrast": contrast,
                    "homogeneity": homogeneity,
                    "energy": energy,
                })

        except Exception as e:
            tqdm.tqdm.write(f"Fehler bei {img_name}: {e}")

    # Als CSV speichern
    feature_df = pd.DataFrame(all_superpixel_data)
    feature_df.to_csv(output_csv_path, index=False)
    print(f"\nFertig! Datensatz mit {len(feature_df)} Zeilen gespeichert unter: {output_csv_path}")
    return feature_df

if __name__ == "__main__":
    num_segments: int = 300
    num_classes: int = 10
    num_segment_features: int = 15

    #extract_features_from_dataset(
    #    "./archive/pokemon.csv",
    #    "./archive/images",
    #    "./archive/superpixel_features.csv",
    #    num_segments=num_segments,
    #)
    cnn_with_slic("", num_classes=num_classes, num_segments=num_segments, num_segment_features=num_segment_features)