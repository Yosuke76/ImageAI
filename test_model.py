import torch

from models.plant_model import create_plant_classifier, save_model, load_model

def main():
    num_classes = 5  # Example number of plant species
    image_channels = 3
    image_height = 64
    image_width = 64
    batch_size = 4

    print("Starte Testlauf für PlantClassifier....")

    model = create_plant_classifier(
        num_classes=num_classes,
        image_channels=image_channels,
        image_height=image_height,
        image_width=image_width
    )

   

    dummy_input = torch.randn(batch_size, image_channels, image_height, image_width)
   # Run a forward pass
    output = model(dummy_input)

    print("Output-Shape vor dem Speichern:", output.shape)

    save_path = "plant_model_test.pth"
    save_model(model, save_path)
    print(f"Modell gespeichert unter: {save_path}")

    loaded_model = load_model(
        save_path,
        num_classes=num_classes,
        image_channels=image_channels,
        image_height=image_height,
        image_width=image_width,
    )

    output_loaded = loaded_model(dummy_input)
    print("Output-Shape nach dem Laden:", output_loaded.shape)

if __name__ == "__main__":
    main()