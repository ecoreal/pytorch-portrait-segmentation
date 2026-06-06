# Dataset Notes

This repository currently includes 360 image/mask pairs:

- `picture/source360/`: source portrait images
- `picture/mask360/`: corresponding binary portrait masks

Each source image and mask should share the same file stem. For example:

```text
picture/source360/003ff269-1cec-4ec5-8782-a9566afbe05f.png
picture/mask360/003ff269-1cec-4ec5-8782-a9566afbe05f.png
```

## Release Checklist

Before publishing the repository publicly, confirm the data source and license:

- If the images and masks are self-created or fully authorized, document that here.
- If they come from a public dataset, add the dataset name, URL, citation, and original license.
- If the rights are unclear, replace the images with a small demo set that you are allowed to redistribute, or remove the dataset from the public repository and provide download instructions instead.

The MIT License in `LICENSE` applies to the source code. It does not automatically grant reuse rights for third-party images or annotations.
