# shallow-image-classification

## description
- A simple image classification implementation, where description and classification are done seperately.
- Descriptions used
    - tiny image (tiny).
    - bag of visual words (bow).
- Classifiers used
    - k nearest neighbour (knn).
    - support vector (svc).
- This is unlike a deep\[-neural-network\] classification, where description and classification are done together.

## roadmap
Problems in `hw3.pdf` are solved.

## code
- All source code is in `scene_recognition.py`.
- All data is in `scene_classification_data/`.
- Description of data format is given in `hw3.pdf`.

## documentation
- Code is the documentation of itself.

## usage
- Use `python3 scene_recognition.py` to classify images and visualize results using
    - `tiny + knn`.
    - `bow + knn`.
    - `bow + svc`.
- A summary of the methods and corresponding results is given in `report.pdf`.

## demonstration
- Tiny image description.

![](./github/tiny.png)

- `tiny + knn` confusion matrix.

![](./github/tiny+knn.png)

- `bow + knn` confusion matrix.

![](./github/bow+knn.png)

- `bow + svc` confusion matrix.

![](./github/bow+svc.png)
