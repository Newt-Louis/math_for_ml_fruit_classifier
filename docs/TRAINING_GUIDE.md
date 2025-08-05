# Model Training Guide

To evaluate the training process, we will run two separate training sessions with different configurations.

---

## Round 1: Basic Training

This initial training round is designed for a quick check of the pipeline and to generate a baseline model.

* **Epochs**: 1
* **Data Augmentation**: Disabled

**Execution:**

Run the following command to start training:

```bash
poetry run scripts.train_model.py

Upon completion, a weight file named fruit_classifier.pt will be generated. You can use this file for initial testing and evaluation.
```
---

## Round 2: Advanced Training

The second round involves a full training process aimed at producing the best-performing model.

* **Epochs**: 10
* **Data Augmentation**: Enabled

**Execution:**

1.  **Start Re-training:**
    Run the command below to begin the advanced training process. This will take a significant amount of time, estimated at **6-8 hours**.

    ```bash
    poetry run scripts.re_train.py
    ```

2.  **Evaluate the Model:**
    After training is complete, you will have 10 epoch files for evaluation and one final model, `fruit_classifier_best.pt`, which represents the best-performing model. Run the following command to evaluate the results:

    ```bash
    poetry run scripts.evaluate.py
    ```
---

## Running without Poetry

If you prefer not to use Poetry, you can run the scripts directly with Python:

```bash
# Advanced training command
python run scripts.re_train.py

# Evaluation command
python run script.evaluate.py
```
---

## 📝 Important Note

To use the desired trained model, ensure you update the model's filename in the configuration file.

* **Configuration File**: `src/core/config.php`
* **Example**: Change the model path to point to `fruit_classifier_best.pt` to use the best model.