# Campus Crowd Predictor

An end-to-end machine-learning classification pipeline that estimates the expected campus canteen crowd level as **Low**, **Medium**, or **High**.

## Project overview

The model uses three input features:

- `hour`: hour of the observation
- `day`: day number (1 = Monday through 7 = Sunday)
- `event`: 0 if no campus event is recorded, 1 if an event is happening

The target label is created from `queue_count` using these demonstration thresholds:

| Queue count | Crowd level |
|---:|---|
| 0–7 | Low |
| 8–17 | Medium |
| 18 or more | High |

A Decision Tree Classifier is trained using an 80/20 train-test split.

## Project files

- `main.py` — complete training and prediction script
- `crowd_data.csv` — dataset required to run the script (add this file to the repository)
- `requirements.txt` — Python libraries required
- `Project_Report.pdf` — project documentation

## Dataset format

Place `crowd_data.csv` in the same folder as `main.py`. It must contain these columns:

```csv
hour,day,event,queue_count
12,3,1,25
```

The row above is only an example of the required CSV format, not a claim about a real campus observation. The report describes a 63-row synthetic practice dataset; use the actual CSV that belongs to your project.

## How to run

1. Install Python.
2. Open a terminal in this project folder.
3. (Optional) Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the pipeline:

   ```bash
   python main.py
   ```

## Machine-learning workflow

1. Read the CSV using Pandas.
2. Inspect rows, dimensions, data types, and missing values.
3. Calculate the mean and median queue count using NumPy.
4. Create the `crowd_level` target label.
5. Select `hour`, `day`, and `event` as model inputs.
6. Split the data into training and testing sets.
7. Train a Decision Tree Classifier.
8. Generate predictions and compare them with actual test labels.

## Important limitations

The report states that the included crowd observations are synthetic practice data, not measurements from a real campus canteen. The dataset is small, the Low/Medium/High thresholds are chosen for demonstration, and the model uses only three input features. Results should therefore be treated as a learning demonstration, not as evidence of actual campus crowd levels.

## Future improvements

Collect real queue observations across multiple days and time slots, validate the crowd thresholds, and retrain and evaluate the model on representative data.
