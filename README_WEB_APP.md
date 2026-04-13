# Car Fuel Prediction Web App

This upgrade adds a full web stack:

- Flask backend API with tuned Random Forest models
- React frontend with modern UI, charts, history, and insights

## Project Structure

- backend/
  - app.py
  - requirements.txt
  - data/
  - saved_models/
  - ml/
- frontend/
  - package.json
  - src/

## Backend Setup

1. Open terminal in backend folder.
2. Install dependencies:

   pip install -r requirements.txt

3. Run server:

   python app.py

Backend runs on http://localhost:5000

## Frontend Setup

1. Open terminal in frontend folder.
2. Install dependencies:

   npm install

3. Run development server:

   npm run dev

Frontend runs on http://localhost:5173

## API Endpoints

- GET /health
- POST /train
- POST /predict
- GET /insights
- GET /history
- GET /history/export

## Notes

- Models are automatically trained and persisted if missing.
- Prediction history is stored in backend/prediction_history.csv.
- /history/export downloads history as CSV.
