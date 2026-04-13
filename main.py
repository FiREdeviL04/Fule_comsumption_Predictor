"""Application entry point for the fuel prediction desktop app."""

from ui.app import FuelPredictionApp


if __name__ == "__main__":
    app = FuelPredictionApp()
    app.run()