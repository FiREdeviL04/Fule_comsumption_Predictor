"""Tkinter desktop application for fuel consumption prediction."""
from __future__ import annotations

from collections import deque
from datetime import datetime
from tkinter import END, LEFT, VERTICAL, messagebox, StringVar, Tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from model.predict import FuelPredictionService
from model.utils import (
    PredictionInput,
    TARGET_LABELS,
    append_prediction_history,
    clean_dataset,
    load_dataset,
    load_prediction_history,
)
from ui.components import create_labeled_entry, create_section


class FuelPredictionApp:
    """Desktop GUI for training-backed fuel consumption predictions."""

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Fuel Consumption Predictor")
        self.root.geometry("1320x900")
        self.root.minsize(1180, 820)

        self._configure_style()

        try:
            self.predictor = FuelPredictionService.from_disk()
            self.dataset = clean_dataset(load_dataset())
        except Exception as exc:  # pragma: no cover - startup safety
            messagebox.showerror("Startup Error", str(exc))
            raise

        self.sample_row = self.dataset.sample(n=1, random_state=42).iloc[0]
        self.history_records = deque(maxlen=5)
        for record in load_prediction_history(limit=5).to_dict("records"):
            self.history_records.append(record)

        self.engine_size_var = StringVar()
        self.cylinders_var = StringVar()
        self.emissions_var = StringVar()
        self.status_var = StringVar(value="Ready")

        self.prediction_text_vars = {
            "fuel_model": StringVar(value="Fuel Consumption: --"),
            "hwy_model": StringVar(value="Highway Fuel Consumption: --"),
            "comb_model": StringVar(value="Combined Fuel Consumption: --"),
        }
        self.insight_text_vars = {
            "fuel_model": StringVar(value="R²: -- | MAE: -- | CV Mean: --"),
            "hwy_model": StringVar(value="R²: -- | MAE: -- | CV Mean: --"),
            "comb_model": StringVar(value="R²: -- | MAE: -- | CV Mean: --"),
        }

        self.prediction_canvas = None
        self.importance_canvas = None
        self.insight_importance_canvas = None
        self.feature_importance_items = []

        self._build_interface()
        self._populate_sample_data()
        self._refresh_model_insights()
        self._refresh_history_view()
        self._refresh_graphs()

    def _configure_style(self) -> None:
        """Configure a clean, modern ttk theme."""

        style = ttk.Style(self.root)
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")

        style.configure("TFrame", background="#f4f7fb")
        style.configure("TLabel", background="#f4f7fb", foreground="#1f2937", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#0f172a")
        style.configure("Section.TLabelframe", background="#f4f7fb")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 11, "bold"), foreground="#0f172a")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background="#1d4ed8", foreground="white")
        style.map("Accent.TButton", background=[("active", "#1e40af")], foreground=[("active", "white")])

    def _build_interface(self) -> None:
        """Build the notebook pages and bind layout events."""

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Fuel Consumption Predictor", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Random Forest prediction system with validation, persistence, and model insights.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        self.prediction_tab = ttk.Frame(self.notebook, padding=16)
        self.graph_tab = ttk.Frame(self.notebook, padding=16)
        self.insights_tab = ttk.Frame(self.notebook, padding=16)

        self.notebook.add(self.prediction_tab, text="Prediction")
        self.notebook.add(self.graph_tab, text="Graph")
        self.notebook.add(self.insights_tab, text="Model Insights")

        self._build_prediction_tab()
        self._build_graph_tab()
        self._build_insights_tab()

    def _build_prediction_tab(self) -> None:
        """Construct the prediction inputs, actions, and history sections."""

        self.prediction_tab.columnconfigure(0, weight=2)
        self.prediction_tab.columnconfigure(1, weight=1)
        self.prediction_tab.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(self.prediction_tab)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_panel.columnconfigure(1, weight=1)

        input_section = create_section(left_panel, "Prediction Inputs")
        input_section.grid(row=0, column=0, sticky="ew")
        input_section.columnconfigure(1, weight=1)

        self.engine_entry = create_labeled_entry(input_section, "Engine Size", self.engine_size_var, 0)
        self.cylinders_entry = create_labeled_entry(input_section, "Cylinders", self.cylinders_var, 1)
        self.emissions_entry = create_labeled_entry(input_section, "Emissions", self.emissions_var, 2)

        self.engine_entry.configure(validate="key", validatecommand=(self.root.register(self._validate_float_text), "%P"))
        self.cylinders_entry.configure(validate="key", validatecommand=(self.root.register(self._validate_integer_text), "%P"))
        self.emissions_entry.configure(validate="key", validatecommand=(self.root.register(self._validate_integer_text), "%P"))

        actions = ttk.Frame(left_panel)
        actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(actions, text="Predict", style="Accent.TButton", command=self._on_predict).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(actions, text="Reset", command=self._reset_form).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(actions, text="Load Sample Data", command=self._populate_sample_data).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        progress_section = create_section(left_panel, "Status")
        progress_section.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        progress_section.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(progress_section, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_section, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(8, 0))

        results_section = create_section(left_panel, "Predictions")
        results_section.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        results_section.columnconfigure(0, weight=1)

        ttk.Label(results_section, textvariable=self.prediction_text_vars["fuel_model"]).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(results_section, textvariable=self.prediction_text_vars["hwy_model"]).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(results_section, textvariable=self.prediction_text_vars["comb_model"]).grid(row=2, column=0, sticky="w", pady=4)

        right_panel = ttk.Frame(self.prediction_tab)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

        history_section = create_section(right_panel, "Last 5 Predictions")
        history_section.grid(row=0, column=0, sticky="nsew")
        history_section.columnconfigure(0, weight=1)
        history_section.rowconfigure(0, weight=1)

        list_container = ttk.Frame(history_section)
        list_container.grid(row=0, column=0, sticky="nsew")
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)

        self.history_listbox = ttk.Treeview(
            list_container,
            columns=("timestamp", "summary"),
            show="headings",
            height=13,
        )
        self.history_listbox.heading("timestamp", text="Time")
        self.history_listbox.heading("summary", text="Summary")
        self.history_listbox.column("timestamp", width=150, anchor="w")
        self.history_listbox.column("summary", width=280, anchor="w")
        self.history_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(list_container, orient=VERTICAL, command=self.history_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.history_listbox.configure(yscrollcommand=scrollbar.set)

        notes = create_section(right_panel, "Input Notes")
        notes.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        notes.columnconfigure(0, weight=1)
        notes.rowconfigure(0, weight=1)

        message = (
            "Only Engine Size, Cylinders, and Emissions are used by the models. "
            "The app validates numeric input, saves every prediction, and shows a rolling history."
        )
        ttk.Label(notes, text=message, wraplength=360, justify=LEFT).grid(row=0, column=0, sticky="nw")

    def _build_graph_tab(self) -> None:
        """Build chart containers for predictions and feature importance."""

        self.graph_tab.columnconfigure(0, weight=1)
        self.graph_tab.columnconfigure(1, weight=1)
        self.graph_tab.rowconfigure(0, weight=1)

        prediction_chart_section = create_section(self.graph_tab, "Prediction Chart")
        prediction_chart_section.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        prediction_chart_section.columnconfigure(0, weight=1)
        prediction_chart_section.rowconfigure(0, weight=1)

        importance_chart_section = create_section(self.graph_tab, "Feature Importance")
        importance_chart_section.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        importance_chart_section.columnconfigure(0, weight=1)
        importance_chart_section.rowconfigure(0, weight=1)

        self.prediction_chart_holder = ttk.Frame(prediction_chart_section)
        self.prediction_chart_holder.grid(row=0, column=0, sticky="nsew")
        self.prediction_chart_holder.columnconfigure(0, weight=1)
        self.prediction_chart_holder.rowconfigure(0, weight=1)
        self.importance_chart_holder = ttk.Frame(importance_chart_section)
        self.importance_chart_holder.grid(row=0, column=0, sticky="nsew")
        self.importance_chart_holder.columnconfigure(0, weight=1)
        self.importance_chart_holder.rowconfigure(0, weight=1)

    def _build_insights_tab(self) -> None:
        """Create the metrics dashboard for all trained models."""

        self.insights_tab.columnconfigure(0, weight=1)
        self.insights_tab.rowconfigure(1, weight=1)

        table_section = create_section(self.insights_tab, "Model Performance")
        table_section.grid(row=0, column=0, sticky="ew")
        table_section.columnconfigure(0, weight=1)

        self.metrics_table = ttk.Treeview(
            table_section,
            columns=("model", "target", "r2", "mae", "cv", "params"),
            show="headings",
            height=6,
        )
        for column, heading, width in [
            ("model", "Model", 120),
            ("target", "Target", 220),
            ("r2", "R²", 90),
            ("mae", "MAE", 90),
            ("cv", "CV Mean", 100),
            ("params", "Best Parameters", 500),
        ]:
            self.metrics_table.heading(column, text=heading)
            self.metrics_table.column(column, width=width, anchor="w")
        self.metrics_table.grid(row=0, column=0, sticky="ew")

        insights_body = ttk.Frame(self.insights_tab)
        insights_body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        insights_body.columnconfigure(0, weight=1)
        insights_body.columnconfigure(1, weight=1)
        insights_body.rowconfigure(0, weight=1)

        summary_section = create_section(insights_body, "Model Summary")
        summary_section.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        summary_section.columnconfigure(0, weight=1)

        for row_index, model_name in enumerate(["fuel_model", "hwy_model", "comb_model"]):
            ttk.Label(summary_section, textvariable=self.insight_text_vars[model_name]).grid(row=row_index, column=0, sticky="w", pady=8)

        chart_section = create_section(insights_body, "Feature Importance Snapshot")
        chart_section.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        chart_section.columnconfigure(0, weight=1)
        chart_section.rowconfigure(0, weight=1)
        self.insight_importance_holder = ttk.Frame(chart_section)
        self.insight_importance_holder.grid(row=0, column=0, sticky="nsew")
        self.insight_importance_holder.columnconfigure(0, weight=1)
        self.insight_importance_holder.rowconfigure(0, weight=1)

    def _validate_float_text(self, proposed_value: str) -> bool:
        """Allow only empty input or valid floating-point text while typing."""

        if proposed_value == "":
            return True
        try:
            float(proposed_value)
        except ValueError:
            return False
        return True

    def _validate_integer_text(self, proposed_value: str) -> bool:
        """Allow only empty input or integer text while typing."""

        if proposed_value == "":
            return True
        return proposed_value.isdigit()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        """Toggle the loading indicator and status message."""

        self.status_var.set(message or ("Working..." if busy else "Ready"))
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _populate_sample_data(self) -> None:
        """Fill the input fields with a realistic dataset sample."""

        self.engine_size_var.set(f"{float(self.sample_row['ENGINE SIZE']):.1f}")
        self.cylinders_var.set(str(int(self.sample_row['CYLINDERS'])))
        self.emissions_var.set(str(int(self.sample_row['EMISSIONS'])))
        self.status_var.set("Sample data loaded")

    def _reset_form(self) -> None:
        """Clear the input fields and reset result labels."""

        self.engine_size_var.set("")
        self.cylinders_var.set("")
        self.emissions_var.set("")
        for model_name in self.prediction_text_vars:
            suffix = "L/100 km" if model_name in {"fuel_model", "hwy_model", "comb_model"} else ""
            self.prediction_text_vars[model_name].set(f"{TARGET_LABELS[model_name]}: -- {suffix}".strip())
        self.status_var.set("Form reset")

    def _read_inputs(self) -> PredictionInput:
        """Read and validate the user input fields."""

        engine_text = self.engine_size_var.get().strip()
        cylinders_text = self.cylinders_var.get().strip()
        emissions_text = self.emissions_var.get().strip()

        if not engine_text or not cylinders_text or not emissions_text:
            raise ValueError("All input fields are required.")

        try:
            engine_size = float(engine_text)
            cylinders = int(cylinders_text)
            emissions = int(emissions_text)
        except ValueError as exc:
            raise ValueError("Please enter numeric values only.") from exc

        if engine_size <= 0:
            raise ValueError("Engine size must be greater than zero.")
        if cylinders <= 0:
            raise ValueError("Cylinders must be greater than zero.")
        if emissions <= 0:
            raise ValueError("Emissions must be greater than zero.")

        return PredictionInput(
            engine_size=engine_size,
            cylinders=cylinders,
            emissions=emissions,
        )

    def _on_predict(self) -> None:
        """Run prediction, update the UI, and persist the result."""

        try:
            prediction_input = self._read_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self._set_busy(True, "Running prediction...")
        self.root.update_idletasks()

        try:
            predictions = self.predictor.predict(prediction_input)
            metrics = self.predictor.get_metrics()
        except Exception as exc:  # pragma: no cover - runtime safety
            self._set_busy(False, "Prediction failed")
            messagebox.showerror("Prediction Error", str(exc))
            return

        self._set_busy(False, "Prediction complete")

        self._update_prediction_labels(predictions)
        self._update_insight_labels(metrics)
        self._append_history(prediction_input, predictions)
        self._refresh_history_view()
        self._refresh_graphs(predictions)
        self._refresh_model_insights()

    def _update_prediction_labels(self, predictions) -> None:
        """Display the prediction results in the output labels."""

        self.prediction_text_vars["fuel_model"].set(
            f"Fuel Consumption: {predictions['fuel_model']:.2f} L/100 km"
        )
        self.prediction_text_vars["hwy_model"].set(
            f"Highway Fuel Consumption: {predictions['hwy_model']:.2f} L/100 km"
        )
        self.prediction_text_vars["comb_model"].set(
            f"Combined Fuel Consumption: {predictions['comb_model']:.2f} L/100 km"
        )

    def _update_insight_labels(self, metrics) -> None:
        """Refresh the summary labels with the latest stored metrics."""

        for model_name in ["fuel_model", "hwy_model", "comb_model"]:
            model_metrics = metrics.get(model_name, {})
            self.insight_text_vars[model_name].set(
                "R²: {r2:.3f} | MAE: {mae:.3f} | CV Mean: {cv:.3f}".format(
                    r2=float(model_metrics.get("r2_score", 0.0)),
                    mae=float(model_metrics.get("mae", 0.0)),
                    cv=float(model_metrics.get("cv_mean_score", 0.0)),
                )
            )

    def _append_history(self, prediction_input: PredictionInput, predictions) -> None:
        """Save the current prediction to disk and the in-memory history."""

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engine_size": prediction_input.engine_size,
            "cylinders": prediction_input.cylinders,
            "emissions": prediction_input.emissions,
            "fuel_prediction": round(predictions["fuel_model"], 4),
            "hwy_prediction": round(predictions["hwy_model"], 4),
            "comb_prediction": round(predictions["comb_model"], 4),
        }
        append_prediction_history(record)
        self.history_records.append(record)

    def _refresh_history_view(self) -> None:
        """Rebuild the prediction history treeview."""

        for row_id in self.history_listbox.get_children():
            self.history_listbox.delete(row_id)

        for record in reversed(list(self.history_records)):
            summary = (
                f"E:{record['engine_size']} | C:{record['cylinders']} | EM:{record['emissions']} | "
                f"Fuel:{record['fuel_prediction']} | HWY:{record['hwy_prediction']} | COMB:{record['comb_prediction']}"
            )
            self.history_listbox.insert("", END, values=(record["timestamp"], summary))

    def _clear_canvas(self, canvas_attribute: str) -> None:
        """Destroy any existing Matplotlib canvas before drawing a new one."""

        canvas = getattr(self, canvas_attribute)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
            setattr(self, canvas_attribute, None)

    def _refresh_graphs(self, predictions=None) -> None:
        """Render the prediction and importance charts."""

        self._clear_canvas("prediction_canvas")
        self._clear_canvas("importance_canvas")
        self._clear_canvas("insight_importance_canvas")

        if predictions is None:
            predictions = {
                "fuel_model": 0.0,
                "hwy_model": 0.0,
                "comb_model": 0.0,
            }

        prediction_figure = Figure(figsize=(5.2, 4.4), dpi=110)
        prediction_axis = prediction_figure.add_subplot(111)
        labels = ["Fuel", "Highway", "Combined"]
        values = [
            predictions["fuel_model"],
            predictions["hwy_model"],
            predictions["comb_model"],
        ]
        colors = ["#2563eb", "#16a34a", "#dc2626"]
        prediction_axis.bar(labels, values, color=colors)
        prediction_axis.set_title("Predicted Consumption")
        prediction_axis.set_ylabel("L/100 km")
        prediction_axis.grid(axis="y", linestyle="--", alpha=0.3)
        prediction_figure.tight_layout()

        self.prediction_canvas = FigureCanvasTkAgg(prediction_figure, master=self.prediction_chart_holder)
        self.prediction_canvas.draw()
        chart_widget = self.prediction_canvas.get_tk_widget()
        chart_widget.grid(row=0, column=0, sticky="nsew")

        importance_values = self.predictor.get_feature_importances("comb_model")
        importance_figure = Figure(figsize=(5.2, 4.4), dpi=110)
        importance_axis = importance_figure.add_subplot(111)
        if importance_values:
            feature_names = [feature for feature, _ in importance_values]
            scores = [score for _, score in importance_values]
            importance_axis.barh(feature_names, scores, color="#7c3aed")
            importance_axis.set_xlabel("Importance")
            importance_axis.set_title("Combined Model Feature Importance")
            importance_axis.grid(axis="x", linestyle="--", alpha=0.3)
        else:
            importance_axis.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
            importance_axis.set_axis_off()
        importance_figure.tight_layout()

        self.importance_canvas = FigureCanvasTkAgg(importance_figure, master=self.importance_chart_holder)
        self.importance_canvas.draw()
        importance_widget = self.importance_canvas.get_tk_widget()
        importance_widget.grid(row=0, column=0, sticky="nsew")

        insight_figure = Figure(figsize=(5.2, 4.4), dpi=110)
        insight_axis = insight_figure.add_subplot(111)
        if importance_values:
            feature_names = [feature for feature, _ in importance_values]
            scores = [score for _, score in importance_values]
            insight_axis.barh(feature_names, scores, color="#7c3aed")
            insight_axis.set_xlabel("Importance")
            insight_axis.set_title("Feature Importance Snapshot")
            insight_axis.grid(axis="x", linestyle="--", alpha=0.3)
        else:
            insight_axis.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
            insight_axis.set_axis_off()
        insight_figure.tight_layout()

        self.insight_importance_canvas = FigureCanvasTkAgg(insight_figure, master=self.insight_importance_holder)
        self.insight_importance_canvas.draw()
        insight_widget = self.insight_importance_canvas.get_tk_widget()
        insight_widget.grid(row=0, column=0, sticky="nsew")

    def _refresh_model_insights(self) -> None:
        """Populate the metrics table with model performance data."""

        for row_id in self.metrics_table.get_children():
            self.metrics_table.delete(row_id)

        for row in self.predictor.get_display_rows():
            self.metrics_table.insert(
                "",
                END,
                values=(
                    row["Model"],
                    row["Target"],
                    f"{float(row['R2']):.3f}",
                    f"{float(row['MAE']):.3f}",
                    f"{float(row['CV Mean']):.3f}",
                    str(row["Best Params"]),
                ),
            )

    def run(self) -> None:
        """Launch the Tkinter event loop."""

        self.root.mainloop()
