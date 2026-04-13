"""Reusable Tkinter UI helpers."""
from __future__ import annotations

from tkinter import ttk


def create_labeled_entry(parent, label_text: str, text_variable, row: int, column: int = 0):
    """Create a label/entry pair and return the entry widget."""

    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=column, sticky="w", padx=(0, 12), pady=8)

    entry = ttk.Entry(parent, textvariable=text_variable)
    entry.grid(row=row, column=column + 1, sticky="ew", pady=8)
    return entry


def create_section(parent, title: str):
    """Create a titled frame for a page section."""

    frame = ttk.LabelFrame(parent, text=title, padding=12)
    frame.columnconfigure(1, weight=1)
    return frame
