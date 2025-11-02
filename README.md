# Kinematics Analyzer

This GUI is a Python-based tool (using PyQt5 for the interface and Matplotlib for plots) designed to mimic key features of commercial software like DIPS for rock slope stability analysis. It helps geologists/engineers analyze discontinuity data (e.g., joint orientations from a rock face) to identify potential failure modes like planar sliding, wedge sliding, or toppling.
The app works in a workflow: Load/Edit Data → Set Parameters & Cluster → Run Analysis → View Results & Plot. It's modular—data is processed in kinematics.py, and the UI in gui.py handles display/interaction.

