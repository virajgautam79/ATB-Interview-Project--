# Office Space Utilization & Occupancy Analysis

A comprehensive Python project that loads, cleans, and analyzes office space data. This repository provides exploratory data analysis (EDA) on key metrics—such as average occupancy, utilization, and hours used—from an office dataset. Visualizations are generated using Matplotlib and Seaborn with a custom, professional color palette.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Requirements](#data-requirements)


## Overview

This project performs a detailed analysis of office space data contained in a CSV file (`office_data.csv`). It cleans the dataset, computes various descriptive statistics, and produces multiple visualizations including histograms, box plots, scatter plots, correlation matrices, and time series trends. The goal is to understand space utilization, occupancy, and usage patterns across different buildings and space types (e.g., Desks and Meeting Rooms).

## Features

- **Data Loading & Cleaning:**  
  - Imports and cleans CSV data, handling missing values and data type conversions.
  - Converts percentage strings to floats and time strings to minutes.

- **Exploratory Data Analysis (EDA):**  
  - Univariate analysis: descriptive statistics, histograms, and box plots.
  - Bivariate analysis: correlation matrices, scatter plots, and pair plots.
  - Outlier detection and analysis with heatmaps and bar plots.

- **Building & Floor Analysis:**  
  - Groups data by building and space type.
  - Computes weighted averages and medians for utilization and occupancy.
  - Analyzes metrics across different floors and capacities.

- **Time Series Analysis:**  
  - Resamples the data by week and month.
  - Visualizes seasonal trends and monthly median metrics.

- **Customized Visualization:**  
  - Uses  ATB color palette and professional chart formatting for clarity and style.

## Data Requirements

The analysis requires a CSV file named `office_data.csv` with the following key columns:
- **Week:** Date (used for time series analysis)
- **Building:** Name/identifier of the building (e.g., Calgary Building 1)
- **Floor:** Floor number or identifier
- **Space Type:** Type of space (e.g., Desks, Meeting Room)
- **Capacity:** Capacity of the space
- **Avg. Occupancy:** Average occupancy (provided as a percentage string)
- **Avg Utilization:** Average utilization (provided as a percentage string)
- **Avg Hours Used (HH:MM):** Average hours used in HH:MM format


