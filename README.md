# SSURGO Soil Analysis System

A comprehensive Python toolkit for processing and analyzing SSURGO (Soil Survey Geographic Database) data using advanced machine learning techniques including Variational Autoencoders (VAE) and clustering algorithms.

## Overview

This system processes SSURGO soil data and provides an end-to-end pipeline to:
- Prepare SSURGO spatial + tabular soil data
- Extract and aggregate soil properties by depth layers (10cm, 30cm, 100cm)
- Apply advanced clustering techniques using VAE for dimensionality reduction
- Generate comprehensive soil classification and analysis reports
- Create interactive visualizations of soil property distributions
- Perform spatial analysis with geographic information systems (GIS) integration

## Features
### Core Functionality
- **Multi-depth Analysis**: Process soil horizons at 10cm, 30cm, and 100cm depths
- **Advanced ML Pipeline**: VAE + K-Means, Agglomerative, Birch, Fuzzy C-Means, and Gaussian Mixture clustering
- **Comprehensive Evaluation**: Silhouette Score, and QC metrics such as Calinski-Harabasz Index, Gap Statistic, elbow method
- **Generate visualizations of latent spaces, soil property distributions, and cluster areas
- **Spatial Integration**: Integrate results back into MU polygons for spatial mapping, GIS processing with coordinate transformations and area calculations
- **Quality Control**: Automated data validation and component coverage analysis

### Data Processing Modules
- **Geographic Processing**: Spatial data handling and coordinate system management
- **Horizon Processing**: Soil layer aggregation and organic carbon calculations
- **Restriction Processing**: Soil limitation and constraint analysis
- **Component Analysis**: Major component identification and percentage calculations
- **Utils**: Weighted averaging, horizon→component→MU transformations

## Quick Start

### Prerequisites
- Python 3.8+
- GDAL/OGR libraries for spatial data processing
- Required Python packages (see requirements.txt)

### Installation
\`\`\`bash
# Clone or download the project
git clone <repository-url>
cd ssurgo-soil-analysis

# install dependencies
pip install -r requirements.txt

# Verify GDAL installation
python -c "import gdal; print('GDAL version:', gdal.__version__)"

## Data Requirements
## Input Data
## Download Missouri state gSSURGO data from the link : https://nrcs.app.box.com/v/soils/file/1680543039768 
Note: process the downloaded gSSURGO data in GIS software and convert them into the recommended file format.
Place the following in your data/ directory:
- **Spatial Data**: mupolygon.shp (spatial polygons)
- **Tabular Data**: mapunit.csv, component.csv, muagg.csv, corestrictions.csv, chorizon.csv, chfrags.csv
- **Geographic Boundaries**: County or study area boundaries

### Data Format
All input data should follow SSURGO database schema standards. The system expects:
- Consistent MUKEY identifiers across all datasets
- Proper coordinate reference systems for spatial data
- Complete attribute tables with required fields

## Pipeline Analysis Workflow
1. **Sha_pipeline_runbook.py** → End-to-end runbook that executes all the python files
2. **main.py** → integrates SSURGO spatial + tabular data, outputs main_df.csv & prepared_df.parquet
3. **data_preparation.py** → scales data, outputs data_scaled.npy
4. **vae_training.py** → trains VAE, outputs z_mean.npy
5. **clustering_evaluation.py** → clustering model selection, label generation over a latent feature space
6. **clustering_selection.py** → evaluates methods/k ranges, outputs best_k_for.json
7. **clustering_algorithms.py** → run one method+k, outputs main_df_with_{method}_k{k}.csv
8. **metric_plots.py** → enerate clustering metrics plots over k
9. **latent_plots.py** → plots latent 2D scatter
10. **visualization.py** → scatter, boxplots, and area-by-cluster charts
11. **spatial_maps.py**  → map best labels (multi-method)
12. **spatial_mapping.py** → map one method+k to polygons
13. **similarity_index.py** → compare two clustering algorithms outputs 
## Outputs

1. **Datasets**
  - main_df.csv, prepared_df.parquet
  - data_scaled.npy, z_mean.npy
2. **Clustering Results**
  - main_df_with_{method}_k{k}.csv
  - labels_{method}_k{k}.csv
3. **Visualizations**
  - latent_{method}_k{k}.png
  - boxplot_scaled_vars_{method}_k{k}.png
  - area_by_cluster_{method}_k{k}.png
4. **Spatial Products**
  - map_{method}_k{k}.png
  - MO_30cm_clusters_{method}_k{k}.gpkg
  - MO_30cm_clusters_{method}_k{k}.shp

## Basic Usage
Full Pipeline (click-to-run)
python sha_pipeline_runbook.py

## Project Structure

```text
soil-analysis-pipeline/
├── README.md
├── requirements.txt
├── main.py
├── data_preparation.py
├── vae_training.py
├── clustering_evaluation.py
├── clustering_selection.py
├── clustering_algorithms.py
├── metric_plots.py
├── latent_plots.py
├── visualization.py
├── spatial_maps.py
├── spatial_mapping.py
├── geographic_processing.py
├── horizon_processing.py
├── restriction_processing.py
├── utils.py
├── pipeline_runbook.py
├── data/
│   ├── mupoly.shp
│   ├── mapunit.csv
│   ├── component.csv
│   ├── muagg.csv
│   ├── corestrictions.csv
│   ├── chorizon.csv
│   ├── cfrag.csv
│   └── aggResult/   # outputs
── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──                                                                                                    │
│   clustering_selection.py  --->  clustering_evaluation.py  --->  clustering_algorithms.py         │
│   (CLI + Input/Output)           (metrics, model selection)    (model factory + fit/predict)      │
│                                                                                                   │
── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──                                                                                    
## Contributing
We welcome improvements! Please:

## License
MIT License
Copyright (c) 2025 [Dipal Shah / Center for regenerative agriculture, University of Missouri]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation
If you use this software in your research, please cite:

## Support

- Create an issue in the project repository
- Contact developer: dipalshah@missouri.edu

## Acknowledgments
This project builds upon:
- SSURGO database from USDA-NRCS
- Scikit-learn machine learning library
- GeoPandas spatial analysis tools
- TensorFlow/Keras for VAE implementation
- Article: ```text
├── Algorithms for Quantitative Pedology: A Toolkit for Soil Scientists.
├── A regional soil classification framework to improve soil health diagnosis and management.

