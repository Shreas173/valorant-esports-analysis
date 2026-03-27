# Valorant Esports Performance Analysis

This repository contains a comprehensive data analysis and machine learning project focused on Valorant esports performance metrics. The analysis covers over 700,000 player observations across multiple competitive events, regions, and skill tiers.

## Project Overview

This study presents a data-driven approach to understanding player performance in Valorant esports through:

- **Exploratory Data Analysis** of 702,367 player observations with 30+ performance metrics
- **Machine Learning Clustering** to identify distinct player archetypes
- **Predictive Modeling** using Random Forest classification (AUC: 0.9824)
- **Academic Research Paper** documenting findings and methodologies

## Key Findings

- **4 Player Archetypes Identified** via K-means clustering:
  - Support Players (69.3%)
  - Balanced Performers (23.0%)
  - Elite Players (5.7%)
  - Specialists (3.3%)

- **Top Performance Predictors** (by feature importance):
  1. Kills per Round (0.301)
  2. Average Combat Score (0.165)
  3. Average Damage per Round (0.140)

- **Model Performance**: Random Forest classifier achieves 98.24% AUC in identifying high-performing players

## Repository Structure

```
ValorantComp/
├── analysis/               # Data processing and analysis scripts
│   ├── advanced.py        # Advanced analytics and ML models
│   ├── eda.py            # Exploratory data analysis
│   ├── merge_and_clean.py # Data consolidation pipeline
│   ├── stream_merge.py    # Streaming data processor
│   ├── compute_top_*.py   # Analysis computation scripts
│   ├── figures_full/      # Generated visualizations
│   ├── advanced_outputs/  # ML model outputs and results
│   └── requirements.txt   # Python dependencies
├── bronze/                # Raw data (event/region/map/agent structure)
├── paper/                 # Research paper and documentation
│   ├── comprehensive_valorant_analysis.tex  # LaTeX source
│   ├── comprehensive_valorant_analysis.pdf  # Compiled PDF
│   └── figures/           # Paper figures
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shreas/valorant-esports-analysis.git
cd valorant-esports-analysis
```

2. Install dependencies:
```bash
cd analysis
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run Exploratory Data Analysis
```bash
python3 eda.py --input merged_full.csv --out_dir figures_full --report eda_full_report.txt
```

### Run Advanced Analytics
```bash
python3 advanced.py --input merged_full.csv --out_dir advanced_outputs
```

### Compile Research Paper
```bash
cd paper
pdflatex comprehensive_valorant_analysis.tex
```

## Dataset

The analysis uses data from ValorantTracker.gg covering:
- **702,367 player observations**
- **30+ performance metrics** including:
  - Combat statistics (ACS, KDR, KPR, headshot %)
  - Economic efficiency
  - Clutch performance
  - Agent-specific metrics
  - Regional and event data

## Technologies Used

- **Python 3.14+**
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models
- **Matplotlib/Seaborn** - Data visualization
- **PyArrow** - Efficient data processing
- **LaTeX** - Academic paper formatting

## Research Applications

This work contributes to:
- Esports performance evaluation methodologies
- Talent identification in competitive gaming
- Team composition optimization
- Academic research in gaming analytics

## Author

**Shreas Arion - BRAC University** - Data Science & Esports Analytics Research

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{arion2026valorant,
  title={Comprehensive Analysis of Valorant Esports Performance Metrics: A Data-Driven Approach to Player Classification and Performance Prediction},
  author={Arion, Shreas},
  year={2026}
}
```

## Acknowledgments

- Riot Games for creating Valorant
- The competitive Valorant community
- Tournament organizers and data providers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or collaboration opportunities, please open an issue in this repository.
