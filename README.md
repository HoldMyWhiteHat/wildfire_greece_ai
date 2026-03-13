Wildfire Risk Prediction & Visualization Platform (Greece)

An end-to-end machine learning and geospatial visualization system for predicting and exploring wildfire risk across Greece using meteorological data and spatial relationships.

The platform combines deep learning (LSTM + Graph Neural Networks) with interactive geospatial visualization (Folium) and is served through a FastAPI backend for easy exploration by researchers.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Research Motivation

In Mediterranean ecosystems, wildfires are one of the most dangerous natural disasters. This is especially true in Greece, where hot, dry summers and strong winds create highly combustible conditions. Most traditional wildfire monitoring systems are reactive, which means they only work after the fire has started and satellites have found it. But proactive fire management needs predictive risk modelling that finds areas with a higher chance of fire before it starts.

The risk of wildfires is complicated because it depends on how the environment changes over time and how different areas interact with each other. Meteorological variables such as temperature, wind speed, and humidity over time can make plants drier, increasing the availability of fuel and the likelihood of wildfire ignition and spread. At the same time, spatial factors like terrain, vegetation continuity, and local climate patterns make areas with a higher risk of fire that are close to each other.

Most classical statistical models treat geographic regions independently and rely on simple regression techniques. While these models capture general relationships between weather conditions and fire probability, they fail to represent two critical aspects of wildfire dynamics:

1.Temporal dependencies in environmental variables

2.Spatial relationships between neighboring geographic regions

To address these limitations, this project introduces a hybrid deep learning architecture combining Long Short-Term Memory networks (LSTM) with Graph Neural Networks (GNN). This architecture is designed to simultaneously model the temporal evolution of meteorological variables and the spatial structure of wildfire risk across geographic regions.

The resulting system provides a data-driven wildfire risk prediction framework that can support researchers in identifying high-risk areas and improving fire preparedness strategies.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Why Temporal Modeling with LSTM Networks

Cumulative weather patterns, like long periods of drought or high temperatures, have a big effect on the risk of wildfires. To capture these kinds of patterns, models need to be able to learn long-term temporal dependencies.

Long Short-Term Memory (LSTM) networks were selected for this task because they are specifically designed to model sequential data with long-range dependencies. Unlike traditional recurrent neural networks, LSTM architectures include gating mechanisms that regulate information flow through time, enabling the network to retain relevant historical information while discarding irrelevant signals.

In this system, the LSTM model processes sequences of meteorological observations for each geographic grid cell. The model learns patterns such as:

1.Prolonged heat

2.Low humidity

3.Reduced precipitation

4.Elevated vegetation dryness

The output of the LSTM layer is a temporal representation of wildfire risk dynamics for each spatial location.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Why Spatial Modeling with Graph Neural Networks

Temporal models show how weather changes over time, but wildfire risk is also spatial by nature. Plants, landforms, and weather patterns in nearby areas are often very similar. Fire spread can also move into nearby areas, making risk zones that are spatially linked.

To incorporate these spatial relationships, the geographic grid is represented as a graph structure. In this graph:

1.Each grid cell is represented as a node

2.Edges connect spatially adjacent cells

This representation allows the model to learn interactions between neighboring regions.

Graph Neural Networks (GNNs) operate on such graph structures by iteratively aggregating information from neighboring nodes. During each propagation step, nodes update their representations based on the features of their neighbors. This process enables the network to capture spatial dependencies and regional clustering effects.

By applying a GNN layer to the temporal representations produced by the LSTM model, the system learns how wildfire risk propagates across geographic space.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Hybrid LSTM–GNN Architecture

The final predictive architecture combines the strengths of both models.

First, the LSTM network processes temporal weather sequences to extract features describing environmental conditions. These temporal features are then passed to the Graph Neural Network, which models spatial dependencies between geographic regions.

The hybrid architecture can be summarized as:

Temporal Weather Sequences
-> LSTM Temporal Encoder
-> Graph Neural Network Spatial Aggregation
-> Wildfire Risk Prediction

This approach enables the system to simultaneously capture:

1.Long-term meteorological patterns

2.Spatial correlations across geographic regions

3.Potential wildfire propagation dynamics

Compared to purely temporal or purely spatial models, the hybrid architecture provides a more comprehensive representation of wildfire risk dynamics.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Methodology

The proposed system follows a multi-stage pipeline that integrates climate data processing, temporal modeling, spatial modeling, and geospatial visualization.
Data Source:
    This project uses climate reanalysis data from the Copernicus Climate Change Service (C3S).
    ERA5 data provided by the European Centre for Medium-Range Weather Forecasts (ECMWF).

The overall workflow can be summarized as:

ERA5 Climate Data
→ Feature Engineering
→ Temporal Modeling (LSTM)
→ Spatial Modeling (Graph Neural Network)
→ Wildfire Risk Prediction
→ Geospatial Visualization

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Climate Data and Feature Engineering

The system relies on meteorological variables obtained from ERA5 as mentioned above. These datasets provide high-resolution atmospheric observations and reanalysis products.
The datasets i used,cover the summer months of 2023, 2024, and 2025.

Variables used in wildfire risk modeling include:

1.t2m (Temperature at 2m): The temperature of air at 2 meters above the surface of the land, sea, or in-land waters.

2.d2m (Dew point temperature at 2m): The temperature at which the air becomes saturated with water vapor and condensation begins, measured at 2 meters above the surface.

3.u10 (10m u-wind component): The zonal (eastward/westward) component of the wind speed at 10 meters above the surface. Positive values indicate eastward wind, while negative values indicate westward wind.

4.v10 (10m v-wind component): The meridional (northward/southward) component of the wind speed at 10 meters above the surface. Positive values indicate northward wind, while negative values indicate southward wind.

5.tp (Total precipitation): The accumulated amount of water that has fallen from the atmosphere to the surface. It includes both large-scale precipitation and convective precipitation. It is measured in meters (m) per timestep and converted to millimeters (mm).

These variables are aggregated into daily spatial grids covering Greece, where each grid cell represents a geographic region. The resulting dataset forms a spatio-temporal tensor where each location contains a sequence of meteorological observations at exactly 12:00PM.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Geospatial Visualization

Predictions are visualized as interactive geospatial maps using Folium.

Each map represents wildfire risk for a single day.

Features:

1.Color-coded risk levels

2.Zoomable map of Greece

3.Interactive inspection

Example color scale:

Risk Level          Color
Very Low            Green
Low                 Yellow
Medium(or Moderate) Orange
High                Red
Extreme(Very High)  Purple

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Web Interface

The platform provides a lightweight researcher interface served through FastAPI.

Available views:

1.Browse Maps:

    View wildfire risk maps for different days.

2.Compare Maps:

    Compare two different dates side-by-side to study temporal changes.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
API Endpoints

Endpoint            Description

/                   Home page

ui/browse           Browse wildfire maps

ui/compare          Compare two maps

/api/maps           List available map files

/maps/{file}        Serve HTML map

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Technologies Used

1.Python

2.FastAPI

3.Folium

4.Pandas

5.NumPy

6.PyTorch

7.Graph Neural Networks

8.LSTM Networks

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Deployment

The system is deployed using Render, which supports Python web services.

URL: https://wildfire-greece-ai-2.onrender.com/
*It may take several minutes to load.Render's free tier web services are designed for testing and personal projects, which introduces specific limitations, most notably the "cold start" delay.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Future Improvements & Limitations 

Possible extensions include:

1.Satellite imagery integration

2.Real-time weather ingestion

3.Advanced fire spread simulation

4.Reinforcement learning for fire suppression planning

5.Multi-country wildfire risk prediction

Although the proposed system captures important temporal and spatial patterns, several limitations remain:

1.Wildfire ignition events are influenced by human activities that are not captured by meteorological variables alone.

2.The spatial resolution of wildfire risk predictions is constrained by the resolution of the underlying climate dataset.Since meteorological variables are provided on a geographic grid, each prediction represents the average risk within a grid cell rather than a precise location.

3.The model does not explicitly simulate fire spread dynamics after ignition.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Summary

Wildfire risk prediction requires modeling complex interactions between weather patterns and geographic structures.

The hybrid LSTM–GNN architecture was selected because it provides a balanced approach capable of capturing:

1.Temporal environmental dynamics

2.Spatial relationships across geographic regions

This design allows the system to generate wildfire risk predictions that reflect both the history of environmental conditions and the spatial context of each region.

The resulting framework offers a flexible and extensible foundation for future research in wildfire risk prediction and environmental monitoring.