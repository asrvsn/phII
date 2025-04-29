# Analysis
Order of execution is generally important.

Extraction
```bash
python unpack_seg.py
python extract_polygons.py
python extract_offspring.py
```
Analysis

```bash
python summarize.py
python extract_sizes.py
python cell_voronoi.py
python project_compartments.py
python compute_spheroid_metrics.py
python compute_polygon_metrics.py
python compute_projected_metrics.py
```

# 3D surface extraction

```bash
python -m microseg.voxel.extract CZI_FILE -t
python -m microseg.surface.construct.ellipsoid VERTICES_FILE
```
Then check the projection using 
```bash
python -m microseg.surface.project.ellipsoid POLYGONS_FILE ELLIPSOID_FILE
```