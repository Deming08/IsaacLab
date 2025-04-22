# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

OBSTACLE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(2, 2),
    border_width=20.0,
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={

        "boxes_big": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.3,
            box_height_range=(0.25, 0.25),
            platform_width=0.25,
        ),
        "boxes_small": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.3,
            box_height_range=(0.5, 0.5),
            platform_width=0.1,
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,
        ),

    },
)
"""Rough terrains configuration."""
