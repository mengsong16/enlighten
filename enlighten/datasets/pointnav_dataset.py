#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import List, Optional
from tqdm import tqdm


from enlighten.datasets.dataset import ALL_SCENES_MASK, Dataset, not_none_validator, Episode, EpisodeIterator
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import attr
from habitat import logger
import copy

CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"

@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None

@attr.s(auto_attribs=True)
class ShortestPathPoint:
    position: List[Any]
    rotation: List[Any]
    action: Optional[int] = None

@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None


class PointNavDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config) -> bool:
        return os.path.exists(
            config.get("dataset_path").format(split=config.get("split"))
        ) and os.path.exists(config.get("scenes_dir"))

    def has_individual_scene_files(self, config):
        datasetfile_path = config.get("dataset_path").format(split=config.get("split"))
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.get("scenes_dir"))

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        # self.content_scenes_path: {data_path}/content/{scene}.json.gz
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        
        return has_individual_scene_files, dataset_dir
    
    def get_scene_names(self, config, dataset_dir):
        # get scene names from the list of content_scenes
        scenes = config.get("content_scenes")
        
        # if *, get all scene names
        if ALL_SCENES_MASK in scenes:
            scenes = self._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path,
                dataset_dir=dataset_dir,
            )

        return scenes

    def get_scene_names_to_load(self, config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        assert self.check_config_paths_exist(config)

        has_individual_scene_files, dataset_dir = self.has_individual_scene_files(config)
        # if each scene has an individual file, load the specific scenes or all scene names from the folder
        if has_individual_scene_files:
            return self.get_scene_names(config, dataset_dir)
        else:
            print("Not implemented if scenes do not have individual scene files")
            


    # get all scene names under the folder
    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []
        
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes


    def __init__(self, config = None) -> None:
        self.episodes = []

        if config is None:
            return
        
        has_individual_scene_files, dataset_dir = self.has_individual_scene_files(config)
        # if each scene has a separate file
        
        if has_individual_scene_files:
            scenes = self.get_scene_names(config, dataset_dir)
            
            print("Loaded scenes: %d"%len(scenes))
            print("Start loading episodes ...")
            # load episodes from each scene
            self.scene_episode_num = {}
            for scene in tqdm(scenes):
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    n_episode = self.from_json(f.read(), scenes_dir=config.get("scenes_dir"))
                self.scene_episode_num[scene] = n_episode
                
            for key, value in self.scene_episode_num.items():
                print("%s: %d"%(key, value))

            print("Total loaded episodes: %d"%(len(self.episodes)))        
        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )
            

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        n_episode = 0
        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            
            n_episode += 1
            self.episodes.append(episode)
        
        return n_episode

# make a dataset PointNavDatasetV1
def make_dataset(id_dataset, **kwargs):
    logger.info("Initializing dataset: %s"%(id_dataset))
    if id_dataset == "PointNav":
        _dataset = PointNavDatasetV1()
    assert _dataset is not None, "Could not find dataset %s"%(id_dataset)

    return _dataset(**kwargs)  # type: ignore