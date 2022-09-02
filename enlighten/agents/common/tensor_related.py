import glob
import numbers
import os
import re
import shutil
import tarfile
from collections import defaultdict
from io import BytesIO
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import attr
import numpy as np
import torch
from gym.spaces import Box
from PIL import Image
from torch import Size, Tensor
from torch import nn as nn

from habitat import logger

from enlighten.agents.common.tensor_dict import DictTree, TensorDict
from enlighten.utils.tensorboard_utils import TensorboardWriter

from habitat_sim.utils import profiling_utils

from enlighten.utils.image_utils import try_cv2_import
cv2 = try_cv2_import()

@attr.s(auto_attribs=True, slots=True)
class ObservationBatchingCache:
    r"""Helper for batching observations that maintains a cpu-side tensor
    that is the right size and is pinned to cuda memory
    """
    _pool: Dict[Any, Union[torch.Tensor, np.ndarray]] = attr.Factory(dict)

    def get(
        self,
        num_obs: int,
        sensor_name: str,
        sensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        r"""Returns a tensor of the right size to batch num_obs observations together

        If sensor is a cpu-side tensor and device is a cuda device the batched tensor will
        be pinned to cuda memory.  If sensor is a cuda tensor, the batched tensor will also be
        a cuda tensor
        """
        key = (
            num_obs,
            sensor_name,
            tuple(sensor.size()),
            sensor.type(),
            sensor.device.type,
            sensor.device.index,
        )
        if key in self._pool:
            return self._pool[key]

        cache = torch.empty(
            num_obs, *sensor.size(), dtype=sensor.dtype, device=sensor.device
        )
        if (
            device is not None
            and device.type == "cuda"
            and cache.device.type == "cpu"
        ):
            # Pytorch indexing is slow,
            # so convert to numpy
            cache = cache.pin_memory().numpy()

        self._pool[key] = cache
        return cache

# profiling decorator: https://github.com/facebookresearch/habitat-sim/blob/main/habitat_sim/utils/profiling_utils.py
# to capture the time spent in annotated ranges
# this function could be time consuming
# organize observations to sensor tensors
# e.g. batch["pointgoal"] is a [N,2] tensor where N is the number of envs

@profiling_utils.RangeContext("batch_obs")
@torch.no_grad()
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
    cache: Optional[ObservationBatchingCache] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None
        cache: An ObservationBatchingCache.  This enables faster
            stacking of observations and cpu-gpu transfer as it
            maintains a correctly sized tensor for the batched
            observations that is pinned to cuda memory.

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch_t: TensorDict = TensorDict()
    if cache is None:
        batch: DefaultDict[str, List] = defaultdict(list)

    obs = observations[0]
    # Order sensors by size, stack and move the largest first
    sensor_names = sorted(
        obs.keys(),
        key=lambda name: 1
        if isinstance(obs[name], numbers.Number)
        else np.prod(obs[name].shape),
        reverse=True,
    )

    for sensor_name in sensor_names:
        for i, obs in enumerate(observations):
            sensor = obs[sensor_name]
            if cache is None:
                batch[sensor_name].append(torch.as_tensor(sensor))
            else:
                if sensor_name not in batch_t:
                    batch_t[sensor_name] = cache.get(
                        len(observations),
                        sensor_name,
                        torch.as_tensor(sensor),
                        device,
                    )

                # Use isinstance(sensor, np.ndarray) here instead of
                # np.asarray as this is quickier for the more common
                # path of sensor being an np.ndarray
                # np.asarray is ~3x slower than checking
                if isinstance(sensor, np.ndarray):
                    batch_t[sensor_name][i] = sensor
                elif torch.is_tensor(sensor):
                    batch_t[sensor_name][i].copy_(sensor, non_blocking=True)
                # If the sensor wasn't a tensor, then it's some CPU side data
                # so use a numpy array
                else:
                    batch_t[sensor_name][i] = np.asarray(sensor)

        # With the batching cache, we use pinned mem
        # so we can start the move to the GPU async
        # and continue stacking other things with it
        if cache is not None:
            # If we were using a numpy array to do indexing and copying,
            # convert back to torch tensor
            # We know that batch_t[sensor_name] is either an np.ndarray
            # or a torch.Tensor, so this is faster than torch.as_tensor
            if isinstance(batch_t[sensor_name], np.ndarray):
                batch_t[sensor_name] = torch.from_numpy(batch_t[sensor_name])

            batch_t[sensor_name] = batch_t[sensor_name].to(
                device, non_blocking=True
            )

    if cache is None:
        for sensor in batch:
            batch_t[sensor] = torch.stack(batch[sensor], dim=0)

        batch_t.map_in_place(lambda v: v.to(device))

    return batch_t