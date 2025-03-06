# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .h1.h1 import H1
from .h1.h1_phc import H1 as H1_phc
from .h1.h1_s2r import H1 as H1_s2r
from .h1.h1_loco import H1Walk
from .h1.h1_lctk import H1Walk as H1LocoTrack
from .h1.h1_hp import H1 as H1HP
from .h1.h1_mc import H1 as H1MC
# from .h1.h1_mimic import H1 as H1MC
from .h1.h1_h2o import H1Walk as H1H2O
from .h1.h1_mcsr import H1Walk as H1MCSR

from .h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from .h1.h1_xh_config import H1RoughCfg as H1RoughCfgXH
from .h1.h1_xh_config import H1RoughCfgPPO as H1RoughCfgPPOXH
from .h1.h1_jaylon_config import H1RoughCfg as H1RoughCfgJaylon
from .h1.h1_jaylon_config import H1RoughCfgPPO as H1RoughCfgPPOJaylon 
from .h1.h1_track_config import H1RoughCfg as H1RoughCfgTrack
from .h1.h1_track_config import H1RoughCfgPPO as H1RoughCfgPPOTrack 
from .h1.h1_phc_config import H1RoughCfg as H1RoughCfgphc
from .h1.h1_phc_config import H1RoughCfgPPO as H1RoughCfgPPOphc 
from .h1.h1_s2r_config import H1RoughCfg as H1RoughCfgs2r, H1RoughCfgPPO as H1RoughCfgPPOs2r
from .h1.h1_loco_config import H1RoughCfg as H1WalkCfg, H1RoughCfgPPO as H1WalkCfgPPO
from .h1.h1_lctk_config import H1RoughCfg as H1LocoTrackCfg, H1RoughCfgPPO as H1LocoTrackCfgPPO
from .h1.h1_hp_config import H1RoughCfg as H1HPCfg, H1RoughCfgPPO as H1HPCfgPPO
from .h1.h1_mc_config import H1RoughCfg as H1MCCfg, H1RoughCfgPPO as H1MCCfgPPO 
# from .h1.h1_mimic_config import H1RoughCfg as H1MCCfg, H1RoughCfgPPO as H1MCCfgPPO 
from .h1.h1_h2o_config import H1RoughCfg as H1H2OCfg, H1RoughCfgPPO as H1H2OCfgPPO 
from .h1.h1_mcsr_config import H1RoughCfg as H1MCSRCfg, H1RoughCfgPPO as H1MCSRCfgPPO 
import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1", H1, H1RoughCfg(), H1RoughCfgPPO(), 'h1')
task_registry.register( "h1_xh", H1, H1RoughCfgXH(), H1RoughCfgPPOXH(), 'h1_xh')
task_registry.register( "h1_jaylon", H1, H1RoughCfgJaylon(), H1RoughCfgPPOJaylon(), 'h1_jaylon')
task_registry.register( "h1_track", H1, H1RoughCfgTrack(), H1RoughCfgPPOTrack(), 'h1_track') # init from h1_jaylon
task_registry.register( "h1_phc", H1_phc, H1RoughCfgphc(), H1RoughCfgPPOphc(), 'h1_phc') 
task_registry.register( "h1_s2r", H1_s2r, H1RoughCfgs2r(), H1RoughCfgPPOs2r(), 'h1_s2r') 
task_registry.register( "h1_loco", H1Walk, H1WalkCfg(), H1WalkCfgPPO(), 'h1_loco') 
task_registry.register( "h1_lctk", H1LocoTrack, H1LocoTrackCfg(), H1LocoTrackCfgPPO(), 'h1_lctk') 
task_registry.register( "h1_hp", H1HP, H1HPCfg(), H1HPCfgPPO(), 'h1_hp') 
task_registry.register( "h1_mc", H1MC, H1MCCfg(), H1MCCfgPPO(), 'h1_mc') 
task_registry.register( "h1_h2o", H1H2O, H1H2OCfg(), H1H2OCfgPPO(), 'h1_h2o') 
task_registry.register( "h1_mcsr", H1MCSR, H1MCSRCfg(), H1MCSRCfgPPO(), 'h1_mcsr') 
