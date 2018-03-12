# -*- coding: utf-8 -*-
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'

from farneback3d._farneback3d import Farneback, warp_by_flow
