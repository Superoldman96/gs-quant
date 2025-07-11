"""
Copyright 2019 Goldman Sachs.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import datetime as dt
from typing import Union

import pandas as pd

from gs_quant.data import DataCoordinate


class DataSeries:
    """Represents a data series update"""

    def __init__(self, series: pd.Series, coordinate: DataCoordinate):
        self.series = series
        self.coordinate = coordinate


class DataEvent:
    """Represents a data update event"""

    def __init__(self, time: dt.datetime, value: Union[None, str, float], coordinate: DataCoordinate = None):
        self.time = time
        self.value = value
        self.coordinate = coordinate
