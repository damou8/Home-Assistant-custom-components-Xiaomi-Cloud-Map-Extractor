import base64
import json
import logging
import struct
import zlib
from dataclasses import dataclass
from typing import Self, Any

from miio.exceptions import DeviceException
from miio.miot_device import MiotDevice
from vacuum_map_parser_base.map_data import MapData
from vacuum_map_parser_xiaomi.aes_decryptor import gen_md5_key
from vacuum_map_parser_xiaomi.map_data_parser import XiaomiMapDataParser
from vacuum_map_parser_xiaomi.status_mapping import XiaomiVacuumStatusMapping, get_status_mapping

from .base.model import VacuumConfig, VacuumApi
from .base.vacuum_v2 import BaseXiaomiCloudVacuumV2
from ..utils.exceptions import FailedConnectionException

_LOGGER = logging.getLogger(__name__)
OFF_UPDATES = 3

# One trajectory point: type (unsigned byte), then x and y in mm (little-endian int32).
_TRAJECTORY_POINT = struct.Struct("<Bii")

@dataclass
class XiaomiVacuumPropertyMapping:
    """Dataclass containing mapping for map property"""

    # vacuum map service id
    siid: int = 10

    # current map property id in vacuum map service
    piid: int = 1

    # trajectory (cleaning path) object property id, for models that publish it apart from the map
    trajectory_piid: int | None = None

    # vacuum position property id, for models that publish it apart from the map
    position_piid: int | None = None

_NON_STANDARD_MAP_PROP = [
    (
        [
            "xiaomi.vacuum.b108gl",
        ],
        XiaomiVacuumPropertyMapping(siid=7, trajectory_piid=2, position_piid=4),
    ),
    (
        [
            "xiaomi.vacuum.b108gp",
            "xiaomi.vacuum.ov32gl",
            "xiaomi.vacuum.ov43gl",
            "xiaomi.vacuum.ov51",
            "xiaomi.vacuum.ov81",
        ],
        XiaomiVacuumPropertyMapping(siid=9),
    ),
    (
        [
            "xiaomi.vacuum.b106bk",
            "xiaomi.vacuum.b106tr",
            "xiaomi.vacuum.b112",
            "xiaomi.vacuum.b112bk",
            "xiaomi.vacuum.b112gl",
            "xiaomi.vacuum.b112tr",
            "xiaomi.vacuum.c101",
            "xiaomi.vacuum.c101eu",
            "xiaomi.vacuum.c102",
            "xiaomi.vacuum.c104",
            "xiaomi.vacuum.e101gl",
        ],
        XiaomiVacuumPropertyMapping(piid=2),
    ),
]

# The parser's default idle states include 4, which means "Sweeping" on these models,
# so the map would stop updating while the robot cleans.
_NON_STANDARD_STATUS_PROP = [
    (["xiaomi.vacuum.b108gl"], XiaomiVacuumStatusMapping(idle_at=(1, 2, 3, 5, 8, 10))),
]


def parse_trajectory(raw: bytes) -> list[dict[str, int]] | None:
    """Decode a trajectory object: base64-encoded zlib data made of 9-byte points."""
    try:
        data = zlib.decompress(base64.b64decode(raw))
    except (ValueError, zlib.error):
        return None
    if len(data) % _TRAJECTORY_POINT.size:
        return None
    return [{"x": x, "y": y} for _, x, y in _TRAJECTORY_POINT.iter_unpack(data)]


def parse_position(value: Any) -> dict[str, float] | None:
    """Decode a vacuum-position property: {"position": [x, y, yaw in milliradians]}."""
    try:
        x, y, yaw = json.loads(value)["position"]
    except (TypeError, ValueError, KeyError):
        return None
    return {"x": x, "y": y, "yaw": yaw / 1000}


class XiaomiCloudVacuum(BaseXiaomiCloudVacuumV2):
    def __init__(self, vacuum_config: VacuumConfig):
        super().__init__(vacuum_config)
        self._token = vacuum_config.token
        self._host = vacuum_config.host

        self._miot_device = MiotDevice(self._host, self._token, timeout=2)

        self._xiaomi_map_data_parser = XiaomiMapDataParser(
            vacuum_config.palette,
            vacuum_config.sizes,
            vacuum_config.drawables,
            vacuum_config.image_config,
            vacuum_config.texts
        )

        self._status_mapping = next(
            (mapping for models, mapping in _NON_STANDARD_STATUS_PROP if self.model in models),
            get_status_mapping(self.model),
        )
        self._off_counter = 0

        self._vacuum_map = next((mapping for models, mapping in _NON_STANDARD_MAP_PROP if self.model in models), XiaomiVacuumPropertyMapping())
        self._live_data: dict[str, Any] = {}

    @property
    def should_update_map(self: Self) -> bool:
        try:
            status_value = self._miot_device.get_property_by(self._status_mapping.siid,
                                                             self._status_mapping.piid)[0]["value"]

            if status_value in self._status_mapping.idle_at:
                self._off_counter += 1
                _LOGGER.debug(
                    "Vacuum is not moving. Off counter: %d", self._off_counter)
                return self._off_counter <= OFF_UPDATES
            else:
                self._off_counter = 0
                return True
        except DeviceException as de:
            if "token" not in repr(de):
                return False
            raise FailedConnectionException(de)

    @staticmethod
    def vacuum_platform() -> VacuumApi:
        return VacuumApi.XIAOMI

    @property
    def map_archive_extension(self) -> str:
        return "zlib.enc"

    @property
    def map_data_parser(self) -> XiaomiMapDataParser:
        return self._xiaomi_map_data_parser
    
    def _get_object_name(self: Self, piid: int) -> str | None:
        response = self._miot_device.get_property_by(self._vacuum_map.siid, piid)[0].get("value")

        if response is None:
            return None

        if isinstance(response, int):
            return str(response)
        else:
            map_name = None
            try:
                map_name = json.loads(response).get("obj_name", None)
            except json.JSONDecodeError:
                if isinstance(response, str) and "/" in response:
                    map_name = response
            if map_name is None:
                return None
            return map_name.split("/")[-1]

    async def get_map_name(self: Self) -> str:
        return self._get_object_name(self._vacuum_map.piid) or await super().get_map_name()

    async def get_map(self: Self) -> tuple[MapData, bytes]:
        self._live_data = await self._get_live_data()
        return await super().get_map()

    async def _get_live_data(self: Self) -> dict[str, Any]:
        """Get the path and the robot position, which some models publish apart from the map."""
        live_data: dict[str, Any] = {}
        try:
            if self._vacuum_map.trajectory_piid is not None:
                trajectory_name = self._get_object_name(self._vacuum_map.trajectory_piid)
                raw_trajectory = await self.get_raw_map_data(trajectory_name)
                if raw_trajectory is not None and (paths := parse_trajectory(raw_trajectory)) is not None:
                    live_data["paths"] = paths
                else:
                    _LOGGER.debug("No usable trajectory in object \"%s\".", trajectory_name)
            if self._vacuum_map.position_piid is not None:
                value = self._miot_device.get_property_by(self._vacuum_map.siid,
                                                          self._vacuum_map.position_piid)[0].get("value")
                if (position := parse_position(value)) is not None:
                    live_data["position"] = position
        except DeviceException as de:
            _LOGGER.debug("Failed to get the path or the position: %s", de)
        return live_data

    async def get_map_url(self, map_name: str) -> str | None:
        return await self.get_fallback_map_url(map_name)

    def decode_and_parse(self, raw_map: bytes) -> MapData:
        # Try parsing as JSON first (old format), otherwise use raw data directly (new format)
        try:
            raw_map = base64.decodebytes(json.loads(raw_map)["data"].encode("latin1"))
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            # Data may not be JSON-wrapped
            pass
        
        raw_map = raw_map.hex()
        decoded_map = self.map_data_parser.unpack_map(
            raw_map,
            model=self.model.replace("xiaomi", "mi"),
            device_id=str(self._device_id),
        )
        if self._live_data:
            decoded_map = {**json.loads(decoded_map), **self._live_data}
        return self.map_data_parser.parse(decoded_map)
    
    def additional_data(self: Self) -> dict[str, Any]:
        super_data = super().additional_data()
        enc_key = gen_md5_key(
            self.model.replace("xiaomi", "mi"),
            str(self._device_id),
        )

        return {**super_data, "enc_key": enc_key}
