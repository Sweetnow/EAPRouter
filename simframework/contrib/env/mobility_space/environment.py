"""Simulator: Urban Simulator"""

import asyncio
import os
from datetime import datetime
from subprocess import Popen
from typing import Any, List, Literal, Optional, Tuple, Union

import aiohttp
import shapely
from simframework.contrib.env.mobility_space.download_sim import download_binary
from simframework.contrib.env.mobility_space.map import Map
from simframework.contrib.env.mobility_space.utils import (
    find_free_ports,
    wait_for_port,
)
from simframework.env import (
    EnvBase,
    tool,
)
from simframework.logger import get_logger

from .replay_tables import AGENT_POSITION_SCHEMA
from pycityproto.city.geo.v2 import geo_pb2 as geo_pb2
from pycityproto.city.map.v2 import map_pb2 as map_pb2
from pycityproto.city.trip.v2.trip_pb2 import TripMode
from pydantic import BaseModel, ConfigDict, Field
from shapely import wkt
from shapely.geometry import LineString

__all__ = [
    "MobilitySpace",
    "MobilityPerson",
    "MobilityPersonInit",
    "Position",
    "Target",
    "Poi",
]


POI_START_ID = 7_0000_0000
DRIVING_SPEED_RATIO = 0.8  # the speed of driving is 80% of the max speed of the road
WALKING_SPEED = 1.34  # the speed of walking is 1.34 m/s
DRIVING_SPEED = 8.0  # the speed of driving is 8.0 m/s (approx 28.8 km/h)


class PositionInit(BaseModel):
    aoi_id: int = Field(
        ..., description="AOI ID, which is a continuous integer starting from 500000000"
    )
    poi_id: Optional[int] = Field(
        None,
        description="POI ID, which is a continuous integer starting from 700000000 (optional when initializing a person)",
    )


class Position(BaseModel):
    kind: Literal["aoi", "lane"] = Field(
        ..., description="Position kind: 'aoi' or 'lane'"
    )
    aoi_id: Optional[int] = Field(
        None,
        description="AOI ID, which is a continuous integer starting from 500000000",
    )
    poi_id: Optional[int] = Field(
        None,
        description="POI ID, which is a continuous integer starting from 700000000 (optional when initializing a person)",
    )
    xy: Tuple[float, float] = Field(..., description="XY coordinates of the position")
    lnglat: Tuple[float, float] = Field(
        ..., description="Lnglat coordinates of the position"
    )


class MobilityPersonInit(BaseModel):
    """Simplified model for initializing MobilityPerson, containing only id and position."""

    id: int = Field(..., description="Person ID")
    position: PositionInit = Field(..., description="The position of the person.")


class Target(BaseModel):
    position: Position = Field(..., description="The target position of the person.")
    mode: Literal["walking", "driving"] = Field(
        ..., description="The mode of the person."
    )
    path: shapely.LineString = Field(..., description="The path of the person.")
    path_s: float = Field(
        ..., description="The s coordinate of the person on the path."
    )
    path_v: float = Field(..., description="The speed of the person on the path.")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MobilityPerson(BaseModel):
    id: int = Field(..., description="Person ID")
    status: Literal["idle", "moving"] = Field(
        "idle",
        description='Person status. Default: "idle". If person is moving, it should be "moving".',
    )
    position: Position = Field(..., description="The position of the person.")
    target: Optional[Target] = Field(None, description="The target of the person.")

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Response models for tool functions
class TargetResponse(BaseModel):
    """Response model for target information (without internal path details)"""

    position: Position = Field(..., description="The target position of the person")
    mode: Literal["walking", "driving"] = Field(
        ..., description="The mode of the person"
    )


class GetPersonResponse(BaseModel):
    """Response model for get_person() function"""

    id: int = Field(..., description="Person ID")
    status: Literal["idle", "moving"] = Field(..., description="Person status")
    position: Position = Field(..., description="The position of the person")
    target: Optional[TargetResponse] = Field(
        None, description="The target of the person"
    )


class Poi(BaseModel):
    """Point of Interest (POI) model"""

    id: int = Field(..., description="POI ID")
    name: str = Field(..., description="POI name")
    category: str = Field(..., description="POI category")
    position: dict = Field(..., description="POI position (x, y coordinates)")
    distance: Optional[float] = Field(
        None, description="Distance from search center (only for find_nearby_pois)"
    )


class FindNearbyPoisResponse(BaseModel):
    """Response model for find_nearby_pois() function"""

    pois: List[Poi] = Field(..., description="List of POIs found")


class MobilitySpace(EnvBase):
    """
    The environment, including map data, simulator clients, and environment variables.
    """

    TRIPMODE2STR = {
        TripMode.TRIP_MODE_WALK_ONLY: "walking",
        TripMode.TRIP_MODE_DRIVE_ONLY: "driving",
    }

    def __init__(
        self,
        file_path: str,
        home_dir: str,
        persons: List[MobilityPersonInit] | List[dict],
        poi_search_limit: int = 10,
    ):
        """
        Initialize the Environment.

        Args:
            file_path: The path to the map file.
            home_dir: The home directory of the environment.
            persons: The persons to initialize the environment with. Can be a list of MobilityPersonInit objects or dicts.
            poi_search_limit: The limit of POIs to search.
        """
        super().__init__()

        # Expand ~ in paths
        file_path = os.path.expanduser(file_path)
        home_dir = os.path.expanduser(home_dir)

        os.makedirs(home_dir, exist_ok=True)
        self._routing_bin_path = download_binary(home_dir)
        self._file_path = file_path
        self._home_dir = home_dir
        self._map = Map(file_path)
        # type annotation
        self._routing_proc: Optional[Popen] = None

        self.poi_id_2_aoi_id: dict[int, int] = {
            poi["id"]: poi["aoi_id"] for poi in self._map._poi_list
        }
        self._poi_search_limit = poi_search_limit
        """limit of POIs to search"""

        self._lock = asyncio.Lock()
        """lock for routing process"""

        # 位置跟踪存储（用于benchmark数据收集）
        self._person_trajectories: dict[int, list[Tuple[float, float]]] = {
            p["id"] if isinstance(p, dict) else p.id: [] for p in persons
        }
        """Store trajectory points for each person"""

        self._person_visited_aois: dict[int, set[int]] = {
            p["id"] if isinstance(p, dict) else p.id: set() for p in persons
        }
        """Store visited AOIs for each person"""

        # data: convert MobilityPersonInit or dict to MobilityPerson
        person_objects = []
        for p in persons:
            if isinstance(p, dict):
                # Convert MobilityPersonInit dict to MobilityPerson
                person_init = MobilityPersonInit.model_validate(p)
            else:
                person_init = p
            # convert PositionInit to Position
            if person_init.position.poi_id is not None:
                poi = self._map.get_poi(person_init.position.poi_id)
                assert poi is not None
                x, y = poi["position"]["x"], poi["position"]["y"]
            else:
                aoi = self._map.get_aoi(person_init.position.aoi_id)
                assert aoi is not None
                x, y = aoi["shapely_xy"].centroid.x, aoi["shapely_xy"].centroid.y
            position = Position(
                kind="aoi",
                aoi_id=person_init.position.aoi_id,
                poi_id=person_init.position.poi_id,
                xy=(x, y),
                lnglat=self._map.projector(x, y, inverse=True),
            )
            person_objects.append(
                MobilityPerson(
                    id=person_init.id,
                    status="idle",
                    position=position,
                    target=None,
                )
            )

        self._persons: dict[int, MobilityPerson] = {p.id: p for p in person_objects}
        
        # Step counter for replay data
        self._step_counter: int = 0

    def get_aoi_as_position(self, aoi_id: int) -> Position:
        """
        Get the position of an AOI.
        """
        aoi = self._map.get_aoi(aoi_id)
        assert aoi is not None
        x, y = aoi["shapely_xy"].centroid.x, aoi["shapely_xy"].centroid.y
        return Position(
            kind="aoi",
            aoi_id=aoi_id,
            poi_id=None,
            xy=(x, y),
            lnglat=self._map.projector(x, y, inverse=True),
        )

    def get_poi_as_position(self, poi_id: int) -> Position:
        """
        Get the position of a POI.
        """
        poi = self._map.get_poi(poi_id)
        assert poi is not None
        x, y = poi["position"]["x"], poi["position"]["y"]
        return Position(
            kind="aoi",
            aoi_id=poi["aoi_id"],
            poi_id=poi_id,
            xy=(x, y),
            lnglat=self._map.projector(x, y, inverse=True),
        )

    # ============================================================================
    # Probe Functions for Trajectory and Position Tracking (for Benchmark)
    # ============================================================================

    def record_person_position(self, person_id: int):
        """
        Record the current position of a person to their trajectory.

        Args:
            person_id: The ID of the person
        """
        if person_id not in self._persons:
            return

        # Initialize if not already present（处理动态添加的person）
        if person_id not in self._person_trajectories:
            self._person_trajectories[person_id] = []
        if person_id not in self._person_visited_aois:
            self._person_visited_aois[person_id] = set()

        person = self._persons[person_id]

        # Record XY coordinates
        xy = person.position.xy
        if xy not in self._person_trajectories[person_id]:
            self._person_trajectories[person_id].append(xy)

        # Record AOI if at an AOI
        if person.position.aoi_id is not None:
            self._person_visited_aois[person_id].add(person.position.aoi_id)

    def get_person_trajectory(self, person_id: int) -> list[Tuple[float, float]]:
        """
        Get the recorded trajectory of a person.

        Args:
            person_id: The ID of the person

        Returns:
            List of (x, y) coordinates
        """
        return self._person_trajectories.get(person_id, [])

    def get_person_visited_aois(self, person_id: int) -> set[int]:
        """
        Get the set of AOIs visited by a person.

        Args:
            person_id: The ID of the person

        Returns:
            Set of AOI IDs
        """
        return self._person_visited_aois.get(person_id, set())

    def get_all_persons_trajectories(self) -> dict[int, list[Tuple[float, float]]]:
        """
        Get trajectories for all persons.

        Returns:
            Dict mapping person_id to list of (x, y) coordinates
        """
        return self._person_trajectories.copy()

    def get_all_persons_visited_aois(self) -> dict[int, set[int]]:
        """
        Get visited AOIs for all persons.

        Returns:
            Dict mapping person_id to set of AOI IDs
        """
        return {pid: aois.copy() for pid, aois in self._person_visited_aois.items()}

    @classmethod
    def mcp_description(cls) -> str:
        """
        Return a description text for MCP environment module candidate list.
        Includes parameter descriptions and JSON schemas for data models.
        """
        import json

        # Get JSON schemas for nested models
        person_init_schema = MobilityPersonInit.model_json_schema()

        description = f"""{cls.__name__}: Mobility management module for urban navigation and location tracking.

**Description:** {cls.__doc__ or 'No description available'}

**Initialization Parameters (excluding llm):**
- file_path (str): The path to the map file.
- home_dir (str): The home directory of the environment.
- persons (List[MobilityPersonInit] | List[dict]): List of persons to initialize the environment with. Can be MobilityPersonInit objects or dicts matching the schema.
- poi_search_limit (int, optional): The limit of POIs to search. Default: 10.

**MobilityPersonInit JSON Schema:**
```json
{json.dumps(person_init_schema, indent=2)}
```

**Example initialization config:**
```json
{{
  "file_path": "/path/to/map.pb",
  "home_dir": "/path/to/home",
  "persons": [
    {{
      "id": 1,
      "position": {{
        "aoi_id": 500000000
      }}
    }}
  ],
  "poi_search_limit": 10
}}
```
"""
        return description

    @property
    def description(self):
        """Description of the environment module for router selection and function calling"""
        return """Mobility management module for urban navigation and location tracking.

**Available Operations:**
1. Person Location & Status: Query current position, movement state, and trip progress by calling the `get_person` function
2. Move to: Plan and execute trips to POIs/AOIs with walking or driving modes by calling the `move_to` function.
3. Find Nearby POIs: Search nearby places by category within specified radius by calling the `find_nearby_pois` function
4. Get POI Details: Get information about specific POIs by calling the `get_poi` function

**Key Concepts:**
- AOI (Area of Interest): Large city zones like districts or neighborhoods (identified by aoi_id)
- POI (Point of Interest): Specific locations like shops, restaurants, landmarks (identified by poi_id, starts from 700000000)
- Categories: Two-level hierarchy (first-level: 'restaurant', 'outdoor_activity'; second-level: 'cafe', 'park')
- Travel Modes: 'walking' or 'driving'
- Coordinates: Longitude/latitude pairs for precise positioning

**Usage Tips:**
- Use category filters to narrow POI searches
"""

    async def init(self, start_datetime: datetime) -> Any:
        """
        Initialize the environment including the routing.
        """
        await super().init(start_datetime)
        # =========================
        # init syncer
        # =========================
        routing_port = find_free_ports()[0]
        self._server_addr = f"localhost:{routing_port}"
        self._routing_proc = Popen(
            [
                self._routing_bin_path,
                "-listen",
                self._server_addr,
                "-map",
                self._file_path,
                "-log-level",
                "warn",
            ],
            env=os.environ,
        )

        get_logger().info(
            f"start routing at {self._server_addr}, PID={self._routing_proc.pid}"
        )

        # Wait for the routing server to start listening
        get_logger().info(
            f"Waiting for routing server to start on port {routing_port}..."
        )
        if not wait_for_port("localhost", routing_port, timeout=30.0):
            # If the port is not available, kill the process and raise an error
            if self._routing_proc.poll() is None:
                self._routing_proc.kill()
            raise RuntimeError(
                f"Routing server failed to start on port {routing_port} within 30 seconds"
            )
        get_logger().info(f"Routing server is ready on {self._server_addr}")

        if self._replay_writer is not None:
            await self._replay_writer.register_table(AGENT_POSITION_SCHEMA)
            get_logger().info("Registered agent_position table for MobilitySpace")

    async def get_agent_position(self, agent_id: int) -> Tuple[Optional[float], Optional[float]]:
        """Get the position of an agent.

        Args:
            agent_id: The ID of the agent (person).

        Returns:
            A tuple of (longitude, latitude), or (None, None) if not available.
        """
        if agent_id not in self._persons:
            return None, None
        person = self._persons[agent_id]
        lng, lat = person.position.lnglat
        return lng, lat

    async def _write_position(
        self, agent_id: int, step: int, t: datetime, lng: float, lat: float
    ) -> None:
        """Write agent position to the agent_position table.

        Args:
            agent_id: The agent ID.
            step: The simulation step number.
            t: The simulation datetime.
            lng: Longitude coordinate.
            lat: Latitude coordinate.
        """
        if self._replay_writer is not None:
            await self._replay_writer.write("agent_position", {
                "id": agent_id,
                "step": step,
                "t": t,
                "lng": lng,
                "lat": lat,
            })


    @property
    def map(self):
        assert self._map is not None, "Map not initialized"
        return self._map

    @property
    def projector(self):
        return self._map.projector

    def _get_around_pois(
        self,
        center: tuple[float, float],
        radius: Optional[float] = None,
        poi_type: Optional[Union[str, list[str]]] = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get Points of Interest (POIs) around a central point based on type.

        - **Args**:
            - `center` (`Tuple[float, float]`): The central point as a tuple.
            - `radius` (`Optional[float]`): The search radius in meters. If not provided, all POIs are considered.
            - `poi_type` (`Optional[Union[str, List[str]]]`): The category or categories of POIs to filter by. Can be first-level categories, second-level categories, or None for all POIs.

        - **Returns**:
            - `List[Dict]`: A list of dictionaries containing information about the POIs found.
        """
        # If no poi_type specified, return all POIs
        if poi_type is None:
            assert self._map is not None
            all_pois: list[tuple[dict, float]] = self._map.query_pois(
                center=center,
                radius=radius,
            )
            pois = [{"poi": poi, "distance": distance} for poi, distance in all_pois]
            return pois[:limit]

        # Process poi_type input
        if isinstance(poi_type, str):
            poi_type = [poi_type]

        transformed_poi_type: list[str] = []
        for t in poi_type:
            if t in self._map.poi_cate:
                # This is a first-level category, expand to all subcategories
                transformed_poi_type += self._map.poi_cate[t]
            else:
                # This is either a second-level category or unknown category
                transformed_poi_type.append(t)

        poi_type_set = set(transformed_poi_type)

        # query pois within the radius
        assert self._map is not None
        nearest_pois: list[tuple[dict, float]] = self._map.query_pois(
            center=center,
            radius=radius,
        )

        # Filter POIs by category
        pois = []
        for poi, distance in nearest_pois:
            catg = poi["category"]
            # Check if any category in the POI matches our filter
            if any(c in poi_type_set for c in catg):
                pois.append({"poi": poi, "distance": distance})

        return pois[:limit]

    # ============================================================================
    # Mobility Management Functions for LLM Function Calling
    # ============================================================================

    @tool(readonly=True, kind="observe")
    async def get_person(self, person_id: int) -> GetPersonResponse:
        """
        Get the current location and status of a person, including position, movement state, and trip progress.

        Args:
            person_id: The ID of the person to query

        Returns:
            The context containing detailed location and movement information
        """
        if person_id not in self._persons:
            raise ValueError(f"Person {person_id} not found")

        person = self._persons[person_id]

        # Construct target response if target exists
        target_response = None
        if person.target is not None:
            target_response = TargetResponse(
                position=person.target.position,
                mode=person.target.mode,
            )

        return GetPersonResponse(
            id=person.id,
            status=person.status,
            position=person.position,
            target=target_response,
        )

    @tool(readonly=False)
    async def move_to(
        self,
        person_id: int,
        aoi_id_or_poi_id: int,
        mode: Literal["walking", "driving"] = "driving",
    ):
        """
        Plan and start a trip for a person to move to a specific AOI or POI.

        Args:
            person_id: The ID of the person to move
            aoi_id_or_poi_id: The target location ID (AOI ID or POI ID starting from 700000000)
            mode: The travel mode (walking or driving)

        Returns:
            Empty response indicating success
        """
        if person_id not in self._persons:
            raise ValueError(f"Person {person_id} not found")

        person = self._persons[person_id]

        # 1. choose mode

        # choose the start position
        if person.position.kind == "aoi":
            start_position = {
                "aoi_position": {
                    "aoi_id": person.position.aoi_id,
                }
            }
        else:
            # search for nearest road
            radius = 1000
            lanes = []
            while radius < 10000:
                lanes = self._map.query_lane(
                    person.position.xy, radius, 1 if mode == "driving" else 2
                )
                if len(lanes) > 0:
                    break
                radius *= 2
            if len(lanes) == 0:
                return
            lane, s, _ = lanes[0]
            start_position = {
                "lane_position": {
                    "lane_id": lane["id"],
                    "s": s,
                }
            }

        # Get destination position
        if aoi_id_or_poi_id >= POI_START_ID:
            poi = self._map.get_poi(aoi_id_or_poi_id)
            if poi is None:
                return
            destination_position = {
                "aoi_position": {
                    "aoi_id": poi["aoi_id"],
                }
            }
        else:
            aoi = self._map.get_aoi(aoi_id_or_poi_id)
            if aoi is None:
                return
            destination_position = {
                "aoi_position": {
                    "aoi_id": aoi_id_or_poi_id,
                }
            }

        # Call routing service to get route
        url = f"http://{self._server_addr}/city.routing.v2.RoutingService/GetRoute"
        request_data = {
            "type": (
                TripMode.TRIP_MODE_DRIVE_ONLY
                if mode == "driving"
                else TripMode.TRIP_MODE_WALK_ONLY
            ),
            "start": start_position,
            "end": destination_position,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data) as response:
                response_data = await response.json()
        eta = self._map.estimate_route_time(request_data, response_data)
        xys = self._map._route_to_xys(request_data, response_data)
        path = LineString(xys)
        path_v = path.length / eta if eta > 0 else 1

        # Update person state
        person.status = "moving"
        # Determine destination position with poi_id if applicable
        dest_aoi_id = destination_position["aoi_position"]["aoi_id"]
        dest_poi_id = aoi_id_or_poi_id if aoi_id_or_poi_id >= POI_START_ID else None
        if dest_poi_id is not None:
            # Use POI position
            poi = self._map.get_poi(dest_poi_id)
            assert poi is not None
            x, y = poi["position"]["x"], poi["position"]["y"]
            poi_xy = (x, y)
            poi_lnglat = self._map.projector(poi_xy[0], poi_xy[1], inverse=True)
            target_position = Position(
                kind="aoi",
                aoi_id=dest_aoi_id,
                poi_id=dest_poi_id,
                xy=poi_xy,
                lnglat=poi_lnglat,
            )
        else:
            # Use route end position (AOI centroid)
            target_position = Position(
                kind="aoi",
                aoi_id=dest_aoi_id,
                poi_id=None,
                xy=xys[-1],
                lnglat=self._map.projector(xys[-1][0], xys[-1][1], inverse=True),
            )

        person.target = Target(
            position=target_position,
            mode=mode,
            path=path,
            path_s=0.0,
            path_v=path_v,
        )
        xy = path.interpolate(0)
        person.position = Position(
            kind="lane",
            aoi_id=None,
            poi_id=None,
            xy=(xy.x, xy.y),
            lnglat=self._map.projector(xy.x, xy.y, inverse=True),
        )

        return

    @tool(readonly=True)
    async def find_nearby_pois(
        self, x: float, y: float, category: Optional[str], radius: float
    ) -> FindNearbyPoisResponse:
        """
        Discover Points of Interest (POIs) near a location, filtered by category and distance.

        Args:
            x: The longitude (x coordinate) of the search center
            y: The latitude (y coordinate) of the search center
            category: POI category filter - first-level (e.g., 'restaurant'), second-level (e.g., 'cafe'), or None for all categories.
                Available first-level categories (with examples):
                  - 'children_playground': playground, summer_camp, miniature_golf, dog_park
                  - 'cultural_and_artistic': arts_centre, brothel, casino, cinema, community_centre
                  - 'education_institution': college, dancing_school, driving_school, first_aid_school, kindergarten
                  - 'financial_service': atm, payment_terminal, bank, bureau_de_change, money_transfer
                  - 'indoor_entertainment': adult_gaming_centre, amusement_arcade, bowling_alley, disc_golf_course, escape_game
                  - 'medical_care': baby_hatch, clinic, dentist, doctors, hospital
                  - 'nature_and_wildlife_observation': bird_hide, nature_reserve, wildlife_hide, hunting_stand
                  - 'other_special_purpose': animal_boarding, animal_breeding, animal_shelter, animal_training, baking_oven
                  - 'outdoor_activity': bandstand, beach_resort, bird_hide, bleachers, firepit
                  - 'public_service': bbq, bench, dog_toilet, dressing_room, drinking_water
                  - 'restaurant': bar, biergarten, cafe, fast_food, food_court
                  - 'sports_facility': horse_riding, ice_rink, marina, pitch, sports_centre
                  - 'transportation_facility': bicycle_parking, bicycle_repair_station, bicycle_rental, bicycle_wash, boat_rental
                  - 'water_activity': beach_resort, ice_rink, marina, slipway, swimming_area
                You can also use any first-level category or second-level category directly (e.g., 'children_playground' 'cafe', 'park', 'hospital').
            radius: Search radius in meters (e.g., 1000 for 1km)

        Returns:
            Structured data containing POI list with IDs, names, categories, positions, and distances
        """
        pois = self._get_around_pois(
            center=(x, y),
            radius=radius,
            poi_type=category,
            limit=self._poi_search_limit,
        )
        clean_pois = []
        for p in pois:
            clean_poi = Poi(
                id=p["poi"]["id"],
                name=p["poi"]["name"],
                position=p["poi"]["position"],
                category=p["poi"]["category"][-1],
                distance=p["distance"],
            )
            clean_pois.append(clean_poi)

        return FindNearbyPoisResponse(pois=clean_pois)

    @tool(readonly=True)
    async def get_poi(self, poi_id: int) -> Poi:
        """
        Retrieve detailed information about a specific Point of Interest.

        Args:
            poi_id: The unique ID of the POI (starts from 700000000)

        Returns:
            Structured data containing POI details (ID, name, category, position)
        """
        poi = self._map.get_poi(poi_id)
        if poi is None:
            raise ValueError(f"POI {poi_id} not found")

        poi_obj = Poi(
            id=poi["id"],
            name=poi["name"],
            category=poi["category"][-1],
            position=poi["position"],
            distance=None,
        )

        return poi_obj

    async def close(self):
        """
        Terminate the simulation process if it's running.
        """
        if self._routing_proc is not None and self._routing_proc.poll() is None:
            get_logger().info(
                f"Terminating routing at {self._server_addr}, PID={self._routing_proc.pid}, please ignore the PANIC message"
            )
            self._routing_proc.kill()
            # wait for the process to terminate
            self._routing_proc.wait()
        self._routing_proc = None

    async def step(self, tick: int, t: datetime):
        """
        Run forward one step.

        Args:
            tick: The number of ticks (1 tick = 1 second) of this simulation step.
            t: The current datetime of the simulation after this step with the ticks.
        """
        # Update all moving persons
        for person in self._persons.values():
            if person.status == "moving":
                assert person.target is not None
                distance_to_move = person.target.path_v * tick
                person.target.path_s += distance_to_move
                if person.target.path_s >= person.target.path.length:
                    person.status = "idle"
                    person.position = person.target.position
                    person.target = None
                    # update position - use POI position if poi_id exists, otherwise use AOI centroid
                    assert person.position.aoi_id is not None
                    if person.position.poi_id is not None:
                        # Use POI position
                        poi = self._map.get_poi(person.position.poi_id)
                        assert poi is not None
                        x, y = poi["position"]["x"], poi["position"]["y"]
                        person.position.xy = (x, y)
                        person.position.lnglat = self._map.projector(x, y, inverse=True)
                    else:
                        # Use AOI centroid
                        aoi = self._map.get_aoi(person.position.aoi_id)
                        assert aoi is not None
                        x, y = (
                            aoi["shapely_xy"].centroid.x,
                            aoi["shapely_xy"].centroid.y,
                        )
                        person.position.xy = (x, y)
                        person.position.lnglat = self._map.projector(x, y, inverse=True)
                else:
                    # update position
                    xy = person.target.path.interpolate(person.target.path_s)
                    person.position.xy = (xy.x, xy.y)
                    person.position.lnglat = self._map.projector(
                        xy.x, xy.y, inverse=True
                    )

            # 【关键】每步记录person的位置（用于轨迹收集）
            self.record_person_position(person.id)
            
            # Write position to replay database
            lng, lat = person.position.lnglat
            await self._write_position(person.id, self._step_counter, t, lng, lat)

        self.t = t
        self._step_counter += 1

    # ==================== Replay Data Methods ====================
    # Note: get_agent_position and _write_position are defined earlier in the class

    def _dump_state(self) -> dict:
        """
        Dump the internal state of MobilitySpace for serialization.

        Returns:
            dict: A dictionary containing all necessary state information
        """
        # Serialize persons
        persons_data = {}
        for person_id, person in self._persons.items():
            person_dict = person.model_dump()
            # Convert shapely LineString to WKT (Well-Known Text) format if present
            if person.target is not None and person.target.path is not None:
                person_dict["target"]["path"] = person.target.path.wkt
            persons_data[person_id] = person_dict

        return {
            "file_path": self._file_path,
            "home_dir": self._home_dir,
            "poi_search_limit": self._poi_search_limit,
            "persons": persons_data,
        }

    def _load_state(self, state: dict):
        """
        Load the internal state of MobilitySpace from serialized data.

        Args:
            state: The state dictionary produced by _dump_state()
        """
        # Restore configuration parameters
        self._poi_search_limit = state.get("poi_search_limit", 10)

        # Restore persons
        persons_data = state.get("persons", {})
        self._persons = {}
        for person_id_str, person_dict in persons_data.items():
            person_id = int(person_id_str)
            # Convert WKT back to shapely LineString if present
            if (
                person_dict.get("target") is not None
                and person_dict["target"].get("path") is not None
            ):
                path_wkt = person_dict["target"]["path"]
                person_dict["target"]["path"] = wkt.loads(path_wkt)

            # Reconstruct MobilityPerson from dict
            person = MobilityPerson.model_validate(person_dict)
            self._persons[person_id] = person
