import numpy as np
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Optional, Protocol
from enum import Enum
from abc import ABC, abstractmethod


class DescentPhase(Enum):
    COASTING = "coasting"
    SUICIDE_BURN = "suicide_burn"
    APPROACH = "approach"
    TERMINAL = "terminal"


class SimulationState(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    time: float
    fuel_mass: float
    attitude_quaternion: Optional[np.ndarray] = None


class ControlCommand(NamedTuple):
    thrust_vector: np.ndarray
    fuel_consumption_rate: float
    phase: DescentPhase
    is_constrained: bool = False
    constraint_info: Optional[str] = None


@dataclass
class DescentParameters:
    isp: float = 311.0
    g0: float = 9.80665
    max_thrust: float = 45000.0
    burn_margin: float = 1.2
    k_p: float = 0.8
    k_horiz: float = 0.6
    high_gate_altitude: float = 2134.0
    target_low_gate_altitude: float = 150.0
    target_low_gate_vspeed: float = -5.0
    target_descent_rate: float = -1.0


@dataclass
class PlanetaryParameters:
    mu: float
    radius: float

@dataclass
class LandingSiteParameters:
    latitude_rad: float = 0.67 * np.pi / 180.0  # Sea of Tranquility latitude (Apollo 11 site)
    longitude_rad: float = 23.47 * np.pi / 180.0  # Sea of Tranquility longitude


class ControlConstraint(ABC):
    @abstractmethod
    def apply_constraint(self, thrust_cmd: np.ndarray, state: SimulationState) -> Tuple[np.ndarray, bool, str]:
        pass


class FuelConstraint(ControlConstraint):
    def __init__(self, min_fuel_margin: float = 0.0):
        self.min_fuel_margin = min_fuel_margin
    
    def apply_constraint(self, thrust_cmd: np.ndarray, state: SimulationState) -> Tuple[np.ndarray, bool, str]:
        if state.fuel_mass <= self.min_fuel_margin:
            return np.zeros(3), True, f"Fuel depleted (remaining: {state.fuel_mass:.1f} kg)"
        return thrust_cmd, False, ""


class OrientationConstraint(ControlConstraint):
    def __init__(self, max_thrust_angle_deg: float = 45.0, thrust_direction_body: np.ndarray = np.array([0, 0, 1])):
        self.max_thrust_angle_rad = np.deg2rad(max_thrust_angle_deg)
        self.thrust_direction_body = thrust_direction_body / np.linalg.norm(thrust_direction_body)
    
    def apply_constraint(self, thrust_cmd: np.ndarray, state: SimulationState) -> Tuple[np.ndarray, bool, str]:
        if state.attitude_quaternion is None:
            return thrust_cmd, False, ""
        
        if np.linalg.norm(thrust_cmd) < 1e-6:
            return thrust_cmd, False, ""
        
        thrust_direction_inertial = thrust_cmd / np.linalg.norm(thrust_cmd)
        thrust_direction_body_current = self._rotate_vector_by_quaternion_inverse(thrust_direction_inertial, state.attitude_quaternion)
        
        angle = np.arccos(np.clip(np.dot(thrust_direction_body_current, self.thrust_direction_body), -1.0, 1.0))
        
        if angle > self.max_thrust_angle_rad:
            corrected_direction = self._project_to_cone(thrust_direction_body_current, self.thrust_direction_body, self.max_thrust_angle_rad)
            corrected_inertial = self._rotate_vector_by_quaternion(corrected_direction, state.attitude_quaternion)
            constrained_thrust = corrected_inertial * np.linalg.norm(thrust_cmd)
            return constrained_thrust, True, f"Thrust angle limited to {np.rad2deg(self.max_thrust_angle_rad):.1f}°"
        
        return thrust_cmd, False, ""
    
    def _rotate_vector_by_quaternion(self, vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        q = quaternion / np.linalg.norm(quaternion)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        rotation_matrix = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return rotation_matrix @ vector
    
    def _rotate_vector_by_quaternion_inverse(self, vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        q_inv = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
        return self._rotate_vector_by_quaternion(vector, q_inv)
    
    def _project_to_cone(self, vector: np.ndarray, cone_axis: np.ndarray, max_angle: float) -> np.ndarray:
        vector_norm = vector / np.linalg.norm(vector)
        proj_parallel = np.dot(vector_norm, cone_axis) * cone_axis
        proj_perp = vector_norm - proj_parallel
        
        if np.linalg.norm(proj_perp) < 1e-8:
            return cone_axis
        
        proj_perp_norm = proj_perp / np.linalg.norm(proj_perp)
        return np.cos(max_angle) * cone_axis + np.sin(max_angle) * proj_perp_norm


class ThrustRateConstraint(ControlConstraint):
    def __init__(self, max_thrust_rate: float = 10000.0):
        self.max_thrust_rate = max_thrust_rate
        self.previous_thrust = np.zeros(3)
        self.dt = 0.5
    
    def set_timestep(self, dt: float):
        self.dt = dt
    
    def apply_constraint(self, thrust_cmd: np.ndarray, state: SimulationState) -> Tuple[np.ndarray, bool, str]:
        thrust_change = thrust_cmd - self.previous_thrust
        thrust_rate = np.linalg.norm(thrust_change) / self.dt
        
        if thrust_rate > self.max_thrust_rate:
            limited_change = thrust_change * (self.max_thrust_rate * self.dt / np.linalg.norm(thrust_change))
            constrained_thrust = self.previous_thrust + limited_change
            self.previous_thrust = constrained_thrust
            return constrained_thrust, True, f"Thrust rate limited to {self.max_thrust_rate:.0f} N/s"
        
        self.previous_thrust = thrust_cmd
        return thrust_cmd, False, ""


class ApolloDescentController:
    def __init__(self, descent_params: DescentParameters, planetary_params: PlanetaryParameters, 
                 landing_site_params: LandingSiteParameters = None, constraints: List[ControlConstraint] = None):
        self.params = descent_params
        self.planetary = planetary_params
        self.landing_site = landing_site_params or LandingSiteParameters()
        self.entered_low_gate = False
        self.constraints = constraints or []
    
    def update(self, state: SimulationState) -> ControlCommand:
        altitude = self._compute_altitude(state.position)
        r_hat = self._compute_radial_unit_vector(state.position)
        
        # Use landing-site-relative velocity decomposition
        v_north, v_east, v_down = self._decompose_velocity_landing_site(state.velocity, state.position)
        v_horizontal = v_north + v_east  # Combined horizontal velocity
        vertical_velocity = np.dot(state.velocity, r_hat)  # Keep radial velocity for altitude control
        
        if altitude > self.params.high_gate_altitude:
            phase, thrust_cmd = self._high_altitude_phase_ls(state, altitude, r_hat, v_down, v_horizontal, vertical_velocity)
        elif altitude > self.params.target_low_gate_altitude:
            phase, thrust_cmd = self._approach_phase_ls(state, r_hat, v_down, v_horizontal, vertical_velocity)
        else:
            phase, thrust_cmd = self._terminal_phase_ls(state, r_hat, v_down, v_horizontal, vertical_velocity)
        
        thrust_cmd = self._limit_thrust(thrust_cmd)
        thrust_cmd, is_constrained, constraint_info = self._apply_constraints(thrust_cmd, state)
        fuel_rate = self._compute_fuel_consumption(thrust_cmd)
        
        return ControlCommand(
            thrust_vector=thrust_cmd,
            fuel_consumption_rate=fuel_rate,
            phase=phase,
            is_constrained=is_constrained,
            constraint_info=constraint_info
        )
    
    def _compute_altitude(self, position: np.ndarray) -> float:
        return np.linalg.norm(position) - self.planetary.radius
    
    def _compute_radial_unit_vector(self, position: np.ndarray) -> np.ndarray:
        return position / np.linalg.norm(position)
    
    def _decompose_velocity(self, velocity: np.ndarray, r_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v_vert = np.dot(velocity, r_hat) * r_hat
        v_horiz = velocity - v_vert
        return v_vert, v_horiz
    
    def _compute_landing_site_frame(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute North-East-Down frame vectors at the landing site"""
        lat = self.landing_site.latitude_rad
        lon = self.landing_site.longitude_rad
        
        # Landing site position vector in inertial frame
        R = self.planetary.radius
        landing_site_pos = np.array([
            R * np.cos(lat) * np.cos(lon),
            R * np.cos(lat) * np.sin(lon), 
            R * np.sin(lat)
        ])
        
        # Down vector (towards planet center from landing site)
        down = -landing_site_pos / np.linalg.norm(landing_site_pos)
        
        # North vector (tangent to meridian, pointing towards north pole)
        north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
        
        # East vector (tangent to parallel, pointing east)
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        
        return north, east, down
    
    def _decompose_velocity_landing_site(self, velocity: np.ndarray, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose velocity into North-East-Down components relative to landing site"""
        north, east, down = self._compute_landing_site_frame(position)
        
        v_north = np.dot(velocity, north)
        v_east = np.dot(velocity, east) 
        v_down = np.dot(velocity, down)
        
        return v_north * north, v_east * east, v_down * down
    
    def _high_altitude_phase_ls(self, state: SimulationState, altitude: float, r_hat: np.ndarray, 
                               v_down: np.ndarray, v_horizontal: np.ndarray, vertical_velocity: float) -> Tuple[DescentPhase, np.ndarray]:
        """High altitude phase using landing-site-relative frame"""
        v_total = np.linalg.norm(state.velocity)
        g_moon = self.planetary.mu / np.linalg.norm(state.position)**2
        max_a = self.params.max_thrust / state.mass - g_moon
        h_burn = self.params.burn_margin * v_total**2 / (2 * max_a) if max_a > 0 else 0
        
        if altitude <= h_burn:
            vel_error_vert = vertical_velocity - self.params.target_low_gate_vspeed
            a_cmd_vert = -self.params.k_p * vel_error_vert * r_hat
            a_cmd_horiz = -self.params.k_horiz * v_horizontal
            a_cmd = a_cmd_vert + a_cmd_horiz
            thrust_cmd = state.mass * a_cmd
            return DescentPhase.SUICIDE_BURN, thrust_cmd
        else:
            return DescentPhase.COASTING, np.zeros(3)
    
    def _approach_phase_ls(self, state: SimulationState, r_hat: np.ndarray, 
                          v_down: np.ndarray, v_horizontal: np.ndarray, vertical_velocity: float) -> Tuple[DescentPhase, np.ndarray]:
        """Approach phase using landing-site-relative frame"""
        vel_error_vert = vertical_velocity - self.params.target_low_gate_vspeed
        a_cmd_vert = -self.params.k_p * vel_error_vert * r_hat
        a_cmd_horiz = -self.params.k_horiz * v_horizontal
        a_cmd = a_cmd_vert + a_cmd_horiz
        thrust_cmd = state.mass * a_cmd
        return DescentPhase.APPROACH, thrust_cmd
    
    def _terminal_phase_ls(self, state: SimulationState, r_hat: np.ndarray, 
                          v_down: np.ndarray, v_horizontal: np.ndarray, vertical_velocity: float) -> Tuple[DescentPhase, np.ndarray]:
        """Terminal phase using landing-site-relative frame"""
        if not self.entered_low_gate:
            print(f"Entered Low Gate at t={state.time:.2f} s with vertical speed: {vertical_velocity:.2f} m/s")
            self.entered_low_gate = True
        
        vel_error_vert = vertical_velocity - self.params.target_descent_rate
        a_cmd_vert = -self.params.k_p * vel_error_vert * r_hat
        a_cmd_horiz = -self.params.k_horiz * v_horizontal
        a_cmd = a_cmd_vert + a_cmd_horiz
        thrust_cmd = state.mass * a_cmd
        return DescentPhase.TERMINAL, thrust_cmd
    
    def _high_altitude_phase(self, state: SimulationState, altitude: float, r_hat: np.ndarray, 
                           v_vert: np.ndarray, v_horiz: np.ndarray, vertical_velocity: float) -> Tuple[DescentPhase, np.ndarray]:
        v_total = np.linalg.norm(state.velocity)
        g_moon = self.planetary.mu / np.linalg.norm(state.position)**2
        max_a = self.params.max_thrust / state.mass - g_moon
        h_burn = self.params.burn_margin * v_total**2 / (2 * max_a) if max_a > 0 else 0
        
        if altitude <= h_burn:
            vel_error_vert = vertical_velocity - self.params.target_low_gate_vspeed
            a_cmd_vert = -self.params.k_p * vel_error_vert * r_hat
            a_cmd_horiz = -self.params.k_horiz * v_horiz
            a_cmd = a_cmd_vert + a_cmd_horiz
            thrust_cmd = state.mass * a_cmd
            return DescentPhase.SUICIDE_BURN, thrust_cmd
        else:
            return DescentPhase.COASTING, np.zeros(3)
    
    def _approach_phase(self, state: SimulationState, r_hat: np.ndarray, 
                       v_vert: np.ndarray, v_horiz: np.ndarray, vertical_velocity: float) -> Tuple[DescentPhase, np.ndarray]:
        vel_error_vert = vertical_velocity - self.params.target_low_gate_vspeed
        a_cmd_vert = -self.params.k_p * vel_error_vert * r_hat
        a_cmd_horiz = -self.params.k_horiz * v_horiz
        a_cmd = a_cmd_vert + a_cmd_horiz
        thrust_cmd = state.mass * a_cmd
        return DescentPhase.APPROACH, thrust_cmd
    
    def _terminal_phase(self, state: SimulationState, r_hat: np.ndarray, 
                       v_vert: np.ndarray, v_horiz: np.ndarray, vertical_velocity: float) -> Tuple[DescentPhase, np.ndarray]:
        if not self.entered_low_gate:
            print(f"Entered Low Gate at t={state.time:.2f} s with vertical speed: {vertical_velocity:.2f} m/s")
            self.entered_low_gate = True
        
        vel_error_vert = vertical_velocity - self.params.target_descent_rate
        a_cmd_vert = -self.params.k_p * vel_error_vert * r_hat
        a_cmd_horiz = -self.params.k_horiz * v_horiz
        a_cmd = a_cmd_vert + a_cmd_horiz
        thrust_cmd = state.mass * a_cmd
        return DescentPhase.TERMINAL, thrust_cmd
    
    def _limit_thrust(self, thrust_cmd: np.ndarray) -> np.ndarray:
        thrust_mag = np.linalg.norm(thrust_cmd)
        if thrust_mag > self.params.max_thrust:
            thrust_cmd = thrust_cmd * self.params.max_thrust / thrust_mag
        return thrust_cmd
    
    def _compute_fuel_consumption(self, thrust_cmd: np.ndarray) -> float:
        thrust_mag = np.linalg.norm(thrust_cmd)
        return thrust_mag / (self.params.isp * self.params.g0)
    
    def _apply_constraints(self, thrust_cmd: np.ndarray, state: SimulationState) -> Tuple[np.ndarray, bool, str]:
        constrained_thrust = thrust_cmd
        is_constrained = False
        constraint_messages = []
        
        for constraint in self.constraints:
            constrained_thrust, constraint_applied, message = constraint.apply_constraint(constrained_thrust, state)
            if constraint_applied:
                is_constrained = True
                if message:
                    constraint_messages.append(message)
        
        constraint_info = "; ".join(constraint_messages)
        return constrained_thrust, is_constrained, constraint_info
    
    def add_constraint(self, constraint: ControlConstraint):
        self.constraints.append(constraint)
    
    def remove_constraint(self, constraint_type: type):
        self.constraints = [c for c in self.constraints if not isinstance(c, constraint_type)]
    
    def clear_constraints(self):
        self.constraints.clear()