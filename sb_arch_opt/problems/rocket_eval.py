"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

This test suite contains a multi-stage launch vehicle test problem, originally published here:
https://github.com/raul7gs/Space_launcher_benchmark_problem
"""
import math
import logging
import warnings
import numpy as np
from enum import Enum
from typing import List
from dataclasses import dataclass

import scipy.optimize as opt
from scipy.integrate import solve_ivp

try:
    import ambiance
    HAS_ROCKET = True
except ImportError:
    HAS_ROCKET = False

__all__ = ['HAS_ROCKET', 'Rocket', 'Stage', 'Engine', 'HeadShape', 'RocketEvaluator', 'check_dependency']

log = logging.getLogger('sb_arch_opt.rocket')
warnings.filterwarnings('ignore', message='Tolerance of .* reached', category=RuntimeWarning)


def check_dependency():
    if not HAS_ROCKET:
        raise RuntimeError('Rocket dependencies not installed: pip install sb-arch-opt[rocket]')


class Engine(Enum):
    SRB = 'SRB'
    P80 = 'P80'
    GEM60 = 'GEM60'
    VULCAIN = 'VULCAIN'
    RS68 = 'RS68'
    S_IVB = 'S_IVB'


@dataclass(frozen=True)
class Stage:
    engines: List[Engine]
    length: float = 20.  # m


class HeadShape(Enum):
    ELLIPTICAL = 'Elliptical'
    SPHERE = 'Sphere'
    CONE = 'Cone'


@dataclass(frozen=True)
class Rocket:
    stages: List[Stage]
    head_shape: HeadShape = HeadShape.ELLIPTICAL
    cone_angle: float = 45.  # deg
    ellipse_l_ratio: float = .2
    length_diameter_ratio: float = 12.
    max_q: float = 50000.  # Pa
    payload_density: float = 2810.  # kg/m3
    orbit_altitude: float = 400e3  # m
    payload_mass: float = None  # kg


@dataclass(frozen=True)
class RocketPerformance:
    cost: float  # $
    payload_mass: float  # kg
    delta_structural: float  # maxQ constraint (violated if positive)
    delta_payload: float  # payload volume constraint (violated if positive)
    delta_delta_v: float  # delta V constraint (violated if possible)


class RocketEvaluator:

    _rho_interp = None

    @classmethod
    def evaluate(cls, rocket: Rocket) -> RocketPerformance:
        check_dependency()

        # Calculate geometrical data
        stage_lengths = [stage.length for stage in rocket.stages]
        if sum(stage_lengths) == 0:
            stage_lengths = [.1 for _ in range(len(stage_lengths))]
        stage_engines = [''.join(engine.value.lower() for engine in stage.engines) for stage in rocket.stages]
        diameter, surface_head, volume_available, stages_volume, fuel_volumes, ox_volumes, fuel_surfaces, \
            oxidizer_surfaces = cls.calculate_geometry(
                rocket.length_diameter_ratio, stage_lengths, rocket.head_shape.value, rocket.cone_angle,
                rocket.ellipse_l_ratio, stage_engines)

        # Calculate stage propellant and structural masses
        stage_thrusts = []
        stage_mdots = []
        stage_engine_masses = []
        stage_structural_masses = []
        stage_head_structure_masses = []
        stage_prop_masses = []
        stage_n_engines = []
        stage_solid_prop_masses = []
        stage_h2_masses = []
        stage_lox_masses = []
        for i, stage in enumerate(rocket.stages):
            # Engine performance
            is_liquid = True
            thrust, expansion_ratio, mdot = cls.calculate_liquid_engine(stage.engines.count(Engine.VULCAIN),
                                                                        stage.engines.count(Engine.RS68),
                                                                        stage.engines.count(Engine.S_IVB))
            if thrust == 0:
                is_liquid = False
                expansion_ratio = 0
                thrust, mdot = cls.calculate_solid_engine(stage.engines.count(Engine.SRB),
                                                          stage.engines.count(Engine.P80),
                                                          stage.engines.count(Engine.GEM60))
            stage_thrusts.append(thrust)
            stage_mdots.append(mdot)
            stage_n_engines.append(len(stage.engines))

            # Propellant mass
            n_srb, n_p80, n_gem60 = \
                stage.engines.count(Engine.SRB), stage.engines.count(Engine.P80), stage.engines.count(Engine.GEM60)
            propellant_mass, h2_mass, lox_mass = \
                cls.calculate_propellant_mass(n_srb, n_p80, n_gem60, stages_volume[i], fuel_volumes[i], ox_volumes[i])

            stage_prop_mass = (h2_mass+lox_mass) if is_liquid else propellant_mass
            stage_prop_masses.append(stage_prop_mass)
            stage_solid_prop_masses.append(propellant_mass)
            stage_h2_masses.append(h2_mass)
            stage_lox_masses.append(lox_mass)

            # Structural sizing
            n_vulcain, n_rs68, n_s_ivb = \
                stage.engines.count(Engine.VULCAIN), stage.engines.count(Engine.RS68), stage.engines.count(Engine.S_IVB)
            head_surface = surface_head if i == len(rocket.stages)-1 else 0
            mass_casing, mass_tank, mass_insulation, pumps_mass, structure_mass = \
                cls.calculate_structural_mass(
                    propellant_mass, thrust, expansion_ratio, ox_volumes[i], fuel_volumes[i], oxidizer_surfaces[i],
                    fuel_surfaces[i], n_srb, n_p80, n_gem60, n_vulcain, n_rs68, n_s_ivb, head_surface)

            stage_struct_mass = (pumps_mass + mass_tank + mass_insulation) if is_liquid else mass_casing
            stage_engine_masses.append(stage_struct_mass)
            stage_structural_masses.append(stage_struct_mass + structure_mass)
            stage_head_structure_masses.append(structure_mass)

        # Solve trajectory
        cone_angle = rocket.cone_angle if rocket.head_shape == HeadShape.CONE else 0
        length_ratio = rocket.ellipse_l_ratio if rocket.head_shape == HeadShape.ELLIPTICAL else 0
        payload_mass, h_vector, v_vector, delta_delta_v = cls.calculate_trajectory(
            cone_angle, length_ratio, diameter, stage_thrusts, stage_structural_masses, stage_prop_masses, stage_mdots,
            rocket.orbit_altitude, m_payload_fix=rocket.payload_mass)

        # Cost estimation
        cost = cls.calculate_cost(stage_n_engines, stage_engine_masses, stage_solid_prop_masses, stage_h2_masses,
                                  stage_lox_masses, stage_head_structure_masses)

        # Constraints
        delta_structural = cls.calculate_max_q_constraint(rocket.max_q, h_vector, v_vector)
        delta_payload = cls.calculate_payload_constraint(volume_available, payload_mass, rocket.payload_density)

        return RocketPerformance(
            cost=cost, payload_mass=payload_mass,
            delta_structural=delta_structural, delta_payload=delta_payload, delta_delta_v=-delta_delta_v,
        )

    @staticmethod
    def calculate_liquid_engine(vulcain, rs68, s_ivb):
        """Calculation of the stage propulsion properties."""

        thrust = 0
        expansion_ratio = 0
        mdot = 0

        expansion_ratios = {"VULCAIN": 45,
                            "RS68": 21.5,
                            "S_IVB": 28}

        mdot_dic = {"VULCAIN": 188.33,    # Mass flow in kg/s
                    "RS68": 807.39,
                    "S_IVB": 247}

        t_per_engine = {"VULCAIN": 0.8e6,    # Thrust in N
                        "RS68": 2.891e6,
                        "S_IVB": 0.486e6}

        if vulcain + rs68 + s_ivb > 0:
            thrust = vulcain*t_per_engine["VULCAIN"] + rs68*t_per_engine["RS68"] + s_ivb*t_per_engine["S_IVB"]

        if vulcain > 0:
            expansion_ratio = expansion_ratios["VULCAIN"]
            mdot = mdot_dic["VULCAIN"] * vulcain

        if rs68 > 0:
            expansion_ratio = expansion_ratios["RS68"]
            mdot = mdot_dic["RS68"] * rs68

        if s_ivb > 0:
            expansion_ratio = expansion_ratios["S_IVB"]
            mdot = mdot_dic["S_IVB"] * s_ivb

        return thrust, expansion_ratio, mdot

    @staticmethod
    def calculate_solid_engine(srb, p80, gem60):
        """Calculation of the stage propulsion properties."""

        thrust = 0
        mdot = 0

        mdot_dic = {"SRB": 5290,    # Mass flow in kg/s
                    "P80": 764,
                    "GEM60": 463.18}

        thrust_per_engine = {"SRB": 12.45e6,    # Thrust in N
                             "P80": 2.1e6,
                             "GEM60": 1.245e6}

        if srb + p80 + gem60 > 0:
            thrust = srb*thrust_per_engine["SRB"] + p80*thrust_per_engine["P80"] + gem60*thrust_per_engine["GEM60"]
        if srb > 0:
            mdot = mdot_dic["SRB"] * srb
        if p80 > 0:
            mdot = mdot_dic["P80"] * p80
        if gem60 > 0:
            mdot = mdot_dic["GEM60"] * gem60

        return thrust, mdot

    @staticmethod
    def calculate_geometry(l_d, length_stages_list, head_shape, cone_angle, l_ratio, engines_stages):
        """Calculation of the launcher geometric properties."""

        ratio_o_f = 7.937  # Stoichiometric oxidizer to fuel ratio
        densities_per_material = {"LOX": 1140,
                                  "LH2": 71}

        total_length = sum(length_stages_list)
        diameter = total_length / l_d
        radius = diameter / 2

        # Head geometric properties
        if head_shape == "Cone":
            h_cone = radius / math.tan(cone_angle * math.pi / 180)
            volume_available = 1 / 3 * math.pi * h_cone * radius ** 2
            g = (radius ** 2 + h_cone ** 2) ** 0.5
            surface_tip = math.pi * radius * g
        elif head_shape == "Sphere":
            volume_available = 4 / 3 * math.pi * radius ** 3 / 2
            surface_tip = 2 * math.pi * radius ** 2
        else:
            l_ellipse = l_ratio * total_length
            volume_available = 2 / 3 * math.pi * l_ellipse * radius ** 2
            eps = ((l_ellipse ** 2 - radius ** 2) ** 0.5) / l_ellipse
            surface_tip = math.pi * l_ellipse ** 2 + (np.log((1 + eps) / (1 - eps)) * math.pi / eps * radius ** 2) / 2

        stages_volume = []
        fuel_volumes = []
        oxidizer_volumes = []
        fuel_surfaces = []
        oxidizer_surfaces = []

        # Propellant related geometric properties
        for stage in range(len(length_stages_list)):

            length_stage = length_stages_list[stage]
            volume_stage = math.pi * diameter ** 2 / 4 * length_stage
            stages_volume.append(volume_stage)

            vulcain = engines_stages[stage].count("vulcain")
            rs68 = engines_stages[stage].count("rs68")
            s_ivb = engines_stages[stage].count("s_ivb")

            if vulcain + rs68 + s_ivb > 0:
                volume_h2 = volume_stage / (
                            densities_per_material["LH2"] * ratio_o_f / densities_per_material["LOX"] + 1)
                volume_lox = volume_stage - volume_h2
                height_oxidizer = volume_lox / math.pi / radius / radius
                height_fuel = volume_h2 / math.pi / radius / radius
                s_tank_oxidizer = math.pi * radius ** 2 * 2 + 2 * math.pi * radius * height_oxidizer
                s_tank_fuel = math.pi * radius ** 2 * 2 + 2 * math.pi * radius * height_fuel
                fuel_volumes.append(volume_h2)
                oxidizer_volumes.append(volume_lox)
                fuel_surfaces.append(s_tank_fuel)
                oxidizer_surfaces.append(s_tank_oxidizer)
            else:
                volume_h2 = 0
                volume_lox = 0
                s_tank_oxidizer = 0
                s_tank_fuel = 0
                fuel_volumes.append(volume_h2)
                oxidizer_volumes.append(volume_lox)
                fuel_surfaces.append(s_tank_fuel)
                oxidizer_surfaces.append(s_tank_oxidizer)

        return diameter, surface_tip, volume_available, stages_volume, fuel_volumes, oxidizer_volumes, fuel_surfaces, \
            oxidizer_surfaces

    @staticmethod
    def calculate_propellant_mass(srb, p80, gem60, volume_tank, volume_h2, volume_lox):
        """Calculation of the launcher propellant mass."""

        propellant_mass = 0
        h2_mass = 0
        lox_mass = 0

        densities_per_material = {"PBAN": 1715,    # Propellant densities in kg/m^3
                                  "HTPB1912": 1810,
                                  "HTPB_APCP": 1650,
                                  "LOX": 1140,
                                  "LH2": 71}

        corrections_per_material = {"PBAN": 1,    # Correction for ullage volume
                                    "HTPB1912": 1,
                                    "HTPB_APCP": 1,
                                    "LOX": 0.94}

        if srb > 0:
            propellant_mass = densities_per_material["PBAN"] * corrections_per_material["PBAN"] * volume_tank
        elif p80 > 0:
            propellant_mass = densities_per_material["HTPB1912"] * corrections_per_material["HTPB1912"] * volume_tank
        elif gem60 > 0:
            propellant_mass = densities_per_material["HTPB_APCP"] * corrections_per_material["HTPB_APCP"] * volume_tank
        else:
            h2_mass = volume_h2 * densities_per_material["LH2"] * corrections_per_material["LOX"]
            lox_mass = volume_lox * densities_per_material["LOX"] * corrections_per_material["LOX"]

        return propellant_mass, h2_mass, lox_mass

    @staticmethod
    def calculate_structural_mass(mass_propellant, thrust, expansion_ratio, oxidizer_tank_volume, fuel_tank_volume,
                                  s_tank_oxidizer, s_tank_fuel, srb, p80, gem60, vulcain, rs68, s_ivb, head_surface):
        """Calculation of the launcher structural mass."""

        if srb + p80 + gem60 > 0:  # Solid
            mass_casing = 0.135 * mass_propellant
            mass_tank = 0
            mass_insulation = 0
            pumps_mass = 0
        else:  # Liquid
            mass_casing = 0
            n_engines = vulcain + rs68 + s_ivb
            thrust_per_engine = thrust / n_engines

            mass_oxidizer_tank = 12.158 * oxidizer_tank_volume
            mass_fuel_tank = 9.0911 * fuel_tank_volume
            mass_tank = mass_oxidizer_tank + mass_fuel_tank
            mass_oxidizer_insulation = 1.123 * s_tank_oxidizer
            mass_fuel_insulation = 2.88 * s_tank_fuel
            mass_insulation = mass_oxidizer_insulation + mass_fuel_insulation
            pumps_mass = (7.81e-4 * thrust_per_engine + 3.37e-5 * expansion_ratio ** 0.5 + 59) * n_engines

        if head_surface > 0:
            material_density = 2780  # kg/m3
            thickness = 0.005
            structure_mass = head_surface * thickness * material_density
        else:
            structure_mass = 0

        return mass_casing, mass_tank, mass_insulation, pumps_mass, structure_mass

    @classmethod
    def modified_atmosphere(cls, x):

        if cls._rho_interp is None:
            x_sample = np.linspace(0, 80000, 100)
            cls._rho_interp = x_sample, np.log10(ambiance.Atmosphere(x_sample).density)

        if x < 0:
            x = 0
        elif x > 80000:
            x = 80000

        return 10**np.interp(x, *cls._rho_interp)

    @classmethod
    def calculate_trajectory(cls, cone_angle, length_ratio, diameter, T_stages, m_structural_stages, mp_stages,
                             mdot_stages, h_orbit_target, m_payload_fix=None):
        """Calculation of the launcher trajectory."""
        check_dependency()

        mu = 3.986004418e14
        r_earth = 6378e3

        # Drag coefficient calculation depending on head shape
        if cone_angle > 0:
            cd = 0.0122 * cone_angle + 0.162
        elif length_ratio > 0:
            m = -0.01 / 15  # experimental data
            cd = 0.305 + m * (length_ratio - 10)
        else:
            cd = 0.42

        s = np.pi + diameter ** 2 / 4
        stages_max = len(T_stages) - 1
        n_steps = 200

        def simulate_trajectory(m_payload):

            def stage_state_eq(t_, y):    # Trajectory equation
                x, v_ = y
                return [v_, T / ((m0 + m_payload) - mdot * t_) * np.cos(alpha) - 1 / 2 / (
                        (m0 + m_payload) - mdot * t_) * s * cd * cls.modified_atmosphere(x) * v_ ** 2 - g*np.sin(gamma)]

            # First stage
            h_first = []
            v_first = []
            stage = 0
            T = T_stages[stage]
            gamma = 90 * np.pi / 180
            alpha = 0
            m_unloaded = sum(m_structural_stages[stage:])
            m0 = m_unloaded + sum(mp_stages[stage:])
            mdot = mdot_stages[stage]

            g = 9.81
            tfinal = mp_stages[stage] / mdot - 5
            t = np.linspace(0, tfinal, n_steps)
            sol = solve_ivp(fun=stage_state_eq, t_span=[t[0], t[-1]], y0=[0, 0], t_eval=t, dense_output=False)
            h, v = sol.y
            pos, = np.where(h > 10000)

            if len(pos) == 0 and stage < stages_max:  # This means that a second rocket stage is available and needed
                h_first += h.tolist()
                v_first += v.tolist()
                stage = stage + 1
                T = T_stages[stage]
                m_unloaded = sum(m_structural_stages[stage:])
                m0 = m_unloaded + sum(mp_stages[stage:])
                mdot = mdot_stages[stage]
                tfinal = mp_stages[stage] / mdot - 5
                t = np.linspace(0, tfinal, n_steps)
                sol = solve_ivp(fun=stage_state_eq, t_span=[t[0], t[-1]], y0=[h[-1], v[-1]], t_eval=t, dense_output=False)
                h, v = sol.y
                pos, = np.where(h > 10000)
                if len(pos) == 0 and stage < stages_max:  # This means that a third rocket stage is available and needed
                    h_first += h.tolist()
                    v_first += v.tolist()
                    stage = stage + 1
                    T = T_stages[stage]
                    m_unloaded = sum(m_structural_stages[stage:])
                    m0 = m_unloaded + sum(mp_stages[stage:])
                    mdot = mdot_stages[stage]
                    tfinal = mp_stages[stage] / mdot - 5
                    t = np.linspace(0, tfinal, n_steps)
                    sol = solve_ivp(fun=stage_state_eq, t_span=[t[0], t[-1]], y0=[h[-1], v[-1]], t_eval=t, dense_output=False)
                    h, v = sol.y
                    pos, = np.where(h > 10000)
                    if len(pos) == 0:
                        h_first += h.tolist()
                        v_first += v.tolist()
                    else:
                        h_append = h[0:pos[0]]
                        v_append = v[0:pos[0]]
                        h_first += h_append.tolist()
                        v_first += v_append.tolist()
                else:
                    h_append = h[0:pos[0]]
                    v_append = v[0:pos[0]]
                    h_first += h_append.tolist()
                    v_first += v_append.tolist()
            elif len(pos) > 0:
                h_append = h[0:pos[0]]
                v_append = v[0:pos[0]]
                h_first += h_append.tolist()
                v_first += v_append.tolist()
            else:
                h_first += h.tolist()
                v_first += v.tolist()

            # Second stage
            h_0 = h[pos[0]]
            v_0 = v[pos[0]]
            t_0 = t[pos[0]]
            m0 = m0 - mdot * t_0
            mp_remaining = mp_stages[stage] - mdot * t_0  # It could be that the previous stage did not finish
            h_second = []
            v_second = []

            alpha = 5 * np.pi / 180
            gamma = 135 * np.pi / 180

            tfinal = mp_remaining / mdot - 5
            t = np.linspace(0, tfinal, n_steps)
            sol2 = solve_ivp(fun=stage_state_eq, t_span=[t[0], t[-1]], y0=[h_0, v_0], t_eval=t, dense_output=False)
            h, v = sol2.y
            pos2, = np.where(h > 100000)
            if len(pos2) == 0 and stage < stages_max:  # This means that a second rocket stage is available and needed
                h_second += h.tolist()
                v_second += v.tolist()
                stage = stage + 1
                mp_remaining = mp_stages[stage]
                T = T_stages[stage]
                m_unloaded = sum(m_structural_stages[stage:])
                m0 = m_unloaded + sum(mp_stages[stage:])
                mdot = mdot_stages[stage]
                tfinal = mp_stages[stage] / mdot - 5
                t = np.linspace(0, tfinal, n_steps)
                sol2 = solve_ivp(fun=stage_state_eq, t_span=[t[0], t[-1]], y0=[h[-1], v[-1]], t_eval=t, dense_output=False)
                h, v = sol2.y
                pos2, = np.where(h > 100000)
                if len(pos2) == 0 and stage < stages_max:  # A third rocket stage is available and needed
                    h_second += h.tolist()
                    v_second += v.tolist()
                    stage = stage + 1
                    mp_remaining = mp_stages[stage]
                    T = T_stages[stage]
                    m_unloaded = sum(m_structural_stages[stage:])
                    m0 = m_unloaded + sum(mp_stages[stage:])
                    mdot = mdot_stages[stage]
                    tfinal = mp_stages[stage] / mdot - 5
                    t = np.linspace(0, tfinal, n_steps)
                    sol2 = solve_ivp(fun=stage_state_eq, t_span=[t[0], t[-1]], y0=[h[-1], v[-1]], t_eval=t, dense_output=False)
                    h, v = sol2.y
                    pos2, = np.where(h > 100000)
                    if len(pos2) == 0:
                        h_second += h.tolist()
                        v_second += v.tolist()
                    else:
                        h_append = h[0:pos2[0]]
                        v_append = v[0:pos2[0]]
                        h_second += h_append.tolist()
                        v_second += v_append.tolist()
                else:
                    h_append = h[0:pos2[0]]
                    v_append = v[0:pos2[0]]
                    h_second += h_append.tolist()
                    v_second += v_append.tolist()
            elif len(pos2) > 0:
                h_append = h[0:pos2[0]]
                v_append = v[0:pos2[0]]
                h_second += h_append.tolist()
                v_second += v_append.tolist()
            else:
                h_second += h.tolist()
                v_second += v.tolist()

            # Third stage
            v_0 = v[pos2[0]]
            t_0 = t[pos2[0]]
            m0 = m0 - mdot * t_0
            mp_remaining = mp_remaining - mdot * t_0
            if m0 + m_payload - mp_remaining <= 0:
                delta_v = 0
            else:
                delta_v = T / mdot * np.log((m0 + m_payload) / (m0 + m_payload - mp_remaining))
            v_orbit_final = v_0 / 2 + delta_v

            while stage < stages_max:
                stage = stage + 1
                T = T_stages[stage]
                m_unloaded = sum(m_structural_stages[stage:])
                m0 = m_unloaded + sum(mp_stages[stage:])
                mdot = mdot_stages[stage]
                tfinal = mp_stages[stage] / mdot
                delta_v = T / ((m0 + m_payload) - mdot * tfinal) * tfinal
                v_orbit_final += delta_v

            h_vector_ = h_first + h_second
            v_vector_ = v_first + v_second

            return v_orbit_final, h_vector_, v_vector_

        def try_payload(m_payload):
            v_orbit = (mu / (r_earth + h_orbit_target)) ** 0.5
            try:
                v_final_, h_vector_, v_vector_ = simulate_trajectory(m_payload)

                # Orbit minimum speed
                v_target_diff = v_final_ - v_orbit

                return v_target_diff, v_final_, h_vector_, v_vector_
            except (IndexError, ValueError):
                return -v_orbit, 0, [], []

        # Evaluate for fixed payload mass
        if m_payload_fix:
            v_tgt_diff, _, h_vector, v_vector = try_payload(m_payload_fix)
            if v_tgt_diff < 0:
                return 0, [], []
            return m_payload_fix, h_vector, v_vector, v_tgt_diff

        # Check if rocket could be feasible even without payload
        v_tgt_diff, _, _, _ = try_payload(0)
        if v_tgt_diff <= 0:
            return 0, [], [], v_tgt_diff

        m_payload, res = opt.newton(
            lambda mp_: try_payload(mp_)[0], 100, tol=1., maxiter=50, full_output=True, disp=False)
        if not res.converged:
            return 0, [], [], v_tgt_diff

        v_tgt_diff, _, h_vector, v_vector = try_payload(m_payload)
        return m_payload, h_vector, v_vector, v_tgt_diff

    @staticmethod
    def calculate_cost(n_engines_list, m_engine_list, m_solid_list, m_h2_list, m_lox_list, structure_mass_list):
        """Calculation of the launcher cost."""

        # Fuel costs
        fuel_cost_per_kg = {"Solid": 5,
                            "LOX": 0.27,
                            "LH2": 6.1}

        total_cost = 0

        for index in range(len(n_engines_list)):
            n_engines = n_engines_list[index]
            m_engine = m_engine_list[index]
            m_solid = m_solid_list[index]
            m_h2 = m_h2_list[index]
            m_lox = m_lox_list[index]
            structure_mass = structure_mass_list[index]

            # Engine cost
            if m_solid > 0:  # Solid
                production_cost_engines_stage = 0.85 * n_engines * 2.3 * (m_engine + m_solid) ** 0.399  # page 125 TC
            else:    # Liquid
                production_cost_engines_stage = 0.85 * n_engines * 5.16 * m_engine ** 0.45  # page 129 TRANSCOST

            engine_cost_years = production_cost_engines_stage
            engine_cost_stage = engine_cost_years * 366518  # Conversion to dollars

            # Propellant cost
            if m_h2 > 0:
                fuel_cost_stage = m_h2 * fuel_cost_per_kg["LH2"] + m_lox * fuel_cost_per_kg["LOX"]
            else:
                fuel_cost_stage = m_solid * fuel_cost_per_kg["Solid"]

            # Structure cost
            if structure_mass > 0:
                material_cost = 3  # Dollars per kg
                structure_cost_stage = material_cost * structure_mass
            else:
                structure_cost_stage = 0

            total_cost = total_cost + engine_cost_stage + fuel_cost_stage + structure_cost_stage

        return total_cost

    @classmethod
    def calculate_max_q_constraint(cls, qmax, h, v):
        """Calculation of the launcher structural constraint. It is considered to have overpassed the maximum
        structural load if it is positive."""
        check_dependency()

        if len(h) == 0 or len(v) == 0:
            return qmax

        rho = []
        for altitude in h:
            if altitude < 0:
                rho.append(1.225)
            else:
                rho.append(cls.modified_atmosphere(altitude))

        rho_array = np.array(rho)
        qvector = 0.5 * np.multiply(np.multiply(v, v), rho_array)
        difference = np.subtract(qvector, qmax * np.ones(len(qvector)))
        constraint = np.max(difference)

        return constraint

    @staticmethod
    def calculate_payload_constraint(available_volume, mass, density):
        """Calculation of the constraint. The constraint is satisfied when negative"""

        volume_payload = mass / density
        difference = volume_payload - available_volume

        return difference
