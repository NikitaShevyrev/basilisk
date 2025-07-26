# scenarioApolloThrottledDescent.py
# Apollo-style powered lunar descent with velocity-based throttle and suicide burn

import numpy as np
import matplotlib.pyplot as plt
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody
from Basilisk.simulation import spacecraft, extForceTorque

# Create simulation
scSim = SimulationBaseClass.SimBaseClass()
simTaskName = "simTask"
simProcessName = "simProcess"
dynProcess = scSim.CreateNewProcess(simProcessName)
simulationTimeStep = macros.sec2nano(0.5)
dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

# Initialize spacecraft
scObject = spacecraft.Spacecraft()
scObject.ModelTag = "apolloLander"
dry_mass = 7500.0
initial_fuel_mass = 7500.0
scObject.hub.mHub = dry_mass + initial_fuel_mass
scObject.hub.IHubPntBc_B = np.diag([9000., 8000., 6000.])

# Setup Moon gravity
gravFactory = simIncludeGravBody.gravBodyFactory()
moon = gravFactory.createMoon()
moon.isCentralBody = True
mu = moon.mu
moon_radius = 1737.4e3

# Initial orbit
initAlt = 15000
oe = orbitalMotion.ClassicElements()
oe.a = moon_radius + initAlt
oe.e = 0.05  # Add slight eccentricity to introduce vertical component
oe.i = 0.0 * macros.D2R
oe.Omega = 0.0
oe.omega = 0.0
oe.f = 180.0 * macros.D2R  # Start at periapsis (descending)
rN, vN = orbitalMotion.elem2rv(mu, oe)
scObject.hub.r_CN_NInit = rN
scObject.hub.v_CN_NInit = vN

# Attach gravity
gravFactory.addBodiesTo(scObject)
scSim.AddModelToTask(simTaskName, scObject)

# Thrust setup
thrustForce = extForceTorque.ExtForceTorque()
thrustForce.ModelTag = "DescentThruster"
thrustForce.extForce_N = [[0.], [0.], [0.]]
scObject.addDynamicEffector(thrustForce)
scSim.AddModelToTask(simTaskName, thrustForce)

# Logging setup
simulationTime = macros.sec2nano(3600.)
samplingTime = simulationTimeStep
stateLog = scObject.scStateOutMsg.recorder(samplingTime)
scSim.AddModelToTask(simTaskName, stateLog)

# Constants
isp = 311.0
g0 = 9.80665
max_thrust = 45000.0
fuel_mass = initial_fuel_mass
initialMass = dry_mass + fuel_mass
initial_thrust_duration = 10.0  # seconds
burn_margin = 1.2
target_descent_rate = -1.0  # m/s
k_p = 0.8

# Initialize
scSim.InitializeSimulation()
altitude_log, velocity_log, fuel_mass_log, thrust_mag_log = [], [], [], []

# Run simulation
for step in range(int(simulationTime / simulationTimeStep)):
    sim_time = step * macros.NANO2SEC * simulationTimeStep
    scSim.ConfigureStopTime((step+1)*simulationTimeStep)

    stateData = scObject.scStateOutMsg.read()
    r = np.array(stateData.r_BN_N)
    v = np.array(stateData.v_BN_N)

    altitude = np.linalg.norm(r) - moon_radius
    vertical_velocity = np.dot(r, v) / np.linalg.norm(r)
    speed = np.linalg.norm(v)

    altitude_log.append(altitude)
    velocity_log.append(speed)
    fuel_mass_log.append(fuel_mass)

    current_mass = scObject.hub.mHub
    g_moon = mu / np.linalg.norm(r)**2
    max_a = max_thrust / current_mass - g_moon
    h_burn = burn_margin * vertical_velocity**2 / (2 * max_a) if max_a > 0 else 0

    desired_acceleration = np.zeros(3)

    if sim_time < initial_thrust_duration:
        # Kickstart descent
        throttle = 0.1
        acc_cmd = throttle * max_thrust / current_mass
        retro_dir = -v / np.linalg.norm(v)
        desired_acceleration = acc_cmd * retro_dir

    elif altitude <= h_burn:
        # Suicide burn phase
        vel_error = vertical_velocity - target_descent_rate
        acc_cmd = -k_p * vel_error
        desired_acceleration = acc_cmd * (r / np.linalg.norm(r))

    else:
        # Throttle based on descent speed
        throttle = min(1.0, abs(vertical_velocity) / 1700.0) if vertical_velocity < 0 else 0.0
        acc_cmd = throttle * max_thrust / current_mass
        retro_dir = -v / np.linalg.norm(v)
        desired_acceleration = acc_cmd * retro_dir

    thrust_mag = 0.0
    if fuel_mass > 0:
        required_thrust = current_mass * desired_acceleration
        thrust_mag = np.linalg.norm(required_thrust)
        if thrust_mag > max_thrust:
            required_thrust *= max_thrust / thrust_mag
            thrust_mag = max_thrust

        mdot = thrust_mag / (isp * g0)
        dm = mdot * macros.NANO2SEC * simulationTimeStep
        fuel_mass -= dm
        fuel_mass = max(fuel_mass, 0.0)
        scObject.hub.mHub = dry_mass + fuel_mass

        thrustForce.extForce_N = required_thrust.reshape(3, 1)
    else:
        thrustForce.extForce_N = [[0.], [0.], [0.]]

    thrust_mag_log.append(thrust_mag)
    scSim.ExecuteSimulation()

    if altitude <= 1.0:
        print(f"Landing at {sim_time:.2f} s")
        print(f"Final velocity: {velocity_log[-1]:.2f} m/s")
        print(f"Final fuel: {fuel_mass_log[-1]:.2f} kg")
        break

# Plot
plt.figure(figsize=(12, 10))
time_sec = np.arange(len(altitude_log)) * macros.NANO2SEC * simulationTimeStep

plt.subplot(411)
plt.plot(time_sec, np.array(altitude_log)/1000)
plt.ylabel('Altitude [km]')
plt.grid()

plt.subplot(412)
plt.plot(time_sec, velocity_log)
plt.ylabel('Velocity [m/s]')
plt.grid()

plt.subplot(413)
plt.plot(time_sec, fuel_mass_log)
plt.ylabel('Fuel Mass [kg]')
plt.grid()

plt.subplot(414)
plt.plot(time_sec, thrust_mag_log)
plt.xlabel('Time [s]')
plt.ylabel('Thrust [N]')
plt.grid()

plt.suptitle('Apollo-Style Lunar Descent with Velocity-Based Throttle and Suicide Burn')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

del scSim
