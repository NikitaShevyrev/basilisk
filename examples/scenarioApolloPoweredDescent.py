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
oe.e = 0.05
oe.i = 0.0 * macros.D2R
oe.Omega = 0.0
oe.omega = 0.0
oe.f = 180.0 * macros.D2R
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
initial_thrust_duration = 10.0
burn_margin = 1.2
target_descent_rate = -1.0
k_p = 0.8
k_horiz = 0.6

# Initialize
scSim.InitializeSimulation()
altitude_log, velocity_log, fuel_mass_log, thrust_mag_log = [], [], [], []

# Run simulation
high_gate_altitude = 2134
target_low_gate_altitude = 150
target_low_gate_vspeed = -5.0
low_gate_tolerance = 10.0
entered_low_gate = False

for step in range(int(simulationTime / simulationTimeStep)):
    sim_time = step * macros.NANO2SEC * simulationTimeStep
    scSim.ConfigureStopTime((step + 1) * simulationTimeStep)

    stateData = scObject.scStateOutMsg.read()
    r = np.array(stateData.r_BN_N)
    v = np.array(stateData.v_BN_N)

    altitude = np.linalg.norm(r) - moon_radius
    r_hat = r / np.linalg.norm(r)
    v_vert = np.dot(v, r_hat) * r_hat
    v_horiz = v - v_vert
    vertical_velocity = np.dot(v, r_hat)
    current_mass = scObject.hub.mHub

    if altitude > high_gate_altitude:
        # Predictive suicide burn calculation
        v_total = np.linalg.norm(v)
        g_moon = mu / np.linalg.norm(r)**2
        max_a = max_thrust / current_mass - g_moon
        h_burn = burn_margin * v_total**2 / (2 * max_a) if max_a > 0 else 0

        if altitude <= h_burn:
            # Begin full suicide burn
            vel_error_vert = vertical_velocity - target_low_gate_vspeed
            a_cmd_vert = -k_p * vel_error_vert * r_hat
            a_cmd_horiz = -k_horiz * v_horiz
            a_cmd = a_cmd_vert + a_cmd_horiz
            thrust_cmd = current_mass * a_cmd
            thrust_mag = np.linalg.norm(thrust_cmd)
            if thrust_mag > max_thrust:
                thrust_cmd *= max_thrust / thrust_mag
        else:
            # Coasting
            thrust_cmd = np.zeros(3)

    elif high_gate_altitude >= altitude > target_low_gate_altitude:
        # Approach phase - aim to zero horizontal and reduce vertical to target low gate speed
        vel_error_vert = vertical_velocity - target_low_gate_vspeed
        a_cmd_vert = -k_p * vel_error_vert * r_hat
        a_cmd_horiz = -k_horiz * v_horiz
        a_cmd = a_cmd_vert + a_cmd_horiz
        thrust_cmd = current_mass * a_cmd
        thrust_mag = np.linalg.norm(thrust_cmd)
        if thrust_mag > max_thrust:
            thrust_cmd *= max_thrust / thrust_mag

    else:
        # Terminal descent
        if not entered_low_gate:
            print(f"Entered Low Gate at t={sim_time:.2f} s with vertical speed: {vertical_velocity:.2f} m/s")
            entered_low_gate = True

        vel_error_vert = vertical_velocity - target_descent_rate
        a_cmd_vert = -k_p * vel_error_vert * r_hat
        a_cmd_horiz = -k_horiz * v_horiz
        a_cmd = a_cmd_vert + a_cmd_horiz
        thrust_cmd = current_mass * a_cmd
        thrust_mag = np.linalg.norm(thrust_cmd)
        if thrust_mag > max_thrust:
            thrust_cmd *= max_thrust / thrust_mag

    thrust_mag = np.linalg.norm(thrust_cmd)
    mdot = thrust_mag / (isp * g0)
    dm = mdot * macros.NANO2SEC * simulationTimeStep
    fuel_mass -= dm
    fuel_mass = max(fuel_mass, 0.0)

    if fuel_mass <= 0:
        thrust_cmd = np.zeros(3)

    scObject.hub.mHub = dry_mass + fuel_mass
    thrustForce.extForce_N = thrust_cmd.reshape(3, 1)

    altitude_log.append(altitude)
    velocity_log.append(np.linalg.norm(v))
    fuel_mass_log.append(fuel_mass)
    thrust_mag_log.append(thrust_mag)

    scSim.ExecuteSimulation()

    if altitude <= 1.0:
        print(f"Landing at {sim_time:.2f} s")
        print(f"Final velocity vector: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}] m/s")
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
