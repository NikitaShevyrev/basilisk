# scenarioApolloThrottledDescent.py
# Apollo-style powered lunar descent with velocity-based throttle and suicide burn

import numpy as np
import matplotlib.pyplot as plt
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody
from Basilisk.simulation import spacecraft, extForceTorque
from apollo_descent_controller import (
    ApolloDescentController, DescentParameters, PlanetaryParameters, LandingSiteParameters, SimulationState,
    FuelConstraint, OrientationConstraint, ThrustRateConstraint
)

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

# Initial orbit over Sea of Tranquility
initAlt = 16000
oe = orbitalMotion.ClassicElements()
oe.a = moon_radius + initAlt
oe.e = 0.0  # Circular orbit to reduce initial velocity
oe.i = 0.67 * macros.D2R  # Low inclination orbit for Sea of Tranquility access
oe.Omega = 0.0
oe.omega = 0.0
oe.f = 0.0 * macros.D2R  # Start at ascending node
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

# Setup control parameters
descent_params = DescentParameters(
    isp=311.0,
    g0=9.80665,
    max_thrust=40000.0,
    burn_margin=2.0,
    k_p=1.2,
    k_horiz=5.0,
    high_gate_altitude=2500,
    target_low_gate_altitude=150,
    target_low_gate_vspeed=-5.0,
    target_descent_rate=-1.0
)

planetary_params = PlanetaryParameters(
    mu=mu,
    radius=moon_radius
)

landing_site_params = LandingSiteParameters()  # Defaults to Sea of Tranquility

# Setup constraints
constraints = [
    FuelConstraint(min_fuel_margin=50.0),
    ThrustRateConstraint(max_thrust_rate=40000.0)
]

# Initialize controller
controller = ApolloDescentController(descent_params, planetary_params, landing_site_params, constraints)

# Fuel tracking
fuel_mass = initial_fuel_mass
initialMass = dry_mass + fuel_mass

# Initialize
scSim.InitializeSimulation()
altitude_log, velocity_log, fuel_mass_log, thrust_mag_log = [], [], [], []

# Run simulation

for step in range(int(simulationTime / simulationTimeStep)):
    sim_time = step * macros.NANO2SEC * simulationTimeStep
    scSim.ConfigureStopTime((step + 1) * simulationTimeStep)

    # Get simulation state
    stateData = scObject.scStateOutMsg.read()
    r = np.array(stateData.r_BN_N)
    v = np.array(stateData.v_BN_N)
    current_mass = scObject.hub.mHub
    
    # Create simulation state for controller
    sim_state = SimulationState(
        position=r,
        velocity=v,
        mass=current_mass,
        time=sim_time,
        fuel_mass=fuel_mass,
        attitude_quaternion=None
    )
    
    # Get control command from controller
    control_cmd = controller.update(sim_state)
    thrust_cmd = control_cmd.thrust_vector
    mdot = control_cmd.fuel_consumption_rate
    
    # Log constraint information if applicable
    if control_cmd.is_constrained and step % 20 == 0:
        print(f"t={sim_time:.1f}s: {control_cmd.constraint_info}")
    
    # Update fuel mass
    dm = mdot * macros.NANO2SEC * simulationTimeStep
    fuel_mass -= dm
    fuel_mass = max(fuel_mass, 0.0)

    if fuel_mass <= 0:
        thrust_cmd = np.zeros(3)
    
    altitude = np.linalg.norm(r) - moon_radius

    scObject.hub.mHub = dry_mass + fuel_mass
    thrustForce.extForce_N = thrust_cmd.reshape(3, 1)

    altitude_log.append(altitude)
    velocity_log.append(np.linalg.norm(v))
    fuel_mass_log.append(fuel_mass)
    thrust_mag_log.append(np.linalg.norm(thrust_cmd))

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
