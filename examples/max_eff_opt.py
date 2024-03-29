import matplotlib.pyplot as plt
import numpy as np

import openmdao.api as om

from ccblade_propeller import Propeller

prob = om.Problem()

n_cp = 5
hub_pos_nondim = 0.2
prob.model.add_subsystem("propeller", Propeller(
    n_cp=n_cp, hub_pos_nondim=hub_pos_nondim))

# Lower and upper limits on the chord design variable, in meters.
chord_lower = 0.5*0.0254
chord_upper = 5.0*0.0254
prob.model.add_design_var(
    "propeller.chord_cp", lower=chord_lower, upper=chord_upper, ref=1e-2)

# Lower and upper limits on the twist design variable, radians.
theta_lower = np.radians(5)
theta_upper = np.radians(85)
prob.model.add_design_var(
    "propeller.theta_cp", lower=theta_lower, upper=theta_upper, ref=1e0)

# Lower and upper limits on the propeller RPM design variable
rpm_lower = 500
rpm_upper = 8000
prob.model.add_design_var(
    "propeller.rpm", lower=rpm_lower, upper=rpm_upper, ref=1e3)

prob.model.add_design_var('propeller.Rtip')

# Target thrust value in Newtons.
thrust_target = 97.246
prob.model.add_constraint(
    "propeller.thrust", equals=thrust_target, units="N", ref=1e2)

prob.model.add_objective("propeller.efficiency", ref=-1)

prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")

prob.setup()

radii_non_dim = np.linspace(hub_pos_nondim, 1, n_cp)
Rtip = 24.0*0.0254/2
radii_cp = Rtip * radii_non_dim
c = 1.5*0.0254   # Constant chord in meters.
chord_cp0 = c*np.ones(n_cp)
P = 16.0*0.0254  # Pitch in meters (used in the twist distribution).
theta_cp0 = np.arctan(P/(2*np.pi*radii_cp))

print(f"radii_cp: {radii_cp}")
print(f"chord_cp: {chord_cp0}")
print(f"theta_cp: {theta_cp0}")

prop_x0 = {
    "propeller.rpm": 7200,
    'propeller.chord_cp': chord_cp0,
    'propeller.theta_cp': theta_cp0,
    'propeller.Rtip': Rtip
}

for key, value in prop_x0.items():
    prob[key][:] = value

M_infty = 0.11
T0 = 273.15 + 15.0
gam = 1.4
speedofsound = np.sqrt(gam*287.058*T0)
v = M_infty * speedofsound
prop_p0 = {
    "propeller.free_stream_velocity": v  # axial velocity in m/sec.
}

for key, value in prop_p0.items():
    prob[key][:] = value

prob.run_model()
prob.model.list_inputs(units=True, prom_name=True)
prob.model.list_outputs(residuals=True, units=True, prom_name=True)

prob.run_driver()
prob.model.list_inputs(units=True, prom_name=True)
prob.model.list_outputs(residuals=True, units=True, prom_name=True)

radii_cp = prob.get_val("propeller.radii_cp", units="inch")
radii = np.squeeze(prob.get_val("propeller.radii", units="inch"))
chord_cp = prob.get_val("propeller.chord_cp", units="inch")
chord = np.squeeze(prob.get_val("propeller.chord", units="inch"))
theta_cp = prob.get_val("propeller.theta_cp", units="deg")
theta = np.squeeze(prob.get_val("propeller.theta", units="deg"))

cmap = plt.get_cmap("tab10")
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.scatter(radii_cp, chord_cp, color=cmap(1), marker="o")
ax0.plot(radii, chord, color=cmap(1))
ax0.set_ylim(0.0, 5.0)
ax1.scatter(radii_cp, theta_cp, color=cmap(1), marker="o")
ax1.plot(radii, theta, color=cmap(1))
ax0.set_ylabel("chord, in.")
ax1.set_ylabel("twist, deg.")
ax1.set_xlabel("radii, in.")
fig.savefig("chord_theta_aero_only.png")
