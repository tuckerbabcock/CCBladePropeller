import numpy as np

import openmdao.api as om
from openmdao.utils.spline_distributions import cell_centered

from omjlcomps import JuliaExplicitComp

from .ccblade_propeller import BEMTRotorComp


class Propeller(om.Group):
    def initialize(self):
        self.options.declare('D', default=1, desc='Propeller diameter')
        self.options.declare('n_b', default=3, desc='Number of blades')
        self.options.declare('n_cp', default=6, desc='Number of control points')
        self.options.declare('n_r', default=30, desc='Number of radial points')
        self.options.declare('af_fname', default="af_data/xf-n0012-il-500000.dat", desc='Airfoil data file')

    def setup(self):
        num_cp = self.options['n_cp']
        num_radial = self.options['n_r']
        num_operating_points = 1

        # Not sure about the atmospheric conditions, so I'll just use the ICAO standard
        # atmosphere at sealevel: (https://www.engineeringtoolbox.com/international-standard-atmosphere-d_985.html)
        p0 = 101325.0
        T0 = 273.15 + 15.0
        gam = 1.4
        speedofsound = np.sqrt(gam*287.058*T0)
        rho0 = gam*p0/speedofsound**2
        mu = rho0*1.461e-5

        # Operating parameters for this case.
        # rpm = 7200.0
        # M_infty = 0.11
        # v = M_infty*speedofsound  # axial velocity in m/sec.
        # omega = 2*np.pi/60.0*rpm  # propeller rotation rate in rad/sec.
        pitch = 0.0

        # D = 24.0*0.0254  # Diameter in meters.
        D = self.options['D']
        Rtip = 0.5*D
        Rhub = 0.2*Rtip  # Just guessing on the hub diameter.
        radii_cp0 = np.linspace(Rhub, Rtip, num_cp)

        c = 0.1   # Constant chord in meters.
        chord_cp0 = c*np.ones(num_cp)

        P = 0.4  # Pitch in meters (used in the twist distribution).
        theta_cp0 = np.arctan(P/(np.pi*D*radii_cp0/Rtip))

        comp = om.IndepVarComp()
        comp.add_output("Rhub", val=Rhub, units="m")
        comp.add_output("Rtip", val=Rtip, units="m")
        comp.add_output("radii_cp", val=radii_cp0, units="m")
        comp.add_output("chord_cp", val=chord_cp0, units="m")
        comp.add_output("theta_cp", val=theta_cp0, units="rad")
        # comp.add_output("v", val=v, shape=num_operating_points, units="m/s")
        # comp.add_output("omega", val=omega,
        #                 shape=num_operating_points, units="rad/s")
        comp.add_output("pitch", val=pitch,
                        shape=num_operating_points, units="rad")
        self.add_subsystem("prop_inputs", comp, promotes_outputs=["*"])

        self.add_subsystem('propeller_rpm',
                           om.ExecComp('omega = 2*pi/60.0*rpm',
                                       omega={'units': 'rad/s'},
                                       rpm={'units': 'rpm'}),
                           promotes=['*'])

        x_cp = np.linspace(0.0, 1.0, num_cp)
        x_interp = cell_centered(num_radial, 0.0, 1.0)
        interp_options = {"delta_x": 0.1}
        comp = om.SplineComp(
            method="akima", interp_options=interp_options, x_cp_val=x_cp, x_interp_val=x_interp)
        comp.add_spline(y_cp_name="radii_cp",
                        y_interp_name="radii", y_units="m")
        comp.add_spline(y_cp_name="chord_cp",
                        y_interp_name="chord", y_units="m")
        comp.add_spline(y_cp_name="theta_cp",
                        y_interp_name="theta", y_units="rad")
        self.add_subsystem("akima_comp", comp,
                           promotes_inputs=["radii_cp",
                                            "chord_cp",
                                            "theta_cp"],
                           promotes_outputs=["radii",
                                             "chord",
                                             "theta"])

        n_b = self.options['n_b']
        af_fname = self.options['af_fname']
        comp = JuliaExplicitComp(jlcomp=BEMTRotorComp(af_fname=af_fname,
                                                      cr75=c/Rtip,
                                                      Re_exp=0.6,
                                                      num_operating_points=num_operating_points,
                                                      num_blades=n_b,
                                                      num_radial=num_radial,
                                                      rho=rho0,
                                                      mu=mu,
                                                      speedofsound=speedofsound))

        self.add_subsystem("bemt_rotor_comp",
                           comp,
                           promotes_inputs=["Rhub",
                                            "Rtip",
                                            "radii",
                                            "chord",
                                            "theta",
                                            "v",
                                            "omega",
                                            "pitch"],
                           promotes_outputs=["thrust",
                                             "torque",
                                             "efficiency"])

        self.add_subsystem("rotor_power",
                           om.ExecComp("power_in = torque * omega",
                                       power_in={'units': 'W'},
                                       torque={'units': 'N*m'},
                                       omega={'units': 'rad/s'}),
                           promotes=['*'])
    