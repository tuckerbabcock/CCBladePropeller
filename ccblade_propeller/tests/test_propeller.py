import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from ccblade_propeller.propeller import Propeller


class TestPropeller(unittest.TestCase):
    def test_propeller_run(self):
        prob = om.Problem()

        prob.model.add_subsystem('prop',
                                 Propeller(D=24.0*0.0254),
                                 promotes=['*'])

        prob.setup()

        prob['v'] = 37.43267316
        prob['rpm'] = 7200
        # prob['omega'] = 753.98223686

        prob.run_model()
        prob.model.list_inputs(units=True, prom_name=True)
        prob.model.list_outputs(residuals=True, units=True, prom_name=True)

        thrust = prob['thrust'][0]
        self.assertAlmostEqual(thrust, 97.51858916)

    def test_propeller_partials(self):
        prob = om.Problem()

        prob.model.add_subsystem('prop',
                                 Propeller(D=24.0*0.0254),
                                 promotes=['*'])

        prob.setup()

        prob['v'] = 37.43267316
        prob['rpm'] = 7200
        # prob['omega'] = 753.98223686

        prob.run_model()
        data = prob.check_partials(form="central")
        assert_check_partials(data)


unittest.main()
