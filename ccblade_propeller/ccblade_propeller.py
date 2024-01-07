import os

# Create a new Julia module that will hold all the Julia code imported into this Python module.
import juliacall; jl = juliacall.newmodule("CCBladePropeller")

d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "ccblade_propeller.jl"))
# Now we have access to everything in `ccblade_propeller.jl`

BEMTRotorComp = jl.BEMTRotorComp
