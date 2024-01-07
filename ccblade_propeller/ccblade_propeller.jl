using CCBlade: CCBlade
using ComponentArrays: ComponentArray
using ConcreteStructs: @concrete
using ForwardDiff: ForwardDiff
using OpenMDAOCore: OpenMDAOCore, VarData, PartialsData, get_rows_cols


function get_airfoil(; af_fname, cr75, Re_exp)
    (info, Re, Mach, alpha, cl, cd) = CCBlade.parsefile(af_fname, false)

    # Extend the angle of attack with the Viterna method.
    (alpha, cl, cd) = CCBlade.viterna(alpha, cl, cd, cr75)
    af = CCBlade.AlphaAF(alpha, cl, cd, info, Re, Mach)

    # Reynolds number correction. The 0.6 factor seems to match the NACA 0012
    # drag data from airfoiltools.com.
    reynolds = CCBlade.SkinFriction(Re, Re_exp)

    # Mach number correction.
    mach = CCBlade.PrandtlGlauert()

    # Rotational stall delay correction. Need some parameters from the CL curve.
    m, alpha0 = CCBlade.linearliftcoeff(af, 1.0, 1.0)  # dummy values for Re and Mach
    # Create the Du Selig and Eggers correction.
    rotation = CCBlade.DuSeligEggers(1.0, 1.0, 1.0, m, alpha0)

    # The usual hub and tip loss correction.
    tip = CCBlade.PrandtlTipHub()

    return af, mach, reynolds, rotation, tip
end


@concrete struct BEMTRotorComp <: OpenMDAOCore.AbstractExplicitComp
    num_operating_points
    num_blades
    num_radial
    rho
    mu
    speedofsound
    airfoil_interp
    mach
    reynolds
    rotation
    tip
    compute_forwarddiffable!
    x
    y
    J
    forwarddiff_config
end


function BEMTRotorComp(; af_fname, cr75, Re_exp, num_operating_points, num_blades, num_radial, rho, mu, speedofsound, use_hubtip_losses=true)
    # Get the airfoil polar interpolator and various correction factors.
    af, mach, reynolds, rotation, tip = get_airfoil(af_fname=af_fname, cr75=cr75, Re_exp=Re_exp)

    if ! use_hubtip_losses
        tip = nothing
    end

    function compute_forwarddiffable!(y, x)
        T = eltype(x)

        # Unpack the inputs.
        Rhub = x[:Rhub]
        Rtip = x[:Rtip]
        radii = x[:radii]
        chord = x[:chord]
        theta = x[:theta]
        v = x[:v]
        omega = x[:omega]
        pitch = x[:pitch]

        # Create the CCBlade rotor struct.
        turbine = false
        precone = zero(T)
        rotor = CCBlade.Rotor(Rhub, Rtip, num_blades, precone, turbine, mach, reynolds, rotation, tip)

        # Create the CCBlade sections.
        sections = CCBlade.Section.(radii, chord, theta, Ref(af))

        # Create the CCBlade operating points.
        Vx = v
        Vy = omega.*radii
        ops = CCBlade.OperatingPoint.(Vx, Vy, rho, pitch, mu, speedofsound)

        # Solve the BEMT equations.
        outs = CCBlade.solve.(Ref(rotor), sections, ops)

        # Get the thrust and torque, then the efficiency, etc.
        # coefficients.
        thrust, torque = CCBlade.thrusttorque(rotor, sections, outs)
        eff, CT, CQ = CCBlade.nondim(thrust, torque, Vx, omega, rho, rotor, "propeller")
        if thrust > zero(T)
            figure_of_merit, CT, CP = CCBlade.nondim(thrust, torque, Vx, omega, rho, rotor, "helicopter")
        else
            figure_of_merit = zero(T)
        end

        # Put the outputs in the output array.
        y[:thrust] = thrust
        y[:torque] = torque
        y[:eff] = eff
        y[:figure_of_merit] = figure_of_merit
        y[:Np] .= getproperty.(outs, :Np)
        y[:Tp] .= getproperty.(outs, :Tp)
        y[:ui] .= getproperty.(outs, :u)
        y[:vi] .= getproperty.(outs, :v)
        y[:alpha] .= getproperty.(outs, :alpha)

        return nothing
    end

    # Initialize the input and output vectors needed by ForwardDiff.jl.
    X = ComponentArray(
        Rhub=0.0, Rtip=0.0, radii=zeros(Float64, num_radial), chord=zeros(Float64, num_radial),
        theta=zeros(Float64, num_radial), v=0.0, omega=0.0, pitch=0.0)
    Y = ComponentArray(
        thrust=0.0, torque=0.0, eff=0.0, figure_of_merit=0.0,
        Np=zeros(Float64, num_radial), Tp=zeros(Float64, num_radial),
        ui=zeros(Float64, num_radial), vi=zeros(Float64, num_radial), alpha=zeros(Float64, num_radial))
    J = Y.*X'

    # Get the JacobianConfig object, which we'll reuse each time when calling
    # the ForwardDiff.jacobian! function (apparently good for efficiency).
    config = ForwardDiff.JacobianConfig(compute_forwarddiffable!, Y, X)

    return BEMTRotorComp(num_operating_points, num_blades, num_radial, rho, mu, speedofsound, af, mach, reynolds, rotation, tip, compute_forwarddiffable!, X, Y, J, config)
end


# Need a setup function, just like a Python OpenMDAO `Component`.
function OpenMDAOCore.setup(self::BEMTRotorComp)
    num_operating_points = self.num_operating_points
    num_radial = self.num_radial

    # Declare the OpenMDAO inputs.
    input_data = Vector{VarData}()
    push!(input_data, VarData("Rhub", shape=1, val=0.2*0.0254, units="m"))
    push!(input_data, VarData("Rtip", shape=1, val=12.0*0.0254, units="m"))
    push!(input_data, VarData("radii", shape=num_radial, val=collect(range(0.2*0.0254, 12*0.0254; length=num_radial)), units="m"))
    push!(input_data, VarData("chord", shape=num_radial, val=1., units="m"))
    push!(input_data, VarData("theta", shape=num_radial, val=range(60, 20; length=num_radial).*pi/180, units="rad"))
    push!(input_data, VarData("v", shape=num_operating_points, val=1., units="m/s"))
    push!(input_data, VarData("omega", shape=num_operating_points, val=100.0, units="rad/s"))
    push!(input_data, VarData("pitch", shape=num_operating_points, val=0., units="rad"))

    # Declare the OpenMDAO outputs.
    output_data = Vector{VarData}()
    push!(output_data, VarData("thrust", shape=num_operating_points, val=1.0, units="N"))
    push!(output_data, VarData("torque", shape=num_operating_points, val=1.0, units="N*m"))
    push!(output_data, VarData("efficiency", shape=num_operating_points, val=1.0))
    push!(output_data, VarData("figure_of_merit", shape=num_operating_points, val=1.0))
    push!(output_data, VarData("theta_with_pitch", shape=(num_operating_points, num_radial), val=1.0, units="rad"))
    push!(output_data, VarData("Np", shape=(num_operating_points, num_radial), val=1.0, units="N/m"))
    push!(output_data, VarData("Tp", shape=(num_operating_points, num_radial), val=1.0, units="N/m"))
    push!(output_data, VarData("ui", shape=(num_operating_points, num_radial), val=1.0, units="m/s"))
    push!(output_data, VarData("vi", shape=(num_operating_points, num_radial), val=1.0, units="m/s"))
    push!(output_data, VarData("alpha", shape=(num_operating_points, num_radial), val=1.0, units="rad"))

    # Declare the OpenMDAO partial derivatives.
    ss_sizes = Dict(:i=>num_operating_points, :j=>num_radial, :k=>1)
    partials_data = Vector{PartialsData}()

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:k])
    push!(partials_data, PartialsData("Np", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Np", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Tp", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Tp", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("ui", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("ui", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("vi", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("vi", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("alpha", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("alpha", "Rtip", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:j])
    push!(partials_data, PartialsData("Np", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Np", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Np", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Tp", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Tp", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Tp", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("ui", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("ui", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("ui", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("vi", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("vi", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("vi", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("alpha", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("alpha", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("alpha", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("theta_with_pitch", "theta", rows=rows, cols=cols, val=1.0))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i])
    push!(partials_data, PartialsData("Np", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Np", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Np", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Tp", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Tp", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("Tp", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("ui", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("ui", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("ui", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("vi", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("vi", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("vi", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("alpha", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("alpha", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("alpha", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("theta_with_pitch", "pitch", rows=rows, cols=cols, val=1.0))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:k])
    push!(partials_data, PartialsData("thrust", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "Rtip", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:j])
    push!(partials_data, PartialsData("thrust", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "theta", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i])
    push!(partials_data, PartialsData("thrust", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "pitch", rows=rows, cols=cols))

    return input_data, output_data, partials_data
end


function OpenMDAOCore.compute!(self::BEMTRotorComp, inputs, outputs)
    num_operating_points = self.num_operating_points
    X, Y = self.x, self.y
    X[:Rhub] = inputs["Rhub"][1]
    X[:Rtip] = inputs["Rtip"][1]
    X[:radii] .= inputs["radii"]
    X[:chord] .= inputs["chord"]
    X[:theta] .= inputs["theta"]
    for n in 1:num_operating_points
        X[:v] = inputs["v"][n]
        X[:omega] = inputs["omega"][n]
        X[:pitch] = inputs["pitch"][n]

        self.compute_forwarddiffable!(Y, X)

        outputs["thrust"][n] = Y[:thrust]
        outputs["torque"][n] = Y[:torque]
        outputs["efficiency"][n] = Y[:eff]
        outputs["figure_of_merit"][n] = Y[:figure_of_merit]
        outputs["Np"][n, :] .= Y[:Np]
        outputs["Tp"][n, :] .= Y[:Tp]
        outputs["ui"][n, :] .= Y[:ui]
        outputs["vi"][n, :] .= Y[:vi]
        outputs["alpha"][n, :] .= Y[:alpha]

        outputs["theta_with_pitch"][n, :] .= inputs["theta"] .+ inputs["pitch"][n]
    end


    return nothing
end

# Now for the big one: the `linearize!` function will calculate the derivatives
# of the BEMT component residuals wrt the inputs and outputs. We'll use the
# Julia package ForwardDiff.jl to actually calculate the derivatives.
function OpenMDAOCore.compute_partials!(self::BEMTRotorComp, inputs, partials)
    # Unpack the options we'll need.
    num_operating_points = self.num_operating_points
    num_radial = self.num_radial

    # Working arrays and configuration for ForwardDiff's Jacobian routine.
    X = self.x
    Y = self.y
    J = self.J
    config = self.forwarddiff_config

    # These need to be transposed because of the differences in array layout
    # between NumPy and Julia. When I declare the partials above, they get set up
    # on the OpenMDAO side in a shape=(num_operating_points, num_radial), and
    # are then flattened. That gets passed to Julia. Since Julia uses column
    # major arrays, we have to reshape the array with the indices reversed, then
    # transpose them.

    dNp_dRhub = transpose(reshape(partials["Np", "Rhub"], num_radial, num_operating_points))
    dNp_dRtip = transpose(reshape(partials["Np", "Rtip"], num_radial, num_operating_points))
    dNp_dradii = transpose(reshape(partials["Np", "radii"], num_radial, num_operating_points))
    dNp_dchord = transpose(reshape(partials["Np", "chord"], num_radial, num_operating_points))
    dNp_dtheta = transpose(reshape(partials["Np", "theta"], num_radial, num_operating_points))
    dNp_dv = transpose(reshape(partials["Np", "v"], num_radial, num_operating_points))
    dNp_domega = transpose(reshape(partials["Np", "omega"], num_radial, num_operating_points))
    dNp_dpitch = transpose(reshape(partials["Np", "pitch"], num_radial, num_operating_points))

    dTp_dRhub = transpose(reshape(partials["Tp", "Rhub"], num_radial, num_operating_points))
    dTp_dRtip = transpose(reshape(partials["Tp", "Rtip"], num_radial, num_operating_points))
    dTp_dradii = transpose(reshape(partials["Tp", "radii"], num_radial, num_operating_points))
    dTp_dchord = transpose(reshape(partials["Tp", "chord"], num_radial, num_operating_points))
    dTp_dtheta = transpose(reshape(partials["Tp", "theta"], num_radial, num_operating_points))
    dTp_dv = transpose(reshape(partials["Tp", "v"], num_radial, num_operating_points))
    dTp_domega = transpose(reshape(partials["Tp", "omega"], num_radial, num_operating_points))
    dTp_dpitch = transpose(reshape(partials["Tp", "pitch"], num_radial, num_operating_points))

    dui_dRhub = transpose(reshape(partials["ui", "Rhub"], num_radial, num_operating_points))
    dui_dRtip = transpose(reshape(partials["ui", "Rtip"], num_radial, num_operating_points))
    dui_dradii = transpose(reshape(partials["ui", "radii"], num_radial, num_operating_points))
    dui_dchord = transpose(reshape(partials["ui", "chord"], num_radial, num_operating_points))
    dui_dtheta = transpose(reshape(partials["ui", "theta"], num_radial, num_operating_points))
    dui_dv = transpose(reshape(partials["ui", "v"], num_radial, num_operating_points))
    dui_domega = transpose(reshape(partials["ui", "omega"], num_radial, num_operating_points))
    dui_dpitch = transpose(reshape(partials["ui", "pitch"], num_radial, num_operating_points))

    dvi_dRhub = transpose(reshape(partials["vi", "Rhub"], num_radial, num_operating_points))
    dvi_dRtip = transpose(reshape(partials["vi", "Rtip"], num_radial, num_operating_points))
    dvi_dradii = transpose(reshape(partials["vi", "radii"], num_radial, num_operating_points))
    dvi_dchord = transpose(reshape(partials["vi", "chord"], num_radial, num_operating_points))
    dvi_dtheta = transpose(reshape(partials["vi", "theta"], num_radial, num_operating_points))
    dvi_dv = transpose(reshape(partials["vi", "v"], num_radial, num_operating_points))
    dvi_domega = transpose(reshape(partials["vi", "omega"], num_radial, num_operating_points))
    dvi_dpitch = transpose(reshape(partials["vi", "pitch"], num_radial, num_operating_points))

    dalpha_dRhub = transpose(reshape(partials["alpha", "Rhub"], num_radial, num_operating_points))
    dalpha_dRtip = transpose(reshape(partials["alpha", "Rtip"], num_radial, num_operating_points))
    dalpha_dradii = transpose(reshape(partials["alpha", "radii"], num_radial, num_operating_points))
    dalpha_dchord = transpose(reshape(partials["alpha", "chord"], num_radial, num_operating_points))
    dalpha_dtheta = transpose(reshape(partials["alpha", "theta"], num_radial, num_operating_points))
    dalpha_dv = transpose(reshape(partials["alpha", "v"], num_radial, num_operating_points))
    dalpha_domega = transpose(reshape(partials["alpha", "omega"], num_radial, num_operating_points))
    dalpha_dpitch = transpose(reshape(partials["alpha", "pitch"], num_radial, num_operating_points))

    dthrust_dRhub = partials["thrust", "Rhub"]
    dthrust_dRtip = partials["thrust", "Rtip"]
    dtorque_dRhub = partials["torque", "Rhub"]
    dtorque_dRtip = partials["torque", "Rtip"]
    defficiency_dRhub = partials["efficiency", "Rhub"]
    defficiency_dRtip = partials["efficiency", "Rtip"]
    dfigure_of_merit_dRhub = partials["figure_of_merit", "Rhub"]
    dfigure_of_merit_dRtip = partials["figure_of_merit", "Rtip"]

    dthrust_dradii = transpose(reshape(partials["thrust", "radii"], num_radial, num_operating_points))
    dthrust_dchord = transpose(reshape(partials["thrust", "chord"], num_radial, num_operating_points))
    dthrust_dtheta = transpose(reshape(partials["thrust", "theta"], num_radial, num_operating_points))
    dtorque_dradii = transpose(reshape(partials["torque", "radii"], num_radial, num_operating_points))
    dtorque_dchord = transpose(reshape(partials["torque", "chord"], num_radial, num_operating_points))
    dtorque_dtheta = transpose(reshape(partials["torque", "theta"], num_radial, num_operating_points))
    defficiency_dradii = transpose(reshape(partials["efficiency", "radii"], num_radial, num_operating_points))
    defficiency_dchord = transpose(reshape(partials["efficiency", "chord"], num_radial, num_operating_points))
    defficiency_dtheta = transpose(reshape(partials["efficiency", "theta"], num_radial, num_operating_points))
    dfigure_of_merit_dradii = transpose(reshape(partials["figure_of_merit", "radii"], num_radial, num_operating_points))
    dfigure_of_merit_dchord = transpose(reshape(partials["figure_of_merit", "chord"], num_radial, num_operating_points))
    dfigure_of_merit_dtheta = transpose(reshape(partials["figure_of_merit", "theta"], num_radial, num_operating_points))

    dthrust_dv = partials["thrust", "v"]
    dthrust_domega = partials["thrust", "omega"]
    dthrust_dpitch = partials["thrust", "pitch"]
    dtorque_dv = partials["torque", "v"]
    dtorque_domega = partials["torque", "omega"]
    dtorque_dpitch = partials["torque", "pitch"]
    defficiency_dv = partials["efficiency", "v"]
    defficiency_domega = partials["efficiency", "omega"]
    defficiency_dpitch = partials["efficiency", "pitch"]
    dfigure_of_merit_dv = partials["figure_of_merit", "v"]
    dfigure_of_merit_domega = partials["figure_of_merit", "omega"]
    dfigure_of_merit_dpitch = partials["figure_of_merit", "pitch"]

    # Unpack the inputs.
    Rhub = inputs["Rhub"][1]
    Rtip = inputs["Rtip"][1]
    radii = inputs["radii"]
    chord = inputs["chord"]
    theta = inputs["theta"]
    v = inputs["v"]
    omega = inputs["omega"]
    pitch = inputs["pitch"]

    X[:Rhub] = inputs["Rhub"][1]
    X[:Rtip] = inputs["Rtip"][1]
    X[:radii] .= inputs["radii"]
    X[:chord] .= inputs["chord"]
    X[:theta] .= inputs["theta"]
    for n in 1:num_operating_points
        # Put the inputs into the input array for ForwardDiff.
        X[:v] = v[n]
        X[:omega] = omega[n]
        X[:pitch] = pitch[n]

        # Get the Jacobian.
        ForwardDiff.jacobian!(J, self.compute_forwarddiffable!, Y, X, config)

        for r in 1:num_radial
            dNp_dradii[n, r] = J[:Np, :radii][r, r]
            dNp_dchord[n, r] = J[:Np, :chord][r, r]
            dNp_dtheta[n, r] = J[:Np, :theta][r, r]

            dTp_dradii[n, r] = J[:Tp, :radii][r, r]
            dTp_dchord[n, r] = J[:Tp, :chord][r, r]
            dTp_dtheta[n, r] = J[:Tp, :theta][r, r]

            dui_dradii[n, r] = J[:ui, :radii][r, r]
            dui_dchord[n, r] = J[:ui, :chord][r, r]
            dui_dtheta[n, r] = J[:ui, :theta][r, r]

            dvi_dradii[n, r] = J[:vi, :radii][r, r]
            dvi_dchord[n, r] = J[:vi, :chord][r, r]
            dvi_dtheta[n, r] = J[:vi, :theta][r, r]

            dalpha_dradii[n, r] = J[:alpha, :radii][r, r]
            dalpha_dchord[n, r] = J[:alpha, :chord][r, r]
            dalpha_dtheta[n, r] = J[:alpha, :theta][r, r]
        end

        dNp_dRhub[n, :] .= J[:Np, :Rhub]
        dNp_dRtip[n, :] .= J[:Np, :Rtip]
        dNp_dv[n, :] .= J[:Np, :v]
        dNp_domega[n, :] .= J[:Np, :omega]
        dNp_dpitch[n, :] .= J[:Np, :pitch]

        dTp_dRhub[n, :] .= J[:Tp, :Rhub]
        dTp_dRtip[n, :] .= J[:Tp, :Rtip]
        dTp_dv[n, :] .= J[:Tp, :v]
        dTp_domega[n, :] .= J[:Tp, :omega]
        dTp_dpitch[n, :] .= J[:Tp, :pitch]

        dui_dRhub[n, :] .= J[:ui, :Rhub]
        dui_dRtip[n, :] .= J[:ui, :Rtip]
        dui_dv[n, :] .= J[:ui, :v]
        dui_domega[n, :] .= J[:ui, :omega]
        dui_dpitch[n, :] .= J[:ui, :pitch]

        dvi_dRhub[n, :] .= J[:vi, :Rhub]
        dvi_dRtip[n, :] .= J[:vi, :Rtip]
        dvi_dv[n, :] .= J[:vi, :v]
        dvi_domega[n, :] .= J[:vi, :omega]
        dvi_dpitch[n, :] .= J[:vi, :pitch]

        dalpha_dRhub[n, :] .= J[:alpha, :Rhub]
        dalpha_dRtip[n, :] .= J[:alpha, :Rtip]
        dalpha_dv[n, :] .= J[:alpha, :v]
        dalpha_domega[n, :] .= J[:alpha, :omega]
        dalpha_dpitch[n, :] .= J[:alpha, :pitch]

        dthrust_dRhub[n] = J[:thrust, :Rhub]
        dthrust_dRtip[n] = J[:thrust, :Rtip]
        dthrust_dradii[n, :] .= J[:thrust, :radii]
        dthrust_dchord[n, :] .= J[:thrust, :chord]
        dthrust_dtheta[n, :] .= J[:thrust, :theta]
        dthrust_dv[n] = J[:thrust, :v]
        dthrust_domega[n] = J[:thrust, :omega]
        dthrust_dpitch[n] = J[:thrust, :pitch]

        dtorque_dRhub[n] = J[:torque, :Rhub]
        dtorque_dRtip[n] = J[:torque, :Rtip]
        dtorque_dradii[n, :] .= J[:torque, :radii]
        dtorque_dchord[n, :] .= J[:torque, :chord]
        dtorque_dtheta[n, :] .= J[:torque, :theta]
        dtorque_dv[n] = J[:torque, :v]
        dtorque_domega[n] = J[:torque, :omega]
        dtorque_dpitch[n] = J[:torque, :pitch]

        defficiency_dRhub[n] = J[:eff, :Rhub]
        defficiency_dRtip[n] = J[:eff, :Rtip]
        defficiency_dradii[n, :] .= J[:eff, :radii]
        defficiency_dchord[n, :] .= J[:eff, :chord]
        defficiency_dtheta[n, :] .= J[:eff, :theta]
        defficiency_dv[n] = J[:eff, :v]
        defficiency_domega[n] = J[:eff, :omega]
        defficiency_dpitch[n] = J[:eff, :pitch]

        dfigure_of_merit_dRhub[n] = J[:figure_of_merit, :Rhub]
        dfigure_of_merit_dRtip[n] = J[:figure_of_merit, :Rtip]
        dfigure_of_merit_dradii[n, :] .= J[:figure_of_merit, :radii]
        dfigure_of_merit_dchord[n, :] .= J[:figure_of_merit, :chord]
        dfigure_of_merit_dtheta[n, :] .= J[:figure_of_merit, :theta]
        dfigure_of_merit_dv[n] = J[:figure_of_merit, :v]
        dfigure_of_merit_domega[n] = J[:figure_of_merit, :omega]
        dfigure_of_merit_dpitch[n] = J[:figure_of_merit, :pitch]
    end

    return nothing
end
