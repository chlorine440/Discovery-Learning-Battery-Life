import pybamm
import numpy as np

def simulator(params, current, t_sim):
   options = {'SEI': 'constant',
            'SEI film resistance': 'distributed'}
   model = pybamm.lithium_ion.DFN(options=options)
   params["Current function [A]"] = current
   parameter_values = pybamm.ParameterValues(params)
   solver = pybamm.CasadiSolver(mode="safe", rtol=1e-3, atol=1e-3, return_solution_if_failed_early=True)
   var_pts = {
   "x_n": 20,  # negative electrode thickness direction mesh size
   "x_s": 10,  # separator thickness direction mesh size
   "x_p": 20,  # positive electrode thickness direction mesh size
   "r_n": 20,  # negative particle radius direction mesh size
   "r_p": 20,  # positive particle radius direction mesh size
   }
   sim = pybamm.Simulation(model, parameter_values=parameter_values,solver=solver, var_pts=var_pts)
   t_eval = np.linspace(0, t_sim*3, num=1000)
   try:
      sim.solve(t_eval=t_eval)
      return sim.solution
   except:
      print('Extreme case error!')
      return 'Extreme case error!'
    
    
    
    
    
    
    
    
    
    