import math
import numpy as np
import matplotlib.pyplot as plt

class pwl_approx:
  
  # --------------------------------------------------- #
  def __init__(self, op_name, region1_nbits, region3_nbits, start_value, end_value):
    self.op_name = op_name
    self.region1_nbits = region1_nbits
    self.region3_nbits = region3_nbits
    self.start_value = start_value
    self.end_value = end_value
    self.lut1=[]
    self.lut2=[]

  # --------------------------------------------------- #
  def create_luts(self):
    cnt=0
    self.lut1 = []
    self.lut2 = []
    
    step1=(self.breakpoint1 - self.startpoint) / (pow(2,self.region1_nbits)-1)
    step2=(self.breakpoint3 - self.breakpoint2) / (pow(2,self.region3_nbits)-1)

    for ii in range(pow(2,self.region1_nbits)):  # create LUT for the first region
      value = self.startpoint + ii * step1
      self.lut1.append(exact_operation(self.op_name, value))

    for ii in range(pow(2,self.region3_nbits)):  # create LUT for the second region
      value = self.breakpoint2 + ii * step2
      #print('   ', ii, value)
      self.lut2.append(exact_operation(self.op_name, value))

    t = np.linspace(-10, 10, 100)
    sig = []
    for jj in range(len(t)):
      sig.append(exact_operation(self.op_name, t[jj])) #1 / (1 + np.exp(-t))
    #print(sig[1:20])


    '''
    fig, ax = plt.subplots()
    ax.axvline(color="grey")
    #ax.axline((0, 0.5), slope=0.25, color="black", linestyle=(0, (5, 5)))
    ax.plot(t, sig, linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    ax.set(xlim=(-3, 3), xlabel="t")
    ax.legend(fontsize=14)
    plt.show()
    '''

    '''    
    for jj in range(pow(2,self.region3_nbits)):
      print(break_point2+jj*step2, " : ", lut2[jj])

    print('----', 'tanh(7.2)', " : ", self.operation(7.2))
    '''
  
  # --------------------------------------------------- #
  def set_breakpoints(self, startpoint, breakpoint1, breakpoint2, breakpoint3, endpoint):
    self.breakpoint1 = breakpoint1
    self.breakpoint2 = breakpoint2
    self.breakpoint3 = breakpoint3
    self.startpoint = startpoint
    self.endpoint = endpoint

  # --------------------------------------------------- #
  def approx_operation(self, operand):
    step1=(self.breakpoint1 - self.startpoint) / (pow(2,self.region1_nbits)-1)
    step2=(self.breakpoint3 - self.breakpoint2) / (pow(2,self.region3_nbits)-1)
    if operand < self.startpoint:
      return self.lut1[0]
    
    elif operand < self.breakpoint1:
      ind_low = math.floor((operand-self.startpoint)/(self.breakpoint1-self.startpoint) * (pow(2,self.region1_nbits)-1))
      ind_high = ind_low + 1
      correction =  (self.lut1[ind_high] - self.lut1[ind_low]) / (ind_high*step1 - ind_low*step1) * (operand - self.startpoint - ind_low*step1)# lineear interpolation
      #print("operand: ", operand,  )
      return self.lut1[ind_low] + correction
    
    elif operand < self.breakpoint2:
      correction =  (self.lut2[0] - self.lut1[pow(2,self.region1_nbits)-1]) / (self.breakpoint2 - self.breakpoint1) * (operand - self.breakpoint1)
      return self.lut1[pow(2,self.region1_nbits)-1] + correction
    
    elif operand < self.breakpoint3:
      ind_low = math.floor((operand-self.breakpoint2)/(self.breakpoint3-self.breakpoint2) * (pow(2,self.region3_nbits)-1))
      ind_high = ind_low + 1
      #print('      hhh ',ind_high, '   ', self.lut2 )
      correction =  (self.lut2[ind_high] - self.lut2[ind_low]) / (ind_high*step2 - ind_low*step2) * (operand - self.breakpoint2 - ind_low*step2)   # lineear interpolation
      #print("operand: ", operand,  )
      return self.lut2[ind_low] + correction
    
    else:
      return self.end_value
    
  # --------------------------------------------------- #
  def error_analysis(self, range, plot_error=False):
    mre = 0
    mae = 0
    N = 0
    error_array = []
    exact_result = []
    approx_result = []
    ind = []
    for operand in np.arange(range[0], range[1], 1/pow(2, 17)):
      #print('kk  ', operand, '    ', abs((object.approx_result(operand) - object.operation(operand))), '      mre:', mre)
      exact_result.append(exact_operation(self.op_name, operand))
      approx_result.append(self.approx_operation(operand))
      error_array.append(abs(approx_result[-1] - exact_result[-1]))
      ind.append(operand)
      mre = mre + pow((approx_result[-1] - exact_result[-1]),2)
      mae = mae + abs(approx_result[-1] - exact_result[-1])
      N = N+1
    mre = mre / N
    mae = mae / N

    if plot_error == True:
      fig, ax = plt.subplots()
      ax.axvline(color="grey")
      ax.plot(ind, error_array, linewidth=2, label="Absolute Error")
      #ax.plot(ind, exact_result, 'r', linewidth=2, label="Exact")
      #ax.plot(ind, approx_result, 'b', linewidth=2, label="Approximate result")
      ax.axvspan(0, self.breakpoint1, alpha=0.2)
      ax.axvspan(self.breakpoint1, self.breakpoint2, color='yellow', alpha=0.2)
      ax.axvspan(self.breakpoint2, self.breakpoint3, color='red', alpha=0.2)
      ax.set(xlim=(0, range[1]), xlabel="Endpoint")
      ax.set(ylim=(0, 0.2), ylabel=r"Absolute Error")#"Mean Absolute Error (x1000)")
      ax.legend(fontsize=14)
      plt.grid(True)
      plt.show()

    return [mre, mae]

  # --------------------------------------------------- #
  def detailed_error_analysis(self):
    N = 0
    resolution = 1/pow(2, 17)
    reg1_error = [0, 0]
    print('-- SP: ', self.startpoint, ' BP1: ', self.breakpoint1, ' BP2: ', self.breakpoint2, ' BP3: ', self.breakpoint3 , ' EP: ', self.endpoint)
    # Region 1: from Startpoint to BP1
    for operand in np.arange(self.startpoint, self.breakpoint1, resolution):
      exact_result = exact_operation(self.op_name, operand)
      approx_result = self.approx_operation(operand)
      reg1_error[0] = reg1_error[0] + abs((approx_result - exact_result) / exact_result) # MRE
      reg1_error[1] = reg1_error[1] + abs(approx_result - exact_result) # MAE
      N = N + 1
    reg1_error[0] = reg1_error[0] / N
    reg1_error[1] = reg1_error[1] / N

    # Region 2: from BP1 to BP2
    N = 0
    reg2_error = [0, 0]
    for operand in np.arange(self.breakpoint1+resolution, self.breakpoint2, resolution):
      exact_result = exact_operation(self.op_name, operand)
      approx_result = self.approx_operation(operand)
      reg2_error[0] = reg2_error[0] + abs((approx_result - exact_result) / exact_result) # MRE
      reg2_error[1] = reg2_error[1] + abs(approx_result - exact_result) # MAE
      N = N + 1
    reg2_error[0] = reg2_error[0] / N
    reg2_error[1] = reg2_error[1] / N

    # Region 3: from BP2 to BP3
    N = 0
    reg3_error = [0, 0]
    for operand in np.arange(self.breakpoint2+resolution, self.breakpoint3, resolution):
      exact_result = exact_operation(self.op_name, operand)
      approx_result = self.approx_operation(operand)
      reg3_error[0] = reg3_error[0] + abs((approx_result - exact_result) / exact_result) # MRE
      reg3_error[1] = reg3_error[1] + abs(approx_result - exact_result) # MAE
      N = N + 1
    reg3_error[0] = reg3_error[0] / N
    reg3_error[1] = reg3_error[1] / N

    # Region 4: from BP3 to Endpoint
    N = 0
    reg4_error = [0, 0]
    for operand in np.arange(self.breakpoint3+resolution, self.endpoint, resolution):
      exact_result = exact_operation(self.op_name, operand)
      approx_result = self.approx_operation(operand)
      reg4_error[0] = reg4_error[0] + abs((approx_result - exact_result) / exact_result) # MRE
      reg4_error[1] = reg4_error[1] + abs(approx_result - exact_result) # MAE
      N = N + 1
    reg4_error[0] = reg4_error[0] / N
    reg4_error[1] = reg4_error[1] / N

    total_mre = (reg1_error[0] + reg2_error[0] + reg3_error[0] + reg4_error[0])/4
    total_mae = (reg1_error[1] + reg2_error[1] + reg3_error[1] + reg4_error[1])/4

    return [reg1_error, reg2_error, reg3_error, reg4_error, total_mre, total_mae]


# --------------------------------------------------- #
def exact_operation(op_name, operand):
  match op_name:
      case "tanh":
          return math.tanh(operand)
      case "sigmoid":
          return 1/(1 + np.exp(-operand))
      case "der_tanh":
          return 1-math.tanh(operand)**2
      case "der_sigmoid":
          sigmoid = 1/(1 + np.exp(-operand))
          return sigmoid * (1 - sigmoid)
      case "2nd_der_tanh":
        return -2 * math.tanh(operand) * (1 - math.tanh(operand)**2)
      case "2nd_der_sigmoid":
          return 0 # TO FIX
    
