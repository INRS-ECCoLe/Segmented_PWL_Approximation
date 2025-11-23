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
    plot_with_value_curves = True
    mse = 0
    mae = 0
    N = 0
    error_array = []
    exact_result = []
    approx_result = []
    ind = []
    for operand in np.arange(range[0], range[1], 1/pow(2, 17)):
      #print('kk  ', operand, '    ', abs((object.approx_result(operand) - object.operation(operand))), '      mse:', mse)
      exact_result.append(exact_operation(self.op_name, operand))
      approx_result.append(self.approx_operation(operand))
      error_array.append(abs(approx_result[-1] - exact_result[-1]))
      ind.append(operand)
      mse = mse + pow((approx_result[-1] - exact_result[-1]),2)
      mae = mae + abs(approx_result[-1] - exact_result[-1])
      N = N+1
    mse = mse / N
    mae = mae / N

    if plot_error == True:
      fig, ax = plt.subplots(figsize=(6, 5))
      ax.axvline(color="grey")
      ax.plot(ind, error_array, linewidth=2, label="Absolute Error")
      if plot_with_value_curves == True:
        ax.plot(ind, exact_result, 'g', linewidth=2.5, label="Exact $tanh'(x)$")
        ax.plot(ind, approx_result, 'red', linewidth=1.5, linestyle='dashed', label="Approximate $tanh'(x)$")
        ylimit = max(exact_result)
      else:
        ylimit = 0.05
      plt.text(self.breakpoint1/2, 0.98*ylimit, 'Region 1', rotation=90, fontsize=12, color='gray', ha='center', va='top' )
      ax.axvspan(0, self.breakpoint1, alpha=0.2)
      plt.text((self.breakpoint1+self.breakpoint2)/2, 0.98*ylimit, 'Region 2', rotation=90, fontsize=12, color='gray', ha='center', va='top' )
      #plt.text(self.breakpoint1, 1.04*ylimit, 'BP1', fontsize=9, color='black', ha='center', va='top' )
      ax.axvspan(self.breakpoint1, self.breakpoint2, color='yellow', alpha=0.2)
      plt.text((self.breakpoint2+self.breakpoint3)/2, 0.98*ylimit, 'Region 3', rotation=90, fontsize=12, color='gray', ha='center', va='top' )
      #plt.text(self.breakpoint2, 1.04*ylimit, 'BP2', fontsize=9, color='black', ha='center', va='top' )
      ax.axvspan(self.breakpoint2, self.breakpoint3, color='red', alpha=0.2)
      plt.text((self.breakpoint3+self.endpoint)/2, 0.98*ylimit, 'Region 4', rotation=90, fontsize=12, color='gray', ha='center', va='top' )
      #plt.text(self.breakpoint3, 1.04*ylimit, 'BP3', fontsize=9, color='black', ha='center', va='top' )
      ax.axvspan(self.breakpoint3, self.endpoint, color='gray', alpha=0.1)
      ax.tick_params(axis='both', labelsize=12)
      ax.set(xlim=(0, range[1]), ylim=(0, ylimit))
      ax.set_xlabel(r'Input, x', fontsize=14)
      ax.set_ylabel(r"$tanh'(x)$", fontsize=14)
      #ax.set(ylim=(0, ylimit), ylabel=r"Absolute Error")#"Mean Absolute Error (x1000)")
      plt.title(f"BP1={self.breakpoint1:.2f} , BP2={self.breakpoint2:.2f} , BP3={self.breakpoint3:.2f}", pad=1, fontsize=14)
      if plot_with_value_curves == True:
        # reordering the labels
        handles, labels = plt.gca().get_legend_handles_labels()
        # specify order
        order = [1, 2, 0]
        plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=12, loc='right')
      else:
        ax.legend(fontsize=12)
      plt.grid(True)
      plt.show()

    return [mse, mae]

  # --------------------------------------------------- #
  def detailed_error_analysis(self):
    N = 0
    resolution = 1/pow(2, 16)
    reg1_error = [0, 0]
    print('    * BP1: ', self.breakpoint1, ' BP2: ', self.breakpoint2, ' BP3: ', self.breakpoint3 , ' EP: ', self.endpoint)
    # Region 1: from Startpoint to BP1
    for operand in np.arange(self.startpoint, self.breakpoint1, resolution):
      exact_result = exact_operation(self.op_name, operand)
      approx_result = self.approx_operation(operand)
      reg1_error[0] = reg1_error[0] + pow((approx_result - exact_result) , 2) # MSE
      reg1_error[1] = reg1_error[1] + abs(approx_result - exact_result) # MAE
      N = N + 1
    if (N == 0):
      reg1_error[0] = 0
      reg1_error[1] = 0
    else:
      reg1_error[0] = reg1_error[0] / N
      reg1_error[1] = reg1_error[1] / N

    # Region 2: from BP1 to BP2
    N = 0
    reg2_error = [0, 0]
    for operand in np.arange(self.breakpoint1+resolution, self.breakpoint2, resolution):
      exact_result = exact_operation(self.op_name, operand)
      approx_result = self.approx_operation(operand)
      reg2_error[0] = reg2_error[0] + pow((approx_result - exact_result) , 2) # MSE
      reg2_error[1] = reg2_error[1] + abs(approx_result - exact_result) # MAE
      N = N + 1
    if (N == 0):
      reg2_error[0] = 0
      reg2_error[1] = 0
    else:
      reg2_error[0] = reg2_error[0] / N
      reg2_error[1] = reg2_error[1] / N

    # Region 3: from BP2 to BP3
    N = 0
    reg3_error = [0, 0]
    for operand in np.arange(self.breakpoint2+resolution, self.breakpoint3, resolution):
      exact_result = exact_operation(self.op_name, operand)
      approx_result = self.approx_operation(operand)
      reg3_error[0] = reg3_error[0] + pow((approx_result - exact_result) , 2) # MSE
      reg3_error[1] = reg3_error[1] + abs(approx_result - exact_result) # MAE
      N = N + 1
    if (N == 0):
      reg3_error[0] = 0
      reg3_error[1] = 0
    else:
      reg3_error[0] = reg3_error[0] / N
      reg3_error[1] = reg3_error[1] / N

    # Region 4: from BP3 to Endpoint
    N = 0
    reg4_error = [0, 0]
    for operand in np.arange(self.breakpoint3+resolution, self.endpoint, resolution):
      exact_result = exact_operation(self.op_name, operand)
      approx_result = self.approx_operation(operand)
      reg4_error[0] = reg4_error[0] + pow((approx_result - exact_result) , 2) # MSE
      reg4_error[1] = reg4_error[1] + abs(approx_result - exact_result) # MAE
      N = N + 1
    if (N == 0):
      reg4_error[0] = 0
      reg4_error[1] = 0
    else:
      reg4_error[0] = reg4_error[0] / N
      reg4_error[1] = reg4_error[1] / N

    total_mse = (reg1_error[0] + reg2_error[0] + reg3_error[0] + reg4_error[0])/4
    total_mae = (reg1_error[1] + reg2_error[1] + reg3_error[1] + reg4_error[1])/4

    return [reg1_error, reg2_error, reg3_error, reg4_error, total_mse, total_mae]


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
          return 1/(1 + np.exp(-operand)) * (1- 1/(1 + np.exp(-operand))) * (1 - 2/(1 + np.exp(-operand))) # TO FIX
    
