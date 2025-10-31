import math
import numpy as np
import matplotlib.pyplot as plt
from PWL_Approximation import pwl_approx
import PWL_Approximation
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pandas as pd
import csv


# --------------------------------------------------- #
def quantize(input, n_bits):
  return round(input*pow(2,n_bits))/pow(2,n_bits)

# --------------------------------------------------- #
def draw_functions():
    
    xlim_min = -10
    xlim_max = 10
    x = np.linspace(xlim_min, xlim_max, 200)

    # Drawing derivative functions
    func = []
    for jj in range(len(x)):
      func.append(PWL_Approximation.exact_operation('der_tanh', x[jj])) 
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvline(color="grey")
    #ax.axline((0, 0.5), slope=0.25, color="black", linestyle=(0, (5, 5)))
    ax.plot(x, func, linewidth=2, label=r"$f(x)' = \frac{d}{dx} tanh(x)$")
    ax.set_xlabel('x', fontsize=14, fontweight='bold', fontname='Arial')
    ax.set_ylabel(r"$f(x)'$", fontsize=12, fontweight='bold', fontname='Arial')
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(0, 1)
    ax.grid()
    
    # sigmoid
    func = []
    for jj in range(len(x)):
      func.append(PWL_Approximation.exact_operation('der_sigmoid', x[jj])) 
    ax.plot(x, func, 'r', linewidth=2.5, label=r"$f(x)' = \frac{d}{dx} sigmoid(x)$")
    ax.legend(fontsize=12)

    # Drawing second derivative functions
    xlim_min = 0
    xlim_max = 8
    x = np.linspace(xlim_min, xlim_max, 100)
    func = []
    for jj in range(len(x)):
      func.append(PWL_Approximation.exact_operation('2nd_der_tanh', x[jj])) 
    
    min_y = np.min(func)
    min_x = x[np.argmin(func)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvline(color="grey")
    #ax.axline((0, 0.5), slope=0.25, color="black", linestyle=(0, (5, 5)))
    ax.plot(x, func, linewidth=2, label=r"$f(x){''} = \frac{d^2}{dx^2} tanh(x)$")
    ax.set_xlabel('x', fontsize=14, fontweight='bold', fontname='Arial')
    ax.set_ylabel(r"$f(x){''} $", fontsize=12, fontweight='bold', fontname='Arial')
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(-0.8, 0)
    plt.annotate(f'Min: ({min_x:.2f}, {min_y:.2f})',
                 xy=(min_x, min_y),
                 xytext=(min_x + 0.6, min_y + 0.01), # Adjust text position
                 arrowprops=dict(facecolor='navy', shrink=0.05, alpha=0.4))
    ax.grid()
    
    # sigmoid
    func = []
    for jj in range(len(x)):
      func.append(PWL_Approximation.exact_operation('2nd_der_sigmoid', x[jj])) 
    min_y = np.min(func)
    min_x = x[np.argmin(func)]
    #plt.axvline(x=min_x, linestyle='--', color='gray', label='Minimum X')
    plt.annotate(f'Min: ({min_x:.2f}, {min_y:.2f})',
                 xy=(min_x, min_y),
                 xytext=(min_x - 0.01, min_y - 0.1), # Adjust text position
                 arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.4))


    ax.plot(x, func, 'r', linewidth=2.5, label=r"$f(x){''} = \frac{d^2}{dx^2} sigmoid(x)$")
    ax.legend(fontsize=12)

    plt.show()
    
# --------------------------------------------------- #
# find optimal endpoint
def endpoint_analysis(op_name, min_nbits, max_nbits, min_endpoint, max_endpoint, input_range):
  mae_array = [[] for x in range(min_nbits,max_nbits+1)]
  mre_array = [[] for x in range(min_nbits,max_nbits+1)]
  best_endpoints= []
  print('\r', 'MAE ARRAY:  ', mae_array)
  for nbits in range(min_nbits, max_nbits+1):
    pwl_obj = pwl_approx(op_name, 1, nbits, 0, 0)
    print('-------------------------------  BITWIDTH: ', nbits)
    endpoint_list = list(np.arange(min_endpoint, max_endpoint, 0.2))
    for endpoint_temp in endpoint_list:
      endpoint = int(endpoint_temp*10)/10
      pwl_obj.set_breakpoints(0, -1, 0, endpoint, endpoint)
      pwl_obj.create_luts()
      [mre, mae] = pwl_obj.error_analysis(input_range, False)
      print('----', '  end_point:', endpoint,   '     MRE: ', mre,   '  MAE: ', mae, '\r\r')
      mae_array[nbits-3].append(mae) 
      mre_array[nbits-3].append(mre) 
    
    min_mae = min(mae_array[nbits-3])
    #print('               ', min_mae, mae_array[nbits-3].index(min_mae))
    #print('          ', mae_array[nbits-3])
    #print('          ', endpoint_list, 'lllllllllll', int(endpoint_list[mae_array[nbits-3].index(min_mae)]))
    best_endpoints.append([nbits, int(endpoint_list[mae_array[nbits-3].index(min_mae)]*10)/10 ])
    del pwl_obj

  #print('\r', 'MAE ARRAY:  ', mae_array)
  #print('\r', 'MRE ARRAY:  ', mre_array)
  
  fig_mae, ax_mae = plt.subplots()
  ax_mae.axvline(color="grey")
  ax_mae.plot(endpoint_list, [i * 1000 for i in mae_array[0]], linewidth=2, label="num_bits=3")
  ax_mae.plot(endpoint_list, [i * 1000 for i in mae_array[1]], linewidth=2, label="num_bits=4")
  ax_mae.plot(endpoint_list, [i * 1000 for i in mae_array[2]], linewidth=2, label="num_bits=5")
  ax_mae.plot(endpoint_list, [i * 1000 for i in mae_array[3]], linewidth=2, label="num_bits=6")
  ax_mae.plot(endpoint_list, [i * 1000 for i in mae_array[4]], linewidth=2, label="num_bits=7")
  ax_mae.plot(endpoint_list, [i * 1000 for i in mae_array[5]], linewidth=2, label="num_bits=8")
  ax_mae.set(xlim=(2, 10), xlabel="Endpoint")
  ax_mae.set(ylim=(0, 2), ylabel=r"Mean Absolute Error (x$10^{-3})$")#"Mean Absolute Error (x1000)")
  ax_mae.legend(fontsize=14)
  plt.grid(True)
  plt.show()

  fig_mre, ax_mre = plt.subplots()
  ax_mre.axvline(color="grey")
  ax_mre.plot(endpoint_list, mre_array[0], linewidth=2, label="num_bits=3")
  ax_mre.plot(endpoint_list, mre_array[1], linewidth=2, label="num_bits=4")
  ax_mre.plot(endpoint_list, mre_array[2], linewidth=2, label="num_bits=5")
  ax_mre.plot(endpoint_list, mre_array[3], linewidth=2, label="num_bits=6")
  ax_mre.plot(endpoint_list, mre_array[4], linewidth=2, label="num_bits=7")
  ax_mre.plot(endpoint_list, mre_array[5], linewidth=2, label="num_bits=8")
  ax_mre.set(xlim=(2, 10), xlabel="Endpoint")
  ax_mre.set(ylim=(0, 1), ylabel="Mean Relative Error")#"Mean Absolute Error (x1000)")
  ax_mre.legend(fontsize=14)
  plt.grid(True)
  plt.show()

  return best_endpoints

# --------------------------------------------------- #
# Optimize breakpoint positions
def optimize_breakpoints(op_name, addr_nbits, startpoint, endpoint, input_range):
  bp_list = list(np.arange(startpoint, endpoint, 0.1))
  # Find the best middle point
  
  pwl_obj = pwl_approx(op_name, 1, addr_nbits, 0, 0)
  pwl_obj.set_breakpoints(startpoint, -1, 0, endpoint, endpoint)
  pwl_obj.create_luts()
  [mre, mae] = pwl_obj.error_analysis([0,10], False)
  print('\n----1', '     MRE: ', mre,   '  MAE: ', mae, '\r\r')
  #print('----1', pwl_obj.lut2)
  op=0.1
  del pwl_obj
  
  mae_array=[]
  mre_array=[]
  pwl_obj = pwl_approx(op_name, addr_nbits-1, addr_nbits-1, 0, 0)
  step = (endpoint - startpoint) / (pow(2,addr_nbits)-1)
  bp1_list=list(np.arange(startpoint, endpoint, 0.2))
  for bp1 in bp1_list:
    pwl_obj.set_breakpoints(startpoint, bp1, bp1+step, endpoint, endpoint)
    pwl_obj.create_luts()
    [mre, mae] = pwl_obj.error_analysis([0,10], False)
    mae_array.append(float(mae))
    mre_array.append(float(mre))
  #print('----2', '     MRE: ', mre,   '  MAE: ', mae, step*(pow(2,addr_nbits-1)-1),'\r\r')
  print('bp1_list:  ', bp1_list, '\n\n')
  print('mae_array: ', mae_array, '\n\n')
  print('mre_array: ', mre_array)

  min_index = mae_array.index(min(mae_array))
  best_bp1 = bp1_list[min_index]+step/2
  print('best BP1 =', best_bp1, min_index)
  
  mae_array2=[]
  mre_array2=[]
  region2_len_list=list(np.arange(0, endpoint-startpoint, 0.02))
  for region2_len in region2_len_list:
    pwl_obj.set_breakpoints(startpoint, best_bp1-region2_len/2, best_bp1+region2_len/2, endpoint, endpoint)
    pwl_obj.create_luts()
    [mre, mae] = pwl_obj.error_analysis([0,10], False)
    mae_array2.append(float(mae))
    mre_array2.append(float(mre))

  print('region2_len_list:  ', region2_len_list, '\n\n')
  print('mae_array: ', mae_array2, '\n\n')
  print('mre_array: ', mre_array2)
  print('BEST REG2 LEN:', region2_len_list[mae_array2.index(min(mae_array2))])

  print('\n------ MIN MAE:  ', min(mae_array2),  '          MIN MRE: ', min(mre_array2))

  return 0
  
# --------------------------------------------------- #
# Find meddian of BP1 and BP2
def find_middle_bp(op_name, startpoint, endpoint):
  print('Finding middle point between Breakpoint1 and Breakpoint2 ... ')
  second_der_list = []
  operand_list=list(np.arange(startpoint, endpoint, 0.02))
  for ii in operand_list:
    if op_name == "der_tanh":
      second_der_list.append(PWL_Approximation.exact_operation("2nd_der_tanh", ii))
    elif op_name == "der_sigmoid":
      second_der_list.append(PWL_Approximation.exact_operation("2nd_der_sigmoid", ii))

  middle_bp_point = operand_list[second_der_list.index(min(second_der_list))]
  print(' MIDDLE BP POINT: ', middle_bp_point)
  print('-------------------\n')

  '''
  fig_mre, ax_mre = plt.subplots()
  ax_mre.axvline(color="grey")
  ax_mre.plot(operand_list, second_der_list, linewidth=2, label="num_bits=3")
  ax_mre.set(xlim=(startpoint, 3), xlabel="Endpoint")
  ax_mre.set(ylim=(-1, 0), ylabel="Mean Relative Error")#"Mean Absolute Error (x1000)")
  ax_mre.legend(fontsize=14)
  plt.grid(True)
  plt.show()
  '''
  return middle_bp_point

# --------------------------------------------------- #
def error_select(error_result_vec, error_metric):
  if error_metric == "MRE" or error_metric == "mre":
    return error_result_vec[0]
  elif error_metric == "MAE" or error_metric == "mae":
    return error_result_vec[1]
  else:
    print('ERROR: Unknown Error Metric:', error_metric)
    return 0


# --------------------------------------------------- #
def optimize_reg2_exhaustive(op_name, reg1_nbits, reg3_nbits, startpoint, endpoint, middle_bp, breakpoint3, start_value, end_value, error_metric):
  epsilon = 0.1
  reg2_len_div2 = epsilon
  obj = pwl_approx(op_name, reg1_nbits, reg3_nbits, start_value, end_value)
  obj.set_breakpoints(start_point, middle_bp, middle_bp, breakpoint3, end_point)
  obj.create_luts()
  error_result_vec = obj.error_analysis([startpoint, endpoint], False)
  new_error = error_select(error_result_vec, error_metric)
  del obj

  last_error = new_error + epsilon
  while new_error < last_error:
    last_error = new_error
    reg2_len_div2 = reg2_len_div2 + epsilon
    obj = pwl_approx(op_name, reg1_nbits, reg3_nbits, start_value, end_value)
    obj.set_breakpoints(start_point, middle_bp-reg2_len_div2, middle_bp+reg2_len_div2, breakpoint3, end_point)
    obj.create_luts()
    error_result_vec = obj.error_analysis([startpoint, endpoint], False)
    new_error = error_select(error_result_vec, error_metric)
    del obj

  return [reg2_len_div2, last_error]
  
  
  '''
  bp1 = middle_bp
  bp2 = middle_bp

  last_error = new_error + epsilon
  while new_error < last_error:
    last_error = new_error
    obj = pwl_approx(op_name, reg1_nbits, reg3_nbits, start_value, end_value)
    obj.set_breakpoints(start_point, bp1, bp2+epsilon, breakpoint3, end_point)
    obj.create_luts()
    error_result_vec = obj.error_analysis([startpoint, endpoint], False)
    right_expand_error = error_select(error_result_vec, error_metric)
    del obj
    obj = pwl_approx(op_name, reg1_nbits, reg3_nbits, start_value, end_value)
    obj.set_breakpoints(start_point, bp1-epsilon, bp2, breakpoint3, end_point)
    obj.create_luts()
    error_result_vec = obj.error_analysis([startpoint, endpoint], False)
    left_expand_error = error_select(error_result_vec, error_metric)  
    del obj

    if left_expand_error > right_expand_error: # it's better to expand region2 to the right
      bp2 = bp2 + epsilon
    else:
      bp1 = bp1 - epsilon
    new_error = min(left_expand_error, right_expand_error)
    print ('kkkkkk     bp1:', f"{bp1:2f}", '     bp2:', f"{bp2:2f}", '      breakpoint3: ', f"{breakpoint3:6f}",  '        left_expand_error: ', f"{left_expand_error:6f}", '        right_expand_error', f"{right_expand_error:6f}",)
  print ('\n')

  return [[bp1,bp2], last_error]
  '''
  

# --------------------------------------------------- #
def optimize_reg2_gd(op_name, reg1_nbits, reg3_nbits, startpoint, endpoint, middle_bp, breakpoint3, start_value, end_value, error_metric):
  # Gradiant descent optimization of region 2
  epsilon = 0.01
  learning_rate = 100
  start_value = PWL_Approximation.exact_operation(op_name, startpoint)
  end_value = PWL_Approximation.exact_operation(op_name, endpoint)
  delta_error = epsilon
  negative_gradiant = True
  bp2_len = epsilon
  while abs(delta_error) > 0.0000005:
    obj = pwl_approx(op_name, reg1_nbits, reg3_nbits, start_value, end_value)
    obj.set_breakpoints(start_point, middle_bp-bp2_len, middle_bp+bp2_len, breakpoint3, end_point)
    obj.create_luts()
    error_result_vec = obj.error_analysis([startpoint, endpoint], False)
    error_d1 = error_select(error_result_vec, error_metric)
    del obj
    bp2_len = bp2_len + epsilon
    obj2 = pwl_approx(op_name, reg1_nbits, reg3_nbits, start_value, end_value)
    obj2.set_breakpoints(start_point, middle_bp-bp2_len, middle_bp+bp2_len, breakpoint3, end_point)
    obj2.create_luts()
    error_result_vec = obj2.error_analysis([startpoint, endpoint], False)
    error_d2 = error_select(error_result_vec, error_metric)
    del obj2

    print('\n     error_d1:', error_d1, '     error_d2', error_d2)
    delta_error = (error_d1 - error_d2)

    if delta_error<0 and negative_gradiant == True:
      learning_rate = learning_rate /2
      epsilon = epsilon/2
      negative_gradiant = False
      print('** Learning Rate= ', learning_rate, delta_error, '             P', negative_gradiant)
    elif delta_error>0 and negative_gradiant == False:
      learning_rate = learning_rate /2
      epsilon = epsilon/2
      negative_gradiant = True
      print('** Learning Rate= ', learning_rate, delta_error, '             N', negative_gradiant)
    
    bp2_len = bp2_len + learning_rate * (delta_error/epsilon)  
    #print(' ----000    bp3:', bp3, '     learning_rate:', learning_rate, '       learning_rate * (delta_error/epsilon):', learning_rate * (delta_error/epsilon))
    #error_d1 = error_d2
    print('     new_bp2_len: ', bp2_len, '         delta ', delta_error)

  print('     bp2_len: ', bp2_len, 'error_d1 ', error_d1)

  return [bp2_len, error_d1]



# --------------------------------------------------- #
def optimize_BPs(op_name, reg1_nbits, reg3_nbits, startpoint, endpoint, middle_bp, error_metric):
    
  epsilon = 0.01
  learning_rate = 1000
  start_value = PWL_Approximation.exact_operation(op_name, startpoint)
  end_value = PWL_Approximation.exact_operation(op_name, endpoint)

  bp3 = end_point-epsilon
      
  delta_error = epsilon
  negative_gradiant = True
  while abs(delta_error) > 0.0000002:
    [reg2_len_div2, error_d1] = optimize_reg2_exhaustive(op_name, reg1_nbits, reg3_nbits, startpoint, endpoint, middle_bp, bp3, start_value, end_value, error_metric)
    bp3 = bp3 - epsilon
    [reg2_len_div2, error_d2] = optimize_reg2_exhaustive(op_name, reg1_nbits, reg3_nbits, startpoint, endpoint, middle_bp, bp3, start_value, end_value, error_metric)
    print('\n     error_d1:', error_d1, '     error_d2', error_d2)
    delta_error = (error_d1 - error_d2)

    if delta_error<0 and negative_gradiant == True:
      learning_rate = learning_rate /2
      epsilon = epsilon/2
      negative_gradiant = False
      print('**** Learning Rate= ', learning_rate, delta_error, '             P', negative_gradiant)
    elif delta_error>0 and negative_gradiant == False:
      learning_rate = learning_rate /2
      epsilon = epsilon/2
      negative_gradiant = True
      print('**** Learning Rate= ', learning_rate, delta_error, '             N', negative_gradiant)
    
    bp3 = bp3 - learning_rate * (delta_error/epsilon)  
    #print(' ----000    bp3:', bp3, '     learning_rate:', learning_rate, '       learning_rate * (delta_error/epsilon):', learning_rate * (delta_error/epsilon))
    error_d1 = error_d2
    print('     new_bp3: ', bp3, '   delta: ', delta_error)

  return(middle_bp-reg2_len_div2, middle_bp+reg2_len_div2, bp3, error_d1) # return BP1 , BP2 , BP3, Error result


# --------------------------------------------------- #
def error_plot_3d(op_name, reg1_nbits, reg3_nbits, startpoint, endpoint, middle_bp, error_metric, reg2_len_range, reg4_len_range):
  reg2_tick = 0.1
  reg4_tick = 0.5
  print('******  reg2_len:', reg2_len_range, '   middle', middle_bp)

  reg2_option_list = list(np.arange(reg2_len_range[0], reg2_len_range[1], reg2_tick))
  reg4_option_list = list(np.arange(reg4_len_range[0], reg4_len_range[1], reg4_tick))

  print('reg2', reg2_option_list)
  print('reg4', reg4_option_list)

  mm=0
  error_array = []
  for ii in reg4_option_list:
    temp_vec = []
    for jj in reg2_option_list:
      obj = pwl_approx(op_name, reg1_nbits, reg3_nbits, start_value, end_value)
      obj.set_breakpoints(start_point, middle_bp-jj/2, middle_bp+jj/2, end_point-ii, end_point)
      obj.create_luts()
      error_result_vec = obj.error_analysis([startpoint, endpoint], False)
      error_d1 = error_select(error_result_vec, error_metric)
      del obj
      temp_vec.append(error_d1)
    error_array.append(temp_vec)
    mm=mm+1

  print('\nerrot array[1]', error_array[1])
  print('\nerrot array[2]', error_array[2])

  '''
  # Save to Excel
  formatted_array = np.insert(error_array, 0, reg2_option_list, axis=0) 
  formatted_array = np.insert(formatted_array, 0, np.insert(reg4_option_list, 0, 0), axis=1)
  df = pd.DataFrame(formatted_array)
  df.insert(0, "", "")   # insert an empty column at index 0
  with pd.ExcelWriter("oooo.xlsx") as writer:
    # Write a single description row
    pd.DataFrame([["Columns: Region4 Length / Rows: Region2 Length"]]).to_excel(
        writer, index=False, header=False
    )
    # Write array data starting from row 2
    df.to_excel(
        writer, index=False, header=False, startrow=2, float_format="%.6f"
    )
  '''

  
  # Drawing surface plot
  error_array_np = np.array(error_array)
  print('NNNNNNNNNNN MRE   ',error_array_np.min())
  
  X, Y = np.meshgrid(reg2_option_list, reg4_option_list)
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))

  surf = ax.plot_surface(X, Y, error_array_np, cmap='viridis',
                       linewidth=0, antialiased=False)

  # Customize the z axis.
  ax.view_init(elev=15, azim=30)
  ax.set_zlim(0, error_array_np.max())
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter('{x:.04f}')

  plt.title(f"Region1 Bitwidth: {reg1_nbits} , Region3 Bitwidth: {reg3_nbits}", pad=0)
  if error_metric == 'mae':
    ax.set(zlabel='Mean Average Error')
  elif error_metric == 'mre':
    ax.set(zlabel='Mean Relative Error')
  ax.set(ylabel='Region 4 Length')
  ax.set(xlabel='Region 2 Length')
  # Add a color bar which maps values to colors.
  #fig.colorbar(surf, shrink=0.5, aspect=5)

  plt.show()

# --------------------------------------------------- # pwl_optimizer
def pwl_optimizer(op_name, start_value, end_value, error_metric, target_error_value):
  # ---- Uniform PWL test
  pwl_segment_id_width = 0    # number of segments in uniform PWL is 2^pwl_segment_id_width
  print('-----------\nStart uniform PWL test ...')
  current_error = 1
  while(current_error > target_error_value):
    pwl_segment_id_width = pwl_segment_id_width + 1
    print('  Number of segments : ', pow(2,pwl_segment_id_width))
    bp1 = -1
    bp2 = 0
    obj = pwl_approx(op_name, 1, pwl_segment_id_width, start_value, end_value)
    obj.set_breakpoints(start_point, bp1, bp2, end_point, end_point)
    obj.create_luts()
    [mse, mae] = obj.error_analysis([0,end_point], False)
    print('  MRE: ', f"{mse:6f}",   '  MAE: ', f"{mae:6f}", '\r\r')
    if (error_metric == 'mae'):
      current_error = mae
    elif (error_metric == 'mse'):
      current_error = mse
    else:
      print('*** ERROR: Undefined Error Metric: "', error_metric)
      exit
    del obj
  print('  Uniform PWL number of segments: ', pow(2, pwl_segment_id_width), '      ', error_metric, ': ', current_error)
  print('----------\n')

  # ---- Break point optimization
  print('----------\nStart breakpoint optimization ...')

  reg1_nbits = 0  # number of PWL segments in Region 1 = 2 ** reg1_nbits
  reg3_nbits = 1  # number of PWL segments in Region 3 = 2 ** reg3_nbits
  end_value = PWL_Approximation.exact_operation(op_name, end_point)
  middle_bp = find_middle_bp(op_name, 0, end_point)
  current_error = 1
  reg1_error = [1,1]
  reg3_error = [0,0]

  while(current_error > target_error_value):
    if (error_metric == 'mae'):
      if (reg1_error[1] > reg3_error[1]):
        reg1_nbits = reg1_nbits + 1
      else:
        reg3_nbits = reg3_nbits + 1
    elif (error_metric == 'mse'):
      if (reg1_error[0] > reg3_error[0]):
        reg1_nbits = reg1_nbits + 1
      else:
        reg3_nbits = reg3_nbits + 1
    else:
      print('*** ERROR: Undefined Error Metric: "', error_metric)
      exit

    print('\n\n   Trying # PWL1 segments = ', pow(2,reg1_nbits), '     # PWL3 segments = ', pow(2,reg3_nbits))
    print('\n\n   Calling optimize_BPs ...')

    [opt_bp1, opt_bp2, opt_bp3, error_result] = optimize_BPs(op_name='der_tanh', reg1_nbits=reg1_nbits, reg3_nbits=reg3_nbits, 
                                                          startpoint=start_point, endpoint=end_point, middle_bp=middle_bp, error_metric=error_metric)

    print('   * Region1 PWL Segments', pow(2,reg1_nbits), '  Region3 PWL Segments', pow(2,reg3_nbits), 
          'Optimal BP1: ', f"{opt_bp1:.6f}", '    Optimal BP2: ', f"{opt_bp2:.6f}", '    Optimal BP3: ', f"{opt_bp3:.6f}", '       ', error_metric, ' : ', f"{error_result:.6f}")
    current_error = error_result

    obj2 = pwl_approx(op_name, reg1_nbits, reg3_nbits, start_value, end_value)
    obj2.set_breakpoints(start_point, opt_bp1, opt_bp2, opt_bp3, end_point)
    obj2.create_luts()
    [reg1_error, reg2_error, reg3_error, reg4_error, total_mre, total_mae] = obj2.detailed_error_analysis()
    print('      Reg1 MSE = '  , f"{reg1_error[0]:.6f}", ' Reg1 MAE = ', f"{reg1_error[1]:.6f}", 
          '\n      Reg2 MSE = ', f"{reg2_error[0]:.6f}", ' Reg2 MAE = ', f"{reg2_error[1]:.6f}", 
          '\n      Reg3 MSE = ', f"{reg3_error[0]:.6f}", ' Reg3 MAE = ', f"{reg3_error[1]:.6f}", 
          '\n      Reg4 MSE = ', f"{reg3_error[0]:.6f}", ' Reg4 MAE = ', f"{reg4_error[1]:.6f}" )
    del obj2
    
    print('----------\n')

# --------------------------------------------------- #

# --------------------------------------------------- # Main
# ---- Draw the two derivative functions
print('---------------------\n Start running draw_functions to draw first and second derivatives plots ...')
draw_functions()
print('---------------------\n')

# ---- Global initializations
print('---------------------\nStart global parameter initialization ...')
op_name = 'der_tanh'
error_metric='mae'
start_point = 0
end_point = 10
start_value = 0
target_error = 0.0001
end_value = PWL_Approximation.exact_operation(op_name, end_point)
print('Function: ', op_name, '\nstart_point: ', start_point, '\nend_point: ', end_point)
print('---------------------\n')

# ---- PWL Optimizer
print('---------------------\nOptimize uniform PWl and segmented PWL to reach ', error_metric, ' of ', target_error, ' ...')
pwl_optimizer(op_name=op_name, start_value=start_value, end_value=end_value, error_metric=error_metric, target_error_value=target_error)

# ---------------------------------------
breakpoint3 = 5
obj2 = pwl_approx(op_name, 2, 3, start_value, end_value)
obj2.set_breakpoints(start_point, middle_bp-0.2, middle_bp+0.2, breakpoint3, end_point)
obj2.create_luts()
[reg1_error, reg2_error, reg3_error, reg4_error, total_mre, total_mae] = obj2.detailed_error_analysis()
print('Reg1 MSE = '  , f"{reg1_error[0]:.6f}", ' Reg1 MAE = ', f"{reg1_error[1]:.6f}", 
      '\nReg2 MSE = ', f"{reg2_error[0]:.6f}", ' Reg2 MAE = ', f"{reg2_error[1]:.6f}", 
      '\nReg3 MSE = ', f"{reg3_error[0]:.6f}", ' Reg3 MAE = ', f"{reg3_error[1]:.6f}", 
      '\nReg4 MSE = ', f"{reg3_error[0]:.6f}", ' Reg4 MAE = ', f"{reg4_error[1]:.6f}" )
[mre, mae] = obj2.error_analysis([0,end_point], True)
print('\n----', '  BP1:', f"{obj2.breakpoint1:.3f}", '  BP2:', f"{obj2.breakpoint2:.3f}", '  BP3:', f"{obj2.breakpoint3:.3f}",  '  EP:',  end_point, ' ||    MRE: ', f"{mre:.6f}",   '  MAE: ', f"{mae:.6f}", '\r\r')
#
# ----
#middle_bp=4
#error_plot_3d(op_name='der_tanh', reg1_nbits=3, reg3_nbits=4, startpoint=start_point, endpoint=end_point, middle_bp=middle_bp, error_metric='mre', reg2_len_range = [0, middle_bp], reg4_len_range= [0,5.2])#end_point-middle_bp])

print('------------------------------------------------')

