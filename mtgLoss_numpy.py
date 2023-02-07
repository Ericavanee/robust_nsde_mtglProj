import numpy as np


def calibratedOption_payoff(strikes_call, maturities, theta_ls, itr, dt = 1/96, timesteps = 96):
    # generate timegrid given time steps.
    timegrid = np.linspace(0,1,timesteps+1)
    # record the lattest maturity as ind_T.
    ind_T = maturities[-1]
    
    # initiate holders for vanilla option at different maturities
    mat_ls = []
    for i in range(len(maturities)):
        mat_ls.append(np.zeros((itr,len(strikes_call),maturities[i])))
        
    # initiate holder for exotic option
    exotic_option_price = np.zeros((itr, ind_T))
    
    # price vanilla option for each maturity
    for i in range(len(maturities)):
        # outer monte carlo
        my_mat = maturities[i]
        stock, running_max = Heston_stock(*theta_ls[i], itr = itr, dt = dt, timesteps = timesteps)
        for j in range(itr):
            # create a copy of running max
            running_max_copy = running_max
            traj = outer_stock[i] 
            for k in range(0,my_mat):
                if k == 0:
                    inner_stock = outer_stock
                    inner_max = running_max
                else:
                # generate inner monte carlo trials
                    inner_stock, inner_max = Heston_stock(*theta_ls[i], itr, dt, timesteps = my_mat-k)
                
                S_old = inner_stock.T[my_mat-k-2]
                for idx, strike in enumerate(strikes_call):
                    price_vanilla = np.exp(-r*my_mat)*np.clip(np.array(S_old-strike),0, 10**100)
                    # for each maturity and at each time step
                    mat_ls[i][j,idx,k] = price_vanilla.mean()
                
                # only price exotic option when we have ind_T as maturity
                if my_mat == ind_T:
                    # update running max
                    running_max_copy = np.maximum(running_max_copy,inner_max)
                    exotic_option_price[j,k] = np.exp(-r*timegrid[ind_T])*(running_max_copy-S_old).mean()
                    
                
    # taking mean across outer monte carlo trials
    exotic_option_price = exotic_option_price.mean(axis = 0)
    for i in range(len(maturities)):
        mat_ls[i] = mat_ls[i].mean(axis = 0)
            
                   
    return exotic_option_price, mat_ls                  



def kernel(rho, x):
    # x can be a matrix or a scalar, rho is a scalar
    return ((rho-1)/2)*(np.float_power(np.abs(x)+1,-rho))


def mtgLoss_vanilla(rho, calibrated_payoff, market_price):
    # group by maturities
    mat_num = len(calibrated_payoff) # number of maturities
    itr, strike_num = calibrated_payoff[0].shape
    vanilla_loss = []
    for i in range(mat_num):
        sum_ls = []
        for j in range(itr):
            # perform element-wise matrix multiplication
            diff = calibrated_payoff[i][j].T.reshape(strike_num,1)-market_price[i][j]
            compute_mat = np.multiply(diff,kernel(rho,diff))  
            sum_ls.append(np.sum(np.multiply(compute_mat,compute_mat)))
        vanilla_loss.append(sum(sum_ls)) # summing loss over all iterations
    vanilla_loss = sum(vanilla_loss) # summing loss over all maturities
    # compute average loss across iterations
    avg_loss = vanilla_loss/(mat_num*itr)
    
    return avg_loss



# write the function for martingale projection loss
def mtgLoss_pair(rho, calibrated_payoff, market_price, summation = False):
    # group by maturities
    mat_num = len(calibrated_payoff[1]) # number of maturities
    vanilla_loss = []
    for i in range(mat_num):
        # perform element-wise matrix multiplication
        diff = calibrated_payoff[1][i]-market_price[1][i]
        compute_mat = np.multiply(diff,kernel(rho,diff))  
        vanilla_loss.append(np.sum(np.multiply(compute_mat,compute_mat)))
    vanilla_loss = sum(vanilla_loss)
    # exotic options
    diff = calibrated_payoff[0]-market_price[0]
    compute_mat = np.multiply(diff,kernel(rho,diff)) 
    exotic_loss = np.sum(np.multiply(compute_mat,compute_mat))
    
    if summation:
        sum_loss = exotic_loss+vanilla_loss
        print("Total loss of vanilla and exotic options: ", sum_loss)
        return sum_loss
    else:
        print("Loss in vanilla: ", vanilla_loss)
        print("Loss in exotic: ", exotic_loss)
        return (exotic_loss, vanilla_loss)