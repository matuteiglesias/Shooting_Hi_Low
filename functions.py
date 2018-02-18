import numpy as np
import pandas as pd

def compute_mcp(df, bin_RCA_column = 'RCA_bin', ):
    return df.set_index(['ccode','pcode'])[bin_RCA_column].unstack(fill_value = 0)  

def compute_proximity_density(mcp_y, time_period_value = 0):
    """
    Compute proximity and density as in Hausmann Hidalgo 2007.
    
    Input: mcp_y (pandas DataFrame)
        Matrix representing export basket. Rows are countries and columns are products. 
        Typically entries tell if RCA > 1 in such country-product
        
    Output: fi, w (DataFrame tuple)
       These dataframes compute all proximities between products and all densities for products according
       to the product space of a country-year.
    """
    M_CP = mcp_y.as_matrix()

    s = list(M_CP.sum(axis = 0))
    S = np.array([s,]*M_CP.shape[1])
    norm = np.maximum(S.T, S)
    
    fi_m_y = np.true_divide(np.dot(M_CP.T, M_CP), norm)
    
    fi_df_y = pd.DataFrame(fi_m_y, index = mcp_y.columns, columns = mcp_y.columns)
    
    fi_y = pd.DataFrame(fi_df_y.unstack())
    fi_y.index.names = ['p_source', 'p_target']
    fi_y.reset_index(inplace = True)
    fi_y['year'] = time_period_value
    fi_y.columns = ['p_source', 'p_target', 'proximity', 'time_period']
 
        
    s = list(fi_m_y.sum(axis = 0))
    norm = np.array([s,]*M_CP.shape[0])
    
    w_m_y = np.divide(np.dot(M_CP, fi_m_y), norm)
    w_df_y = pd.DataFrame(w_m_y, index = mcp_y.index, columns = mcp_y.columns)


    w_y = pd.DataFrame(w_df_y.unstack())
    w_y.reset_index(inplace = True)
    w_y['year'] = time_period_value
    w_y.columns = ['pcode', 'ccode', 'density', 'time_period']
    
    return fi_y, w_y