from scipy.stats import norm
import numpy as np
import pandas as pd

df = pd.read_excel('gaussian_all_data.xlsx')
df.replace(np.nan, 0 , inplace=True)
df = df.dropna(how='all', axis=0)
D =1
def  Dcs(q,p):
    output1 = 0
    output2 = 0
    output3 = 0
    output4 = 0
    output5 = 0


    # corrpsonds to compoet 1
    for m in range(len(q)):
        for k in range(len(p)):
            output1 +=( q[m]['weight'] * p[k]['weight'] * norm.pdf(q[m]['mu'], loc=p[k]['mu'], scale=np.sqrt(q[m]['var_inv'] + p[k]['var_inv'])))

        # output2 += (q[m]['weight']**2 *np.sqrt(np.abs(1/q[m]['var_inv']))) / (2 * np.pi) ** D/2
        # output2 += (q[m]['weight']**2 *np.sqrt(1/q[m]['var_inv'])) / (2 * np.pi) ** (D/2)

        for m_ in range(len(q)):
            #if m_ < m:
            output3 += q[m]['weight'] * q[m_]['weight'] * norm.pdf(q[m]['mu'], loc=q[m_]['mu'], scale=np.sqrt(q[m]['var_inv'] + q[m_]['var_inv']))


    for k in range(len(p)):

        # output4 += (p[k]['weight'] ** 2 * np.sqrt(np.abs(1/p[k]['var_inv']))) / (2 * np.pi ) ** D / 2
        # output4 += (p[k]['weight'] ** 2 * np.sqrt(1/p[k]['var_inv'])) / (2 * np.pi ) ** (D / 2)

        for k_ in range(len(p)):
            #if k_< k:
            output5 += p[k]['weight'] * p[k_]['weight'] * norm.pdf(p[k]['mu'], loc=p[k_]['mu'],scale=np.sqrt(p[k]['var_inv'] + p[k_]['var_inv']))

    if output1 ==0 or output5==0 or output4==0 or output3 == 0 or output1 == 0:
        print(output1, output3, output5)


    term1 = -np.log10(output1)
    term2 = 1/2 * np.log10(output3)
    term3 = 1/2 * np.log10(output5)
    ans  = term1 + term2 + term3
    print(" got  " + str(ans))
    return ans

sequence_distances = [[0 for i in range(len(df))]for j in range(len(df))]
for i in range(len(df)):
    sum1 = sum((df.loc[i]['Peak 1 a'], df.loc[i]['Peak 2 a'], df.loc[i]['Peak 3 a'], df.loc[i]['Peak NIR a Log']))
    if sum1 == 0.0:
        continue
    q = [{'weight': df.loc[i]['Peak 1 a Log'] , 'mu': df.loc[i]['Peak 1 b'], 'var_inv': (df.loc[i]['Peak 1 c']) ** 2},
         {'weight': df.loc[i]['Peak 2 a Log'] , 'mu': df.loc[i]['Peak 2 b'], 'var_inv': (df.loc[i]['Peak 2 c']) ** 2},
         {'weight': df.loc[i]['Peak 3 a Log'] , 'mu': df.loc[i]['Peak 3 b'], 'var_inv': (df.loc[i]['Peak 3 c']) ** 2},
         {'weight': df.loc[i]['Peak NIR a Log'] , 'mu': df.loc[i]['Peak NIR b'], 'var_inv': (df.loc[i]['Peak NIR c']) ** 2}]
    # q = [d for d in q if any(d.values()) != 0.0]
    q = [d for d in q if not all(value == 0.0 or np.isnan(value) for value in d.values())]

    for j in range(len(df)):
        if i != j :
            sum2 =sum((df.loc[j]['Peak 1 a'], df.loc[j]['Peak 2 a'], df.loc[j]['Peak 3 a'], df.loc[j]['Peak NIR a Log']))
            if sum2 == 0.0:
                continue
            p = [{'weight':df.loc[j]['Peak 1 a Log'], 'mu':df.loc[j]['Peak 1 b'], 'var_inv':(df.loc[j]['Peak 1 c'])**2},
                 {'weight':df.loc[j]['Peak 2 a Log'], 'mu':df.loc[j]['Peak 2 b'], 'var_inv':(df.loc[j]['Peak 2 c'])**2},
                 {'weight':df.loc[j]['Peak 3 a Log'], 'mu':df.loc[j]['Peak 3 b'], 'var_inv':(df.loc[j]['Peak 3 c'])**2},
                 {'weight': df.loc[j]['Peak NIR a Log'] , 'mu': df.loc[j]['Peak NIR b'], 'var_inv': (df.loc[j]['Peak NIR c']) ** 2}]
            # p = [d for d in p if any(d.values()) != 0.0]
            p =[d for d in p if not all(value == 0.0 or np.isnan(value) for value in d.values())]

            sequence_distances[i][j] = Dcs(q,p)

df = pd.DataFrame(sequence_distances[0:])
df.to_excel('CSdistances_gaussian_all_data.xlsx', index=False)

