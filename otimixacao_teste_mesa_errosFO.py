import datetime
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy.integrate import odeint
import matplotlib.pyplot as plt


tempo_ate_diag = 5

df = pd.read_csv('df_2022_04_04.csv', sep=';', low_memory=False)


def datas(df):
    return [x for x in df.columns if x.startswith("DATA")]


for col in datas(df):
    df[col] = pd.to_datetime(df[col], errors="coerce")


def data_mais_recente_df_saved(df_saved):
    global data_mais_recente

    last_df = pd.read_csv(df_saved, sep=';', low_memory=False)

    for col in datas(last_df):
        last_df[col] = pd.to_datetime(last_df[col], errors="coerce")

    data_mais_recente = last_df[datas(last_df)].max().max()
    return data_mais_recente


data_mais_recente = data_mais_recente_df_saved('df.csv')
print(data_mais_recente)

def inf_atuais(df):
    # Troco todos os valores NaN por 0
    df = df.fillna(0)

    # renomeando as colunas para os nomes corretos
    df.columns = ['inf_total_dia', 'rec_total_dia', 'morto_total_dia']

    # criando a coluna de inf atuais (como a total é por dia basta eu subtrair do total de infectados o total
    # de mortos e recuperados)
    df['inf_atuais'] = df.apply(
        lambda row: row['inf_total_dia'] - row['morto_total_dia'] - row['rec_total_dia'], axis=1)

    # crio uma coluna chamada data
    df['Data'] = df.index

    return df


def recalcular_df(df, latencia, tempo_ate_diag):
    # Aqui arrumamos a coluna dos recuperados e salvamos na variavel df_recalculada
    # if 'df_recalculada' not in locals():
    # selecionamos somente as colunas que possuem formato de data e numeros(codigosIBGE).
    df_recalculada = df.select_dtypes(include=['datetime', 'number'])  # .drop(columns=['LABORATORIO'])  # object

    # criamos uma nova coluna para arrumar os infectados com base no inicio de sintomas, se o tempo entre DIS e DD for >= 5 dias
    # trocamos por DIS + 5, caso contrario por DD. Funçao np.where que diz: onde temos isso,
    # troque por isso caso verdadeiro, ou troque por isso caso falso.
    df_recalculada['nova_data_infec'] = np.where(
        (df_recalculada['DATA_DIAGNOSTICO'] - df_recalculada['DATA_INICIO_SINTOMAS']).dt.days >= 5,
        df_recalculada.DATA_INICIO_SINTOMAS + datetime.timedelta(days=(tempo_ate_diag)),  # verdadeiro
        df_recalculada.DATA_DIAGNOSTICO)  # falso

    # Agora utilixamos do tempo médio entre a DIS e a DD que é 5 dias para transladar a coluna para 5 dias antes
    df_recalculada['nova_data_infec'] = df_recalculada['nova_data_infec'] - datetime.timedelta(days=(tempo_ate_diag))

    # data_recuperado diz a respeito do dia que a pessoa possa ter se recuperado. Isso acontece de 7 a 10 dias apos
    # o incio de sintomas (tempo retirado de artigos cientificos)
    data_recuperado = df_recalculada.DATA_DIAGNOSTICO + datetime.timedelta(
        days=latencia - tempo_ate_diag)  # - tempo_ate_diag)

    # aqui comecamos a restruturaço da coluna de recuperados utilizamos a funçao np.where que diz: onde temos isso,
    # troque por isso caso verdadeiro, ou troque por isso caso falso.
    df_recalculada['nova_data_rec'] = np.where((df_recalculada.DATA_OBITO.isna()) &
                                               ((df_recalculada.DATA_RECUPERADO_DIVULGACAO.isna() &
                                                 ((data_recuperado) < data_mais_recente)) |
                                                (df_recalculada.DATA_RECUPERADO_DIVULGACAO >
                                                 data_recuperado)),
                                               data_recuperado,  # verdadeiro
                                               df_recalculada.DATA_RECUPERADO_DIVULGACAO)  # falso

    df_recalculada = df_recalculada.sort_values(by=['DATA_DIAGNOSTICO']).reset_index(drop=True)
    return df_recalculada


def inf_atuais_londrina(df, latencia, tempo_ate_diag):
    df_parana = recalcular_df(df, latencia, tempo_ate_diag)

    df_recalculada = df_parana

    londrina_confPorDia = df_recalculada[df_recalculada['IBGE_RES_PR']
        .isin([4113700])][
        ['nova_data_infec', 'nova_data_rec', 'DATA_OBITO']].apply(
        pd.Series.value_counts).expanding(min_periods=1).sum()

    londrina_inf_atuais = inf_atuais(londrina_confPorDia)

    # suscetiveis = N - Infectados totais
    londrina_inf_atuais['suscetiveis'] = 569733 - londrina_inf_atuais['inf_total_dia']
    londrina_inf_atuais['expostos'] = londrina_inf_atuais['inf_atuais'] * 2

    #print(londrina_inf_atuais)

    #print(londrina_inf_atuais.columns)
    return londrina_inf_atuais.suscetiveis, londrina_inf_atuais.expostos, londrina_inf_atuais.inf_atuais, \
           londrina_inf_atuais.rec_total_dia, londrina_inf_atuais.morto_total_dia




def otimizar(latencia, t0, S, E, I, R, D):
    try:

        # Create a new model
        m = gp.Model("mip1")
        #        m.params.NonConvex = 2
        #        m.params.NumericFocus=1
        m.params.LogToConsole = 0
        # Create variables
        beta = m.addVar(vtype=GRB.CONTINUOUS,ub=GRB.INFINITY,lb=-GRB.INFINITY, name="beta")
        sigma = m.addVar(vtype=GRB.CONTINUOUS,ub=GRB.INFINITY,lb=-GRB.INFINITY, name="sigma")
        gammaR = m.addVar(vtype=GRB.CONTINUOUS,ub=GRB.INFINITY,lb=-GRB.INFINITY, name="gammaR")
        gammaD = m.addVar(vtype=GRB.CONTINUOUS,ub=GRB.INFINITY,lb=-GRB.INFINITY, name="gammaD")

        t = t0
        obj = 0.0
        while t0 <= t < t0 + latencia:
            if t == 0:
                print(S[t])
                obj += (S[t]) ** 2 + (E[t]) ** 2 + (I[t]) ** 2 + (R[t]) ** 2 + (D[t]) ** 2
                # print(obj)
            elif t == 1:
                obj += ((S[t] - (2 * beta * S[t - 1] * I[t - 1] / N)) ** 2
                        + (E[t] - ((2 * beta * S[t - 1] * I[t - 1] / N - sigma * E[t - 1]))) ** 2
                        + (I[t] - ((2 * sigma * E[t - 1] - gammaR * I[t - 1] - gammaD * I[t - 1]))) ** 2
                        + (R[t] - (gammaR * I[t - 1])) ** 2
                        + (D[t] - (gammaD * I[t - 1])) ** 2)
                # print(obj)
            else:
                obj += ((S[t] - (S[t - 2] - 2 * beta * S[t - 1] * I[t - 1] / N)) ** 2
                        + (E[t] - (E[t - 2] + (2 * beta * S[t - 1] * I[t - 1] / N - sigma * E[t - 1]))) ** 2
                        + (I[t] - (I[t - 2] + (2 * sigma * E[t - 1] - gammaR * I[t - 1] - gammaD * I[t - 1]))) ** 2
                        + (R[t] - (R[t - 2] + gammaR * I[t - 1])) ** 2
                        + (D[t] - (D[t - 2] + gammaD * I[t - 1])) ** 2)

                # print(obj)
            t += 1

            # Set objective
        m.setObjective(obj, GRB.MINIMIZE)

        # Add constraint:
        # m.addConstr(beta >= 0.0, "c0")
        # m.addConstr(sigma >= 0.0, "c1")
        # m.addConstr(gammaR >= 0.0, "c2")
        # m.addConstr(gammaD >= 0.0, "c3")

        # Optimize model
        m.optimize()

    #     for v in m.getVars():  # pega todas as variáveis
    #         print('%s %g' % (v.varName, v.x))
    #
    #     print('Obj: %g' % m.objVal)  # printa o total da função objetivo

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')
    print('----------Dia{}------ '.format(t0))
    for v in m.getVars():  # pega todas as variáveis
        if v.x < 0:
            print('alguem menor que 0',v.varName,v.x)
        else:
            print(v.varName,v.x,'Tudo Ok')

    constantes = []
    for v in m.getVars():
        constantes += [v.x]
    return constantes


#print(otimizar(7, 750, S, E, I, R, D))


def solver(latencia, t0, len, S, E, I, R, D):
    # Total population, N.
    N = 569733.0
    # Initial number of infected and recovered individuals, I0 and R0.
    E0, I0, R0,  D0 = E[t0], I[t0], R[t0], D[t0]
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - E0 - I0 - R0 - D0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    #print(otimizar(latencia, t0, S, E, I, R, D))
    beta, sigma, gammaR, gammaD = otimizar(latencia, t0, S, E, I, R, D)
    # A grid of time points (in days)
    t = np.linspace(0, len, len)

    # The SEIRD model differential equations.
    def deriv(y, t, N, beta, sigma, gammaR, gammaD):
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gammaR * I - gammaD * I
        dRdt = gammaR * I
        dDdt = gammaD * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    # Initial conditions vector
    y0 = S0, E0, I0, R0, D0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, sigma, gammaR, gammaD))
    S, E, I, R, D = ret.T
    return S, E, I, R, D, t


def graficos_londrina_optimix(latencia, t0, len, S, E, I, R, D):
    S, E, I, R, D, t = solver(latencia, t0, len, S, E, I, R, D)
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    #fig = plt.figure(facecolor='w', figsize=[14, 7])
    #ax = fig.subplots(1)#, facecolor='#dddddd', axisbelow=True)
    # ax.plot(t, S, 'b', alpha=0.75, lw=2, label='Susceptible')
    # ax.plot(t, E, 'c', alpha=0.75, lw=2, label='Exposed')
    plt.plot(t+t0, I, 'y', alpha=0.75, lw=2, label='Infected')
    # ax.plot(t, R, 'g', alpha=0.75, lw=2, label='Recovered with immunity')
    # ax.plot(t, D, 'r', alpha=0.75, lw=2, label='Dead')
    plt.plot(t, inf_atuais_londrina(df,latencia,tempo_ate_diag)[2], alpha=0.75, lw=2, label='Infectados Londrina')
    # ax.set_xlabel('Time /days')
    # ax.set_ylabel('Number (1000s)')
    plt.ylim(0,4000)
    plt.xlim(0, 200)
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    # for spine in ('top', 'right', 'bottom', 'left'):
    #     ax.spines[spine].set_visible(False)
    #return fig

latencia = 14
S, E, I, R, D = inf_atuais_londrina(df, latencia, tempo_ate_diag)
N = 569733.0
#Para o dia 70 com latencia = 14
i=70
otimizar(latencia, i, S, E, I, R, D)
#print(inf_atuais_londrina(df,latencia,tempo_ate_diag)[3][i:i+latencia])
inf_atuais_londrina(df,latencia,tempo_ate_diag)[3][i:i+latencia].plot()
inf_atuais_londrina(df,latencia,tempo_ate_diag)[4][i:i+latencia].plot()
plt.savefig('Teste_mesa_grafico_base_dia_70_latencia_14.png')
plt.show()

#Para o dia 68 com latencia = 3
latencia = 3
S, E, I, R, D = inf_atuais_londrina(df, latencia, tempo_ate_diag)
N = 569733.0
i=68
otimizar(latencia, i, S, E, I, R, D)
#print(londrina_inf_atuais.rec_total_dia[i:i+latencia])
inf_atuais_londrina(df,latencia,tempo_ate_diag)[3][i:i+latencia].plot()
inf_atuais_londrina(df,latencia,tempo_ate_diag)[4][i:i+latencia].plot()
plt.savefig('Teste_mesa_grafico_base_dia_68_latencia_3.png')
plt.show()



#for i in range(len(londrina_inf_atuais.index)-latencia):
# for i in range(5,200,5):
#     graficos_londrina_optimix(latencia, i, len(londrina_inf_atuais.index), S, E, I, R, D)
#     #plt.show()
# plt.show()
