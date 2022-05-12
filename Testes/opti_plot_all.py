import datetime
import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from gurobipy import GRB

from scipy.integrate import odeint


tempo_ate_diag = 5
latencia = 14

df_cidades = pd.read_csv('df_cidades.csv')

df = pd.read_csv('df_2022_04_04.csv', sep=';', low_memory=False)


def datas(df):
    return [x for x in df.columns if x.startswith("DATA")]


for col in datas(df):
    df[col] = pd.to_datetime(df[col], errors="coerce")


def data_mais_recente_df_saved(df_saved):
    global data_mais_recente

    last_df = pd.read_csv(df_saved, sep=';', low_memory=False)

    for cols in datas(last_df):
        last_df[cols] = pd.to_datetime(last_df[cols], errors="coerce")

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

    # criamos uma nova coluna para arrumar os infectados com base no inicio de sintomas, se o tempo entre DIS e DD
    # for >= 5 dias trocamos por DIS + 5, caso contrario por DD. Funçao np.where que diz: onde temos isso,
    # troque por isso caso verdadeiro, ou troque por isso caso falso.
    df_recalculada['nova_data_infec'] = np.where(
        (df_recalculada['DATA_DIAGNOSTICO'] - df_recalculada['DATA_INICIO_SINTOMAS']).dt.days >= 5,
        df_recalculada.DATA_INICIO_SINTOMAS + datetime.timedelta(days=tempo_ate_diag),  # verdadeiro
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
                                                 (data_recuperado < data_mais_recente)) |
                                                (df_recalculada.DATA_RECUPERADO_DIVULGACAO >
                                                 data_recuperado)),
                                               data_recuperado,  # verdadeiro
                                               df_recalculada.DATA_RECUPERADO_DIVULGACAO)  # falso

    df_recalculada = df_recalculada.sort_values(
                        by=['DATA_DIAGNOSTICO']).reset_index(drop=True)
    return df_recalculada

df_recalculada = recalcular_df(df, latencia, tempo_ate_diag)




def inf_atuais(df, cidades ,populacao, latencia):
    confPorDia = df[df['IBGE_RES_PR'].isin(cidades)]
    confPorDia = confPorDia[['nova_data_infec', 'nova_data_rec', 'DATA_OBITO']]
    confPorDia = confPorDia.apply(pd.Series.value_counts).expanding(min_periods=1).sum()



    first_case = df[datas(df)].min().min()
    last_case = df[datas(df)].max().max()
    new_date_range = pd.date_range(first_case, last_case, freq="D")
    confPorDia = confPorDia.reindex(new_date_range)
    confPorDia = confPorDia.ffill().bfill()

    inf_atuais = confPorDia.fillna(0)
    inf_atuais.columns = ['inf_total_dia', 'rec_total_dia', 'morto_total_dia']
    inf_atuais['inf_atuais'] = inf_atuais.apply(
        lambda row: row['inf_total_dia'] - row['morto_total_dia'] - row
            ['rec_total_dia'], axis=1)
    inf_atuais['Data'] = inf_atuais.index

    # suscetiveis = N - Infectados totais
    inf_atuais['suscetiveis'] = populacao - inf_atuais['inf_total_dia']

    return inf_atuais.suscetiveis, \
           inf_atuais.inf_atuais, \
           inf_atuais.rec_total_dia, \
           inf_atuais.morto_total_dia

def otimizar(populacao, latencia, t0, S, I, R, D):
    try:

        # Create a new model
        m = gp.Model("qp") #qp
        #        m.params.NonConvex = 2
        #        m.params.NumericFocus=1
        m.params.LogToConsole = 0
        # Create variables
        beta = m.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, name="beta")
        gammaR = m.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, name="gammaR")
        gammaD = m.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, name="gammaD")

        t = t0
        obj = 0.0
        while t0 <= t < t0 + latencia:
            if t == 0: #s(t-2) = 0, s(t-1)=0
                obj += (S[t]) ** 2 + (I[t]) ** 2 + (R[t]) ** 2 + (D[t]) ** 2
                # print(obj)
            elif t == 1: #s(t-2) = 0
                obj += ((S[t] + 2 * beta * S[t - 1] * I[t - 1] / populacao) ** 2
                        + (I[t] - 2 * (beta * S[t - 1] * I[t - 1] / populacao
                                       - gammaR * I[t - 1]
                                       - gammaD * I[t - 1])) ** 2
                        + (R[t] - 2 * gammaR * I[t - 1]) ** 2
                        + (D[t] - 2 * gammaD * I[t - 1]) ** 2)
                # print(obj)
            else:
                obj += ((S[t] -
                         (S[t - 2] - 2 * beta * S[t - 1] * I[t - 1] / populacao)) ** 2
                        + (I[t] - (I[t - 2] + 2 *
                                   (beta * S[t - 1] * I[t - 1] / populacao
                                    - gammaR * I[t - 1]
                                    - gammaD * I[t - 1]))) ** 2
                        + (R[t] - (R[t - 2] + 2 * gammaR * I[t - 1])) ** 2
                        + (D[t] - (D[t - 2] + 2 * gammaD * I[t - 1])) ** 2)
                m.update()
            t += 1

            # Set objective
        m.setObjective(obj, GRB.MINIMIZE)

        # Add constraint:
        m.addConstr(beta >= 0.0, "c0")
        m.addConstr(gammaR >= 0.0, "c1")
        m.addConstr(gammaD >= 0.0, "c2")

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
            print('alguem menor que 0', v.varName, v.x)
        else:
            print(v.varName, v.x, 'Tudo Ok')

    constantes = []
    for v in m.getVars():
        constantes += [v.x]
    return constantes



def solver(populacao, latencia, t0, lenght, S, I, R, D):
    N = populacao

    I0, R0, D0 = I[t0], R[t0], D[t0]

    S0 = N - I0 - R0 - D0

    beta, gammaR, gammaD = otimizar(populacao, latencia, t0, S, I, R, D)

    t = np.linspace(0, lenght, lenght)

    # The SIRD model differential equations.
    def deriv(y, t, N, beta, gammaR, gammaD):
        S, I, R, D = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gammaR * I - gammaD * I
        dRdt = gammaR * I
        dDdt = gammaD * I
        return dSdt, dIdt, dRdt, dDdt

    # Initial conditions vector
    y0 = S0, I0, R0, D0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gammaR, gammaD))
    S, I, R, D = ret.T
    return S, I, R, D, t


tamanho_bolinha = 2
def graficos_otimizado(fig, df, cidades, populacao, latencia, t0):
    S, I, R, D = inf_atuais(df ,cidades ,populacao ,latencia)
    df_length = len(S)
    Ss, Is, Rs, Ds, t = solver(populacao, latencia, t0, df_length, S, I, R, D)

    # suscetiveis
    #     ax.plot((t+ t0), Is, 'b', alpha=0.75, lw=2,
    #             color=colors[t0 % len(colors)])

    fig.add_traces(go.Scatter(x=( t+ t0) ,y=Is, mode='lines'))
    fig.add_traces \
        (go.Scatter(x=t ,y=I ,marker_size=tamanho_bolinha ,mode='markers'))



fig = go.Figure()
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()

for i in range(500,501,1):#range(500, 600, 20)
    for cod_ibge in df_cidades.codigos_ibge:
        populacao = df_cidades[df_cidades.codigos_ibge == cod_ibge][
            'populacao'].item()
        print(df_cidades[df_cidades.codigos_ibge == cod_ibge][
                  'nome_cidades'].item())
        graficos_otimizado(fig, df_recalculada, [cod_ibge], populacao,
                           latencia, i)


    for regiao in df_cidades.columns[4:]:
        lista_cidades_por_regiao = list(
            df_cidades[df_cidades[f'{regiao}'] == True].codigos_ibge)
        if regiao == "cidades_grande":
            for cities in lista_cidades_por_regiao:
                populacao = df_cidades[df_cidades.codigos_ibge == cities][
                    'populacao'].item()
                graficos_otimizado(fig1, df_recalculada, [cities], populacao,
                                   latencia, i)
                print(cities)

        elif "macro" in regiao:
            populacao = df_cidades[df_cidades[regiao] == True][
                'populacao'].sum()
            graficos_otimizado(fig2, df_recalculada, lista_cidades_por_regiao,
                               populacao, latencia, i)
            print(regiao)

        else:
            populacao = df_cidades[df_cidades[regiao] == True]['populacao'].sum()
            graficos_otimizado(fig3, df_recalculada, lista_cidades_por_regiao,
                               populacao, latencia, i)
            print(regiao)


figs = [fig,fig1,fig2,fig3]
for fig in figs:
    fig.show()


