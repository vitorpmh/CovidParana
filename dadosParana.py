import datetime
import urllib
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import requests as requests


url = "df.csv" # lugar onde o arquivo csv esta salvo
last_df = pd.read_csv(url, sep=';', low_memory=False) # lemos o arquivo que temos salvo

xdf = last_df # criamos uma xdf para nao corromper a last_df

datas = [x for x in xdf.columns if x.startswith("DATA")] # selecionamos todas as colunas que comecam o texto como data

for col in datas:
    xdf[col] = pd.to_datetime(xdf[col], errors="coerce") # arrumamos todas as datas dessas colunas


data_mais_recente = xdf[datas].max().max() # é a data mais recente da base de dados salva a priori

# eh uma funcao que checa se uma url é funcional(codigo200) e retorna um true ou false
def url_ok(url):
    status_code = urllib.request.urlopen(url).getcode()
    website_is_up = status_code == 200
    return website_is_up  # codigo 200 quer dizer que a pagina está funcionando

    # def url_ok(url):
    #     r = requests.head(url)
    #     return r.status_code == 200

# se ontem eh igual a data mais recente da base de dados que temos salva, utilize a salva
if (dt.today() - timedelta(days=1)).strftime("%Y-%m-%d") == data_mais_recente.strftime("%Y-%m-%d"):
    url = "df.csv"
    df = pd.read_csv(url, sep=';', low_memory=False)
    new_df = df # criamos uma new_df para nao corromper a df
    print("Utilizei df já salva") # avisando que utilizou a df que tinhamos salvo
else:
    print("Baixarei uma df nova") # avisando que vai baixar uma df nova
    anomes = (dt.today() - timedelta(days=1)).date().strftime("%Y-%m")# dia de ontem no fromado aaaa-mm
    ano_mes_dia = (dt.today() - timedelta(days=1)).date().strftime("%d_%m_%Y") # dia de ontem no formato dd_mm_aaaa
    texto = ["informe_epidemiologico_{}_geral.csv".format(ano_mes_dia), # temos dois textos na url da sesa um em caps
             "INFORME_EPIDEMIOLOGICO_{}_Geral.csv".format(ano_mes_dia)] # e outro em minusculo

    for texto in texto:
        url = "https://www.saude.pr.gov.br/sites/default/arquivos_restritos/files/documento/{}/{}".format(anomes, texto)
        print(url)
        if url_ok(url): # se a url ta OK entao... (lembrar da funçao acima)
            print(url) # printa a url que vamos baixar

            df = pd.read_csv(url, sep=';', low_memory=False)
            df.to_csv('df.csv', sep=';', index=False)
            new_df = df # criamos uma variavel _df para nao corromper a variavel df
            break # paramos o loop for caso o primeiro texto tenha funcionado

datas = [x for x in new_df.columns if x.startswith("DATA")] # selecionamos todas as colunas que comecam o texto como data

for col in datas:
    new_df[col] = pd.to_datetime(new_df[col], errors="coerce") # arrumamos todas as datas dessas colunas




data_mais_recente2 = new_df[datas].max().max() # é a data mais recente da base de dados mais recente

latencia = 14 # setamos o tempo de latencia que queremos
tempo_ate_diag = 5 # setamos o tempo médio do inicio de sintomas até o diagnóstico

# Aqui arrumamos a coluna dos recuperados e salvamos na variavel df_recalculada
if 'df_recalculada' not in locals():
    #selecionamos somente as colunas que possuem formato de data e numeros(codigosIBGE).
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
    data_recuperado = df_recalculada.DATA_DIAGNOSTICO + datetime.timedelta(days=latencia - tempo_ate_diag)

    # aqui comecamos a restruturaço da coluna de recuperados utilizamos a funçao np.where que diz: onde temos isso,
    # troque por isso caso verdadeiro, ou troque por isso caso falso.
    df_recalculada['nova_data_rec'] = np.where((df_recalculada.DATA_OBITO.isna()) &
                                               ((df_recalculada.DATA_RECUPERADO_DIVULGACAO.isna() &
                                                 ((data_recuperado) < data_mais_recente)) |
                                                (df_recalculada.DATA_RECUPERADO_DIVULGACAO >
                                                 (data_recuperado))),
                                               data_recuperado,  # verdadeiro
                                               df_recalculada.DATA_RECUPERADO_DIVULGACAO)  # falso

    df_recalculada = df_recalculada.sort_values(by=['DATA_DIAGNOSTICO']).reset_index(drop=True)

# criando a base df_parana para não corromper a df_recalculada
df_parana = df_recalculada


def inf_atuais(df, idx, i):
    if idx <= 26:
        # se a variavel é df_parana, utilizer as 3 colunas
        if df is df_parana:
            df_confPorDia = df_parana[['nova_data_infec','nova_data_rec', 'DATA_OBITO']].apply(
                pd.Series.value_counts).expanding(min_periods=2).sum()
        else:  # caso contrario utilizar a função isin e checar se tem os codigos ibges, apos isso somar por dia
            # todos as 3 colunas para geral as 3 colunas de inf, rec, morto totais
            df_confPorDia = df_parana[df_parana['IBGE_RES_PR'].isin(df['codigo_ibge'])][
                ['nova_data_infec','nova_data_rec', 'DATA_OBITO']].apply(
                pd.Series.value_counts).expanding(min_periods=2).sum()
    else:  # caso contrario utilizar a função isin e checar se tem os codigos ibges, apos isso somar por dia
        # todos as 3 colunas para geral as 3 colunas de inf, rec, morto totais
        df_confPorDia = df_parana[df_parana['IBGE_RES_PR'].isin([df.iloc[i, 1]])][
            ['nova_data_infec','nova_data_rec', 'DATA_OBITO']].apply(
            pd.Series.value_counts).expanding(min_periods=2).sum()

    # Troco todos os valores NaN por 0
    df_confPorDia = df_confPorDia.fillna(0)

    # renomeando as colunas para os nomes corretos
    df_confPorDia.columns = ['inf_total_dia', 'rec_total_dia', 'morto_total_dia']

    # criando a coluna de inf atuais (como a total é por dia basta eu subtrair do total de infectados o total
    # de mortos e recuperados)
    df_confPorDia['inf_atuais'] = df_confPorDia.apply(
        lambda row: row['inf_total_dia'] - row['morto_total_dia'] - row['rec_total_dia'], axis=1)

    # crio uma coluna chamada data
    df_confPorDia['Data'] = df_confPorDia.index

    return df_confPorDia
