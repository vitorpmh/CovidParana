import datetime
import urllib
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
import requests as requests

# lugar onde o arquivo csv esta salvo
url = "df.csv"

# lemos o arquivo que temos salvo
last_df = pd.read_csv(url, sep=';', low_memory=False)

# criamos uma xdf para nao corromper a last_df
xdf = last_df

# selecionamos todas as colunas que comecam o texto como data
datas = [x for x in xdf.columns if x.startswith("DATA")]

# arrumamos todas as datas dessas colunas
for col in datas:
    xdf[col] = pd.to_datetime(xdf[col], errors="coerce")

# é a data mais recente da base de dados salva a priori
data_mais_recente = xdf[datas].max().max()


# eh uma funcao que checa se uma url é funcional(codigo200)
# e retorna um true ou false
def url_ok(url):
    status_code = urllib.request.urlopen(url).getcode()
    # codigo 200 quer dizer que a pagina está funcionando
    website_is_up = status_code == 200
    return website_is_up

    # def url_ok(url):
    #     r = requests.head(url)
    #     return r.status_code == 200


# se ontem eh igual a data mais recente da base de dados que
# temos salva, utilize a salva
if (dt.today() - timedelta(days=1)).strftime(
        "%Y-%m-%d") == data_mais_recente.strftime("%Y-%m-%d"):
    url = "df.csv"
    df = pd.read_csv(url, sep=';', low_memory=False)
    # criamos uma new_df para nao corromper a df
    new_df = df
    # avisando que utilizou a df que tinhamos salvo
    print("Utilizei df já salva")
else:
    # avisando que vai baixar uma df nova
    print("Baixarei uma df nova")
    # dia de ontem no formato aaaa-mm
    anomes = (dt.today() - timedelta(days=1)).date().strftime("%Y-%m")
    # dia de ontem no formato dd_mm_aaaa
    ano_mes_dia = (dt.today() - timedelta(days=1)).date().strftime("%d_%m_%Y")
    # Temos dois textos na url da SESA um em MAIUS. e outro em minusculo
    texto = ["informe_epidemiologico_{}_geral.csv".format(ano_mes_dia),
             "INFORME_EPIDEMIOLOGICO_{}_Geral.csv".format(ano_mes_dia)]

    for texto in texto:
        url = ("https://www.saude.pr.gov.br/sites/default/arquivos_restritos"
               "/files/documento/{}/{}").format(anomes, texto)
        print(url)
        if url_ok(url):  # se a url ta OK entao... (lembrar da funçao acima)
            print(url)  # printa a url que vamos baixar

            new_df = pd.read_csv(url, sep=';', low_memory=False)
            new_df.to_csv('df.csv', sep=';', index=False)
            break  # paramos o loop for caso o primeiro texto tenha funcionado

# selecionamos todas as colunas que comecam o texto como data
datas = [x for x in new_df.columns if x.startswith("DATA")]

# arrumamos todas as datas dessas colunas
for col in datas:
    new_df[col] = pd.to_datetime(new_df[col], errors="coerce")

# é a data mais recente da base de dados mais recente provavelmente
# a baixada agora
data_mais_recente2 = new_df[datas].max().max()

# setamos o tempo de latencia que queremo
latencia = 14

# setamos o tempo médio do inicio de sintomas até o diagnóstico
tempo_ate_diag = 5

# Aqui arrumamos a coluna dos recuperados e salvamos na variavel df_recalculada
if 'df_recalculada' not in locals():
    df_recalculada = df.select_dtypes(include=['datetime', 'number'])

    DIS = df_recalculada['DATA_INICIO_SINTOMAS']
    DD = df_recalculada['DATA_DIAGNOSTICO']
    DR = df_recalculada['DATA_RECUPERADO_DIVULGACAO']
    DO = df_recalculada['DATA_OBITO']

    df_recalculada['nova_data_infec'] = np.where(
        ((DD - DIS).dt.days >= tempo_ate_diag) | (
                    (DD - DIS).dt.days < 0) | DIS.isna(),
        DD - datetime.timedelta(days=(tempo_ate_diag)),  # verdadeiro
        DIS)  # falso np.datetime64("NaT")

    NDI = df_recalculada['nova_data_infec']

    df_recalculada['nova_data_infec'] = np.where(
        DO < DD,
        DO - datetime.timedelta(days=(tempo_ate_diag)),  # verdadeiro
        NDI)  # falso np.datetime64("NaT")

    df_recalculada['nova_data_rec'] = np.where(
        DO.isna() & (DR.isna() | (DR < NDI) | (DR < DD) | (
                    (DR - NDI).dt.days >= latencia)),
        NDI + datetime.timedelta(days=latencia),  # verdadeiro
        DR)  # falso np.datetime64("NaT")

    NDR = df_recalculada['nova_data_rec']

    df_recalculada = df_recalculada.sort_values(
        by=['nova_data_infec']).reset_index(drop=True)

# criando a base df_parana para não corromper a df_recalculada
#df_parana = df_recalculada


def inf_atuais(df, idx, i):
    if idx <= 26:
        # se a variavel é df_parana, utilizer as 3 colunas
        if df is df_recalculada:
            df_confPorDia = df_recalculada[
                ['nova_data_infec', 'nova_data_rec', 'DATA_OBITO']].apply(
                pd.Series.value_counts).expanding(min_periods=2).sum()
        # caso contrario utilizar a função isin e checar se tem os codigos
        # ibges, apos isso somar por dia
        # todos as 3 colunas para geral as 3 colunas de inf, rec, morto totais
        else:
            df_confPorDia = \
                df_recalculada[
                    df_recalculada['IBGE_RES_PR'].isin(df['codigo_ibge'])
                ][['nova_data_infec', 'nova_data_rec', 'DATA_OBITO']].apply(
                pd.Series.value_counts).expanding(min_periods=2).sum()
    # caso contrario utilizar a função isin e checar se tem os codigos ibges,
    # apos isso somar por dia
    # todos as 3 colunas para geral as 3 colunas de inf, rec, morto totais
    else:
        df_confPorDia = \
        df_recalculada[df_recalculada['IBGE_RES_PR'].isin([df.iloc[i, 1]])][
            ['nova_data_infec', 'nova_data_rec', 'DATA_OBITO']].apply(
            pd.Series.value_counts).expanding(min_periods=2).sum()

    # Troco todos os valores NaN por 0
    df_confPorDia = df_confPorDia.fillna(0)

    # renomeando as colunas para os nomes corretos
    df_confPorDia.columns = ['inf_total_dia', 'rec_total_dia',
                             'morto_total_dia']

    # criando a coluna de inf atuais (como a total é por dia basta eu
    # subtrair do total de infectados o total de mortos e recuperados)
    df_confPorDia['inf_atuais'] = df_confPorDia.apply(
        lambda row: row['inf_total_dia'] - row['morto_total_dia'] - row[
            'rec_total_dia'], axis=1)

    # crio uma coluna chamada data
    df_confPorDia['Data'] = df_confPorDia.index

    return df_confPorDia
