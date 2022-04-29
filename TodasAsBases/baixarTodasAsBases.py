import os
import urllib
from datetime import datetime as dt
from datetime import timedelta
from datetime import date


import requests as requests
import pandas as pd

#primeiro dia que foi disponibilixada a base geral de dados da covid
date_first_df = "2020-07-25"


# eh uma funcao que checa se uma url é funcional(codigo200)
# e retorna um true ou false
def url_ok(url):
    status_code = urllib.request.urlopen(url).getcode()
    # codigo 200 quer dizer que a pagina está funcionando
    website_is_up = status_code == 200
    return website_is_up


# se ontem eh igual a data mais recente da base de dados que
# temos salva, utilize a salva



# dia de ontem no formato aaaa-mm
ontem_ano_mes = (dt.today() - timedelta(days=1)).date().strftime("%Y-%m")
# dia de ontem no formato dd_mm_aaaa
ontem_ano_mes_dia = (dt.today() - timedelta(days=1)).date().strftime("%d_%m_%Y")

ano_mes_dia = (dt.today() - timedelta(days=1)).date().strftime("%d_%m_%Y")




dias_ate_hoje = range((dt.today().date() - timedelta(days=1)- date(2020,7,25)).days)


all_files = os.listdir("./bases")
csv_files = list(
    filter(lambda f: f.endswith('.csv'), all_files))
print(csv_files)

for dias_passados in dias_ate_hoje:
    try:
        ano_mes = (date(2020, 7, 25) + timedelta(days=dias_passados)).strftime(
            "%Y-%m")
        ano_mes_dia = (date(2020, 7, 25) + timedelta(days=dias_passados)).strftime(
            "%d_%m_%Y")

        ano_mes_dia_international = (date(2020, 7, 25) + timedelta(
                                     days=dias_passados)).strftime(
                                     "%Y_%m_%d")

        texto = ["informe_epidemiologico_{}_geral.csv".format(ano_mes_dia),
                 "INFORME_EPIDEMIOLOGICO_{}_Geral.csv".format(ano_mes_dia)]

        for texto in texto:
            url = ("https://www.saude.pr.gov.br/sites/default/arquivos_restritos"
                   "/files/documento/{}/{}").format(ano_mes, texto)
            if url_ok(url):  # se a url ta OK entao... (lembrar da funçao acima)
                filename = f'df_parana_{ano_mes_dia_international}.csv'

                if filename not in csv_files:
                    print(ano_mes_dia, "sera baixada")
                    new_df = pd.read_csv(url, sep=';', low_memory=False)
                    new_df.to_csv(f'./bases/{filename}', sep=';', index=False,
                                  compression="gzip")
                else:
                    print(ano_mes_dia, "ja baixado")
                break  # paramos o loop for caso o primeiro texto tenha funcionado
    except:
        pass