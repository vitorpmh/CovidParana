# Aqui estamos importando todas as bases de dado de cidades no arquivo BaseIBGE.py
from BaseIBGE import *

# Estamos importando a base de dados do parana com os recuperados ja organizados na df_parana que é uma planilha de
# todos os ifectados da Covid no paraná
from dadosParana import df_parana

TodasBaseDados = [df_parana, Macrorregional_Noroeste, Macrorregional_Norte, Macrorregional_Leste,
                  Macrorregional_Oeste, regional_saude_1, regional_saude_2, regional_saude_3, regional_saude_4,
                  regional_saude_5, regional_saude_6, regional_saude_7, regional_saude_8, regional_saude_9,
                  regional_saude_10, regional_saude_11, regional_saude_12, regional_saude_13, regional_saude_14,
                  regional_saude_15, regional_saude_16, regional_saude_17, regional_saude_18, regional_saude_19,
                  regional_saude_20, regional_saude_21, regional_saude_22, maior_cidade_ibge, cidade_ibge]

# Aqui estamos importanto a função plot() e quatro figuras em branco que serão preenchidas quando a função plot()
# for executada
from graficos import plot, fig, fig2, fig3, fig4

plot(TodasBaseDados) # preenche as figuras fig, fig2, fig3, fig4 com os graficos dos infectados atuais

figuras = [fig,fig2,fig3,fig4] # salva os gráficos nesse array para apresentar eles
for fig in figuras:
    fig.show()