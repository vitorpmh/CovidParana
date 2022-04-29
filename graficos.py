import pandas as pd
import plotly.graph_objects as go

# Estamos importando a funcao inf_atuais() que retorna a base de dados que tem as colunas total de mortos, infectados e
# recuperado. E os infectados atuais.
from dadosParana import inf_atuais , df_parana

# criando as 4 figuras como objeto global
fig = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
fig4 = go.Figure()


# definindo a função de plot dos gráficos
def plot(base_dados):
    legendas = ["Parana", "Macrorregional Noroeste (Maringá)", "Macrorregional Norte (Londrina)",
                "Macrorregional Leste (Foz do Iguaçu)", "Macrorregional Oeste (Curitiba)"]

    # para toto idice e variavel em base_dados faça isso:
    for idx, var in enumerate(base_dados):
        # se o indice do array for menor ou igual a 4 criamos um grafico (parana e macros juntas)
        if idx <= 4:
            fig.add_traces(go.Scatter(x=inf_atuais(var,idx, 0)["Data"],
                                      y=inf_atuais(var,idx, 0)['inf_atuais'],
                                      mode='lines',
                                      name=legendas[idx]))

        # caso seja maior criamos uma figura só das regionais
        if 4 <idx <=26:
            regionais = 'Regional de Saúde {}'.format(idx - 4)
            legendas += [regionais]

            fig2.add_traces(go.Scatter(x=inf_atuais(var,idx, 0)["Data"],
                                       y=inf_atuais(var,idx, 0)['inf_atuais'],
                                       mode='lines',
                                       name=legendas[idx]))

        # igual a 27 é as maiores cidades
        if idx == 27:
            k = 0
            legenda_maior_cidades = []
            while k < len(var):
                IBGE = str(var.iloc[k, 0])
                legenda_maior_cidades += [IBGE]
                fig3.add_traces(go.Scatter(x=inf_atuais(var,idx, k)["Data"],
                                           y=inf_atuais(var,idx, k)['inf_atuais'],
                                           mode='lines',
                                           name=legenda_maior_cidades[k]))
                k += 1

        # igual a 28 é todas as cidades
        if idx == 28:
            i = 0
            legenda = []
            while i < len(var):
                IBGE = str(var.iloc[i, 0])
                legenda += [IBGE]
                fig4.add_traces(go.Scatter(x=inf_atuais(var,idx, i)["Data"],
                                           y=inf_atuais(var,idx, i)['inf_atuais'],
                                           mode='lines',
                                           name=legenda[i]))
                i += 1

    # aqui etamos deixando todas as figuras mais bonitas
    figuras = [fig, fig2, fig3, fig4]
    for figx in figuras:
        figx.update_layout(xaxis_title="Dia",
                           yaxis_title="Infectados Atuais",
                           legend=dict(itemwidth=30,
                                       borderwidth=0,
                                       font=dict(family="Courier New, monospace",
                                                 size=10,
                                                 color="Black")),
                           font=dict(family="Courier New, monospace",
                                     size=14,
                                     color="Black"))
    return fig, fig2, fig3, fig4
