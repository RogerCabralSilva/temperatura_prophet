import pandas as pd
import streamlit as st
import json
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly
import streamlit as st


def load_model():
    with open('modelo_temp_prophet.json', 'r') as file_in:
        modelo = model_from_json(json.load(file_in))
        return modelo


modelo = load_model()

# Inicializando estado da sessão
if 'historico_previsoes' not in st.session_state:
    st.session_state['historico_previsoes'] = pd.DataFrame(
        columns=['Data', 'Temperatura Prevista (°C)'])

# Adicionando textos ao layout do Streamlit
st.title('Previsão de Temperatura (°C) Utilizando a Biblioteca Prophet')

st.caption('''Este projeto utiliza a biblioteca Prophet para prever Temperatura. O modelo
           criado foi treinado com dados até o dia 05/05/2023 e possui um erro de previsão (RMSE - Erro Quadrático Médio) igual a 2.69 nos dados de teste.
           O usuário pode inserir o número de dias para os quais deseja a previsão, e o modelo gerará um gráfico
           interativo contendo as estimativas baseadas em dados históricos da temperatura.
           Além disso, uma tabela será exibida com os valores estimados para cada dia.''')

st.subheader('Insira o número de dias para previsão:')
dias = st.number_input('Número de dias', min_value=1, value=1, step=1)

# Alerta de temperatura
limite_alerta = st.number_input(
    'Defina o limite de alerta de temperatura (°C):', min_value=0.0, value=30.0)

if 'previsao_feita' not in st.session_state:
    st.session_state['previsao_feita'] = False
    st.session_state['dados_previsao'] = None

if st.button('Prever'):
    st.session_state.previsao_feita = True
    futuro = modelo.make_future_dataframe(periods=dias, freq="D")
    previsao = modelo.predict(futuro)
    st.session_state['dados_previsao'] = previsao

if st.session_state.previsao_feita:
    # Gráfico de previsão
    fig = plot_plotly(modelo, st.session_state['dados_previsao'])

    fig.update_layout({
        'plot_bgcolor': 'rgba(240, 240, 240, 1)',  # Cor de fundo do gráfico
        'paper_bgcolor': 'rgba(240, 240, 240, 1)',  # Cor de fundo externo
        'title': {'text': "Previsão de Temperatura", 'font': {'color': '#333'}},
        'xaxis': {'title': 'Data', 'title_font': {'color': '#333'}, 'tickfont': {'color': '#333'}},
        'yaxis': {'title': 'Temperatura (°C)', 'title_font': {'color': '#333'}, 'tickfont': {'color': '#333'}},
        'legend': {'font': {'color': '#333'}}
    })

    st.plotly_chart(fig)

    # Verificar se alguma temperatura prevista excede o limite
    max_temp = st.session_state['dados_previsao']['yhat'].max()
    if max_temp > limite_alerta:
        st.warning(
            f'🔴 Alerta: A temperatura prevista máxima ({round(max_temp, 2)} °C) excede o limite de alerta definido ({limite_alerta} °C)!')

# Tabela de previsão
if st.session_state['dados_previsao'] is not None:
    previsao = st.session_state['dados_previsao']
    tabela_previsao = previsao[['ds', 'yhat']].tail(dias)
    tabela_previsao.columns = ['Data (Dia/Mês/Ano)', 'TEMP']
    tabela_previsao['Data (Dia/Mês/Ano)'] = tabela_previsao['Data (Dia/Mês/Ano)'].dt.strftime('%d-%m-%Y')
    tabela_previsao['TEMP'] = tabela_previsao['TEMP'].round(2)
    tabela_previsao.reset_index(drop=True, inplace=True)
    st.write(
        'Tabela contendo as previsões de Temperatura (TEMP) para os próximos {} dias:'.format(dias))
    st.dataframe(tabela_previsao, height=300)

    # Download
    csv = tabela_previsao.to_csv(index=False)
    st.download_button(label='Baixar tabela como csv', data=csv,
                       file_name='previsao_ozonio.csv', mime='text/csv')
