import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Configuração da página Streamlit
st.title("Sistema de Previsão de Risco do Cliente com Pipeline")
st.markdown("##### Desenvolvido por Mauro Guimarães")

# Subtítulo
st.markdown("Este é um Aplicativo utilizado para exibir a solução de Ciência de Dados para o problema de predição de Risco do Cliente para concessão de empréstimos.")

# Função para carregar e preparar os dados
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("risco.csv")
        if 'id_cliente' in data.columns:
            data = data.drop(columns='id_cliente')
        # Converter valores de risco para numérico
        data['Risco'] = (data['Risco'] == 'Risco_Alto').astype(int)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {str(e)}")
        return None

# Função para treinar o modelo
def train_model(data):
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    
    # Padronização com StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Treinamento do modelo
    model = SVC(kernel='linear', gamma=1e-5, C=10, random_state=7)
    model.fit(X, y)
    
    return model, scaler

# Carregar dados
data = load_data()

if data is not None:
    # Análise dos dados por classe de risco
    st.subheader("Análise dos Dados por Classe de Risco")
    
    # Separar dados por classe de risco
    alto_risco = data[data.iloc[:,-1] == 1]
    baixo_risco = data[data.iloc[:,-1] == 0]
    
    st.write("Distribuição das classes:")
    risco_counts = pd.Series({
        'Alto Risco': len(alto_risco),
        'Baixo Risco': len(baixo_risco)
    })
    st.write(risco_counts)
    
    # Calcular médias por classe
    if not alto_risco.empty and not baixo_risco.empty:
        media_alto_risco = alto_risco.mean()
        media_baixo_risco = baixo_risco.mean()
        
        # Criar DataFrame comparativo
        comparativo = pd.DataFrame({
            'Média Alto Risco': media_alto_risco,
            'Média Baixo Risco': media_baixo_risco
        }).iloc[:-1]  # Removendo a última linha que é a média da variável target
        
        st.write("\nMédias das variáveis por classe de risco:")
        st.write(comparativo)
    
    # Treinar modelo
    model, scaler = train_model(data)
    
    # Interface para previsões
    st.sidebar.subheader("Insira os Dados dos Clientes para Análise do Risco")

    # Valores sugeridos para alto risco baseados nas médias
    valores_sugeridos = {
        'indice_inad': min(float(data['indice_inad'].mean() * 1.2), float(data['indice_inad'].max())),  # 20% maior que a média, limitado ao máximo
        'anot_cadastrais': min(float(data['anot_cadastrais'].mean() * 1.2), float(data['anot_cadastrais'].max())),  # 20% maior que a média, limitado ao máximo
        'class_renda': float(data['class_renda'].quantile(0.25)),  # 25º percentil
        'saldo_contas': float(data['saldo_contas'].quantile(0.25))  # 25º percentil
    }

    # Campos de entrada
    indice_inad = st.sidebar.number_input("Índice de Inadimplência", 
                                        value=valores_sugeridos['indice_inad'],
                                        min_value=float(data['indice_inad'].min()),
                                        max_value=float(data['indice_inad'].max()))
    anot_cadastrais = st.sidebar.number_input("Anotações Cadastrais", 
                                            value=valores_sugeridos['anot_cadastrais'],
                                            min_value=float(data['anot_cadastrais'].min()),
                                            max_value=float(data['anot_cadastrais'].max()))
    class_renda = st.sidebar.number_input("Classificação da Renda", 
                                        value=valores_sugeridos['class_renda'],
                                        min_value=float(data['class_renda'].min()),
                                        max_value=float(data['class_renda'].max()))
    saldo_contas = st.sidebar.number_input("Saldo de Contas", 
                                         value=valores_sugeridos['saldo_contas'],
                                         min_value=float(data['saldo_contas'].min()),
                                         max_value=float(data['saldo_contas'].max()))

    # Botão de previsão
    btn_predict = st.sidebar.button("Realizar Predição do Risco")

    # Área principal para exibição dos resultados
    if btn_predict:
        # Preparar dados de entrada
        input_data = np.array([[indice_inad, anot_cadastrais, class_renda, saldo_contas]])
        
        # Padronizar os dados de entrada
        input_data_scaled = scaler.transform(input_data)
        
        # Fazer previsão
        result = model.predict(input_data_scaled)
        
        st.markdown("---")  # Linha superior
        st.subheader("O Risco Previsto do Cliente é:")
        
        if result[0] == 1:
            st.error("Alto Risco")
        else:
            st.success("Baixo Risco")
        st.markdown("---")  # Linha inferior
        
        # Mostrar os valores utilizados
        st.subheader("Valores utilizados para a previsão:")
        valores_entrada = pd.DataFrame({
            'Variável': ['Índice de Inadimplência', 'Anotações Cadastrais', 'Classificação da Renda', 'Saldo de Contas'],
            'Valor': [indice_inad, anot_cadastrais, class_renda, saldo_contas]
        })
        st.write(valores_entrada)

    # Mostrar informações dos dados
    st.subheader("Amostra dos Dados")
    st.write(data.head())
    
    # Mostrar estatísticas
    st.subheader("Estatísticas Descritivas")
    st.write(data.describe())
