import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Verifica e baixa 'punkt', caso ainda não estejam disponíveis
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Verifica e baixa 'stopwords', caso ainda não estejam disponíveis
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Carregar o dataset
file_path = "data/medical_data.csv"
def load_data(file_path):
    """
        Carrega o dataset de doenças, sintomas e tratamentos de um arquivo CSV.
        Args:
            file_path (str): Caminho para o arquivo CSV.
        Returns:
            pandas.DataFrame: DataFrame contendo os dados carregados.
    """
    print(f"Carregando dados de: {file_path}")
    # Usar o delimitador de vírgula e aspas para lidar com campos que contêm vírgulas
    df = pd.read_csv(file_path, sep=",", quotechar='"')
    return df

# Pré-processamento de texto
def preprocess_text(text):
    """
        Realiza o pré-processamento de texto, incluindo tokenização, 
        conversão para minúsculas, remoção de pontuação e stopwords.
        Args:
            text (str): O texto a ser pré-processado.
        Returns:
            list: Uma lista de tokens pré-processados.
    """
    if not isinstance(text, str): # Garantir que o input é uma string
        return []

    # Converter para minúsculas
    text = text.lower()
    # Remover pontuação e caracteres especiais, mantendo apenas letras e números
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Tokenização simples por espaço
    tokens = text.split()
    # Remover stopwords
    stop_words = set(stopwords.words("portuguese"))

    """
        Loop para filtrar tokens, removendo stopwords e tokens com apenas um caractere.
            - Isso ajuda a reduzir o ruído nos dados, mantendo apenas palavras relevantes para o treinamento de word embeddings.
    """
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return filtered_tokens

# Função principal para processar o dataset
def process_medical_data(file_path):
    """
        Processa o dataset médico, aplicando pré-processamento aos sintomas e tratamentos.
        Args:
            file_path (str): Caminho para o arquivo CSV do dataset.
        Returns:
            list: Uma lista de listas de tokens, pronta para treinamento de word embeddings.
    """
    df = load_data(file_path)
    
    processed_sentences = []
    """
        Itera sobre cada linha do DataFrame, aplicando o pré-processamento aos campos de sintomas, tratamentos e doenças.
        - Para cada campo, o texto é pré-processado e adicionado à lista de sentenças processadas, desde que haja tokens válidos após o pré-processamento.
        - Isso garante que apenas informações relevantes e limpas sejam usadas para o treinamento de word embeddings, melhorando a qualidade dos vetores gerados.
        'df.iterrows()' é usado para iterar sobre as linhas do DataFrame, permitindo acessar os valores de cada coluna para pré-processamento.
    """
    for index, row in df.iterrows():
        processed_symptoms = preprocess_text(row["Sintomas"])
        if processed_symptoms: # Adicionar apenas se houver tokens válidos
            processed_sentences.append(processed_symptoms)
        
        # Processar tratamentos
        processed_treatments = preprocess_text(row["Tratamento"])
        if processed_treatments:
            processed_sentences.append(processed_treatments)
        
        # Adicionar a doença como um token também, se relevante
        processed_disease = preprocess_text(row["Doenca"])
        if processed_disease:
            processed_sentences.append(processed_disease)
        
    return processed_sentences

if __name__ == "__main__":

    data_file = "data/medical_data.csv" 
    processed_data = process_medical_data(data_file)
    
    print("\nDados pré-processados (primeiras 5 entradas):")
    for i, sentence in enumerate(processed_data[:5]):
        print(f"Entrada {i+1}: {sentence}")
    
    print(f"\nTotal de sentenças processadas: {len(processed_data)}")

    # Salvar os dados pré-processados em um arquivo de texto para uso posterior
    # Cada lista de tokens será salva como uma linha no arquivo, com tokens separados por espaço
    with open("data/processed_medical_text.txt", "w", encoding="utf-8") as f:
        for sentence_list in processed_data:
            f.write(" ".join(sentence_list) + "\n")
    print("Dados pré-processados salvos em data/processed_medical_text.txt")
