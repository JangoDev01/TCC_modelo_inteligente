import json
import re
import unicodedata
import numpy as np
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

# Normalização de texto #

"""
    - Remove acentos, pontuação e converte para minúsculas
    - Retorna uma lista de tokens
    - Exemplo: 
        "Dor de cabeça e febre" -> ["dor", "de", "cabeca", "e", "febre"]

    "unicodedata.normalize('NFD', texto)" :
        - decompoe os caracteres acentuados em seus componentes básicos (letra + acento)
"""
def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return texto.split()

# Vetor de frase (média dos vetores) #

"""
    - Recebe um modelo FastText e uma lista de tokens
    - Retorna o vetor médio dos tokens presentes no modelo
    - Se nenhum token estiver presente, retorna um vetor zero
"""
def sentence_vector(model, tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Carregar base de doenças #
caminho_json = "data/medical_data.json"
def carregar_doencas(caminho_json):
    with open(caminho_json, 'r', encoding='utf-8') as f:
        return json.load(f)

# Preparar vetores das doenças #

"""
    - Para cada doença, cria um vetor médio dos seus sintomas
    - Retorna uma lista de tuplas (doença, vetor)
    - Exemplo: 
        [(doenca1, vetor1), (doenca2, vetor2), ...]
"""
def preparar_base(model, doencas):
    base_vetores = []
    for d in doencas:
        sintomas = d['sintomas']
        tokens = normalizar(' '.join(sintomas))
        vetor = sentence_vector(model, tokens)
        base_vetores.append((d, vetor))
    return base_vetores

# Similaridade híbrida #

"""
    - Combina similaridade vetorial (cosine) com um score de matching direto
    - O score final é a soma da similaridade vetorial e 0.1 vezes o número de tokens em comum
    - Exemplo: 
        se a similaridade vetorial for 0.8 e houver 3 tokens em comum, o score final será 0.8 + (0.1 * 3) = 1.1
    - Essa abordagem ajuda a reforçar diagnósticos que têm uma correspondência direta de sintomas, 
      mesmo que a similaridade vetorial seja moderada
"""
def calcular_score(user_tokens, disease_tokens, user_vec, disease_vec):
    # Similaridade vetorial
    sim = cosine_similarity([user_vec], [disease_vec])[0][0]

    # Score por matching direto
    match = len(set(user_tokens) & set(disease_tokens))

    return sim + (0.1 * match)

# Diagnóstico #

"""
    - Recebe o modelo, a base vetorial das doenças e a entrada do usuário
    - Retorna uma lista dos top_n diagnósticos mais prováveis, ordenados por score
    - Exemplo de retorno:
        [(doenca1, score1), (doenca2, score2), ...]
"""
def diagnosticar(model, base_vetores, input_usuario, top_n=3):
    user_tokens = normalizar(input_usuario)
    user_vec = sentence_vector(model, user_tokens)

    resultados = []

    """
        Para cada doença na base vetorial:
            - Normaliza os sintomas da doença para obter os tokens
            - Calcula o score usando a função calcular_score, que combina similaridade vetorial e matching direto
            - Armazena a doença e seu score na lista de resultados
    """
    for doenca, vetor in base_vetores:
        disease_tokens = normalizar(' '.join(doenca['sintomas']))
        score = calcular_score(user_tokens, disease_tokens, user_vec, vetor)
        resultados.append((doenca, score))

    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados[:top_n]

# Função principal #

if __name__ == "__main__":

    # Carregar dados
    doencas = carregar_doencas(caminho_json)

    # Preparar corpus para treino
    corpus = []

    """
        Para cada doença na base de dados:
            - Junta os sintomas em uma única string
            - Normaliza a string usando a função normalizar, que remove acentos, pontuação e converte para minúsculas
            - Adiciona a lista de tokens normalizados ao corpus, que será usado para treinar o modelo FastText
    """
    for d in doencas:
        texto = ' '.join(d['sintomas'])
        corpus.append(normalizar(texto))

    # Treinar modelo FastText

    """
        - Treina um modelo FastText usando o corpus de sintomas normalizados
        - O modelo é configurado com um vetor de tamanho 100, janela de contexto de 3
          e considera palavras que aparecem pelo menos 1 vez (no mínimo) para incluir no vocabulário
    """
    model = FastText(sentences=corpus, vector_size=100, window=3, min_count=1)

    # Preparar base vetorial
    base_vetores = preparar_base(model, doencas)

    # Input do usuário
    input_user = input("Descreva seus sintomas: ")

    resultados = diagnosticar(model, base_vetores, input_user)

    print("\nPossíveis diagnósticos:\n")

    """
        Para cada doença e seu score nos resultados:
            - Imprime o nome da doença, o score formatado com 4 casas decimais e a descrição (se disponível)
            - Se a descrição não estiver disponível, imprime 'N/A'
            - Imprime uma linha de separação para melhor visualização
    """
    for d, score in resultados:
        print(f"Doença: {d['doenca']}")
        print(f"Score: {score:.4f}")
        print(f"Descrição: {d.get('tratamento', 'N/A')}")
        print("-" * 40) # Separação entre resultados para melhor visualização
